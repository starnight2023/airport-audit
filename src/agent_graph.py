# =============================================================================
# src/agent_graph.py — Phase 3: Agent 工作流编排（LangGraph 状态机）
# =============================================================================
# 核心思路：
#   用一个四节点状态机实现稽核流程，只调用 2 次 LLM（Planner + Report），
#   中间的 Executor 和 RuleCheck 全走确定性代码，消除金额幻觉风险。
#
# 状态机结构：
#   [START] → Planner(LLM) → Executor(代码) → RuleCheck(代码) → Report(LLM) → [END]
#         ↑ 决定调哪些工具      ↑ 执行工具调用      ↑ 规则引擎复算       ↑ 生成报告
#           1 次 LLM 调用        0 次 LLM 调用      0 次 LLM 调用       1 次 LLM 调用
#
# 运行方式（需要 DeepSeek API Key）：
#   在项目根目录创建 .env 文件：
#     DEEPSEEK_API_KEY="sk-xxxx"
#     DEEPSEEK_BASE_URL="https://api.deepseek.com"
#     DEEPSEEK_MODEL="deepseek-chat"
#   然后运行：
#     python -c "from src.agent_graph import audit; print(audit('M001', '2024-01'))"
#
# 运行方式（Mock 模式，无需 API Key）：
#   python -c "from src.agent_graph import audit_mock; print(audit_mock('M001', '2024-01'))"
#
# 【面试问答准备】
# Q: "为什么用 LangGraph 而不用原生 Python 循环？"
# A: LangGraph 提供状态图的可观测性——每一步的输入输出都可记录、可断点恢复，
#    支持节点级别的编排而非函数级别的调用。生产级稽核流程比手写 if-else 更可靠。
#
# Q: "怎么保证 Agent 不会无限循环调用工具？"
# A: 状态机只有 4 个节点，Planner→Executor→RuleCheck→Report 是直线路径，
#    没有循环边，天然没有无限循环风险。最大调用步数上限为 1。
# =============================================================================

import json
import os
import operator
import sys
from typing import Any, Optional, Annotated

# ---------------------------------------------------------------------------
# 项目路径 & 环境变量
# ---------------------------------------------------------------------------
# 加载 .env 文件（优先级：手动 export > .env 文件 > 默认值）
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# LangGraph 导入
# ---------------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional as OptionalType


# ===========================================================================
# 一、AgentState — 状态定义
# ===========================================================================

class AgentState(TypedDict):
    """
    稽核 Agent 的完整状态

    每个节点从 state 中读取需要的字段，写入自己产出的字段。
    LangGraph 自动合并各节点的返回值为完整状态。

    【Phase 3 → Phase 4 拓展位】
    此状态结构可直接序列化为 JSON，作为 FastAPI /audit 的响应体。
    """
    # -------- 输入字段（由 audit() 函数传入） --------
    merchant_id: str                          # 商户ID
    month: str                                # 稽核月份

    # -------- Planner 节点产出 --------
    contract: Optional[dict]                  # 合同信息（从 contracts.json 加载）
    tool_calls_plan: Optional[List[dict]]     # 工具调用计划 [{"tool": str, "args": dict}]
    planner_reasoning: Optional[str]          # Planner 的决策理由

    # -------- Executor 节点产出 --------
    revenue_data: Optional[dict]              # query_revenue 结果
    clause_data: Optional[List[dict]]         # extract_clause 结果
    disputes: Optional[List[dict]]           # query_historical_disputes 结果
    tool_errors: List[str]                   # 工具调用中的错误记录

    # -------- RuleCheck 节点产出 --------
    audit_result: Optional[dict]             # 规则引擎复算结果（AuditResult 的 dict）
    issues: List[dict]                       # 异常列表（从 audit_result 提取）

    # -------- Report 节点产出 --------
    report: Optional[dict]                   # 最终稽核报告

    # -------- 运行轨迹（调试用） --------
    # 使用 Annotated + operator.add 实现自动合并追加，而非替换覆盖
    trace: Annotated[List[dict], operator.add]  # 每个节点的执行记录


# ===========================================================================
# 二、LLM 客户端封装
# ===========================================================================

def _load_llm_config() -> tuple:
    """
    从环境变量加载 LLM 配置

    优先级：手动 export > .env 文件 > 默认值
    .env 文件由项目根目录的 dotenv 自动加载。

    Returns:
        (api_key, base_url, model) 三元组
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise ValueError(
            "未设置 DEEPSEEK_API_KEY 环境变量。\n"
            "请在项目根目录创建 .env 文件（已提供模板）：\n"
            "  DEEPSEEK_API_KEY=sk-xxxx\n"
            "  DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
            "  DEEPSEEK_MODEL=deepseek-chat\n"
            "或者通过命令行设置：\n"
            "  export DEEPSEEK_API_KEY='sk-xxxx'\n"
            "如需测试 Mock 模式，请使用 agent_graph.audit_mock()"
        )
    return api_key, base_url, model


def _get_llm_client():
    """
    获取 LLM API 客户端

    从环境变量读取配置（支持 deepseek / openai / 其他兼容服务）。
    兼容 OpenAI SDK 的调用方式。

    Returns:
        (openai.OpenAI 客户端实例, model 名称)
    """
    api_key, base_url, model = _load_llm_config()
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url), model


def _call_llm(messages: list[dict], temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """
    调用 LLM API（兼容 OpenAI 格式）

    支持 DeepSeek / OpenAI / 其他兼容 API，通过环境变量配置：
      DEEPSEEK_API_KEY — API Key
      DEEPSEEK_BASE_URL — API 地址（默认 https://api.deepseek.com）
      DEEPSEEK_MODEL — 模型名称（默认 deepseek-chat）

    Args:
        messages: 消息列表 [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        temperature: 生成温度（0=最确定性，1=最随机）
        max_tokens: 最大生成 token 数

    Returns:
        模型生成的文本内容

    【Phase 3 设计说明】
    使用 temperature=0.1 而非默认 0.7，保证 Planner 和 Report 的输出
    尽可能稳定、可复现。稽核场景需要的是可靠性而非创造性。
    """
    client, model = _get_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},  # 强制 JSON 输出
    )
    return response.choices[0].message.content


# ===========================================================================
# 三、节点：Planner（LLM 调用 #1）
# ===========================================================================

PLANNER_SYSTEM_PROMPT = """你是一个机场非航收入智能稽核系统的规划员。
你的职责是：根据商户的合同类型，决定本次稽核需要调用哪些工具来收集数据。

可用的工具有 3 个：
1. query_revenue(merchant_id, month)
   - 查询商户指定月份的营收数据
   - 返回：申报营业额、实缴金额、报表提交日期
   - 所有合同类型都必需此工具

2. extract_clause(merchant_id, query_text)
   - 查询商户合同中的相关条款
   - 返回：最匹配的条款原文、条款类型、计费参数（提成比例、保底额等）
   - 提成合同和保底合同必需此工具（需要获取计费参数）
   - 固定租金合同可选（金额已经在合同中写明）

3. query_historical_disputes(merchant_id)
   - 查询商户的历史争议记录
   - 返回：近6个月的争议记录和处理结果
   - 可选：如果该商户有争议历史，需要调此工具

不同合同类型的工具建议：
- fixed（固定租金）：必须调 query_revenue，可选调 extract_clause
- commission（营业额提成）：必须调 query_revenue 和 extract_clause（查提成比例）
- hybrid（保底+提成）：必须调 query_revenue 和 extract_clause（查保底额和提成比例）

请输出 JSON 格式（不要加 markdown 代码块标记）：
{{"reasoning": "简要说明为什么需要这些工具", "tool_calls": [{{"tool": "工具名称", "args": {{"参数名": "参数值"}}}}]}}"""


def planner_node(state: AgentState) -> dict:
    """
    Planner 节点：调用 LLM 决定工具调用计划

    读取 state 中的 merchant_id、month 和合同信息，
    生成一个工具调用计划写入 state.tool_calls_plan。

    这是状态机中第一次（也是唯一一次）LLM 决策调用。
    """
    merchant_id = state["merchant_id"]
    month = state["month"]

    # 加载合同信息
    from src.tools import load_contract
    contract = load_contract(merchant_id)
    if contract is None:
        return {
            "contract": None,
            "tool_calls_plan": [],
            "planner_reasoning": "未找到该商户的合同信息，无法规划稽核",
            "trace": [{"node": "planner", "status": "error", "detail": f"商户 {merchant_id} 合同不存在"}],
        }

    contract_type = contract["type"]
    merchant_name = contract.get("merchant_name", "")

    # 构建提示词
    user_prompt = f"""商户信息：
- 商户ID: {merchant_id}
- 商户名称: {merchant_name}
- 合同类型: {contract_type}
- 稽核月份: {month}

请根据合同类型，决定需要调用哪些工具，输出 JSON 格式的工具调用计划。"""

    try:
        response = _call_llm([
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        plan = json.loads(response)
        tool_calls = plan.get("tool_calls", [])
        reasoning = plan.get("reasoning", "")

        return {
            "contract": contract,
            "tool_calls_plan": tool_calls,
            "planner_reasoning": reasoning,
            "trace": [{
                "node": "planner",
                "status": "success",
                "detail": f"合同类型: {contract_type}，计划调用 {len(tool_calls)} 个工具",
                "reasoning": reasoning,
                "tool_calls": tool_calls,
            }],
        }

    except Exception as e:
        # LLM 调用失败时的降级方案：按合同类型走默认路径
        print(f"⚠ Planner LLM 调用失败: {e}，使用默认规划")
        default_tools = _default_plan(contract)
        return {
            "contract": contract,
            "tool_calls_plan": default_tools,
            "planner_reasoning": f"LLM 规划失败，使用默认规划（{e}）",
            "trace": [{
                "node": "planner",
                "status": "degraded",
                "detail": f"LLM 调用失败，启用默认规划，调用 {len(default_tools)} 个工具",
            }],
        }


def _default_plan(contract: dict) -> list[dict]:
    """
    LLM 调用失败时的降级规划方案

    不依赖 LLM，纯按合同类型规则决定工具调用。
    """
    ctype = contract["type"]
    mid = contract["merchant_id"]
    plans = []
    plans.append({"tool": "query_revenue", "args": {"merchant_id": mid, "month": ""}})
    if ctype in ("commission", "hybrid"):
        plans.append({"tool": "extract_clause", "args": {"merchant_id": mid, "query_text": "租金计算方式"}})
    return plans


# ===========================================================================
# 四、节点：Executor（纯代码执行）
# ===========================================================================

def executor_node(state: AgentState) -> dict:
    """
    Executor 节点：执行工具调用计划

    遍历 state.tool_calls_plan，逐个调用 TOOL_REGISTRY 中注册的函数。
    工具失败时记录错误并继续下一工具（不中断流程）。

    这是状态机中唯一执行实际数据获取的节点。
    0 次 LLM 调用，全部确定性代码。
    """
    from src.tools import TOOL_REGISTRY

    plan = state.get("tool_calls_plan", [])
    merchant_id = state["merchant_id"]
    month = state["month"]

    revenue_data = None
    clause_data = None
    disputes = None
    errors = []
    trace_items = []

    if not plan:
        trace_items.append({
            "node": "executor",
            "status": "warning",
            "detail": "工具调用计划为空，跳过执行",
        })
        return {
            "revenue_data": None,
            "clause_data": None,
            "disputes": None,
            "tool_errors": ["工具调用计划为空"],
            "trace": trace_items,
        }

    for item in plan:
        tool_name = item.get("tool", "")
        args = item.get("args", {})

        if tool_name not in TOOL_REGISTRY:
            errors.append(f"未知工具: {tool_name}")
            trace_items.append({
                "node": "executor",
                "status": "error",
                "detail": f"未知工具: {tool_name}",
            })
            continue

        # 自动注入 merchant_id 和 month（如果 args 为空）
        if not args.get("merchant_id"):
            args["merchant_id"] = merchant_id
        if not args.get("month") and tool_name == "query_revenue":
            args["month"] = month

        try:
            func = TOOL_REGISTRY[tool_name]
            result = func(**args)

            if result["status"] == "success":
                # 根据工具类型存入对应字段
                if tool_name == "query_revenue":
                    revenue_data = result["data"]
                elif tool_name == "extract_clause":
                    clause_data = result["data"]
                elif tool_name == "query_historical_disputes":
                    disputes = result["data"]

                trace_items.append({
                    "node": "executor",
                    "status": "success",
                    "detail": f"工具 {tool_name} 调用成功",
                    "tool": tool_name,
                })
            else:
                errors.append(f"{tool_name}: {result.get('message', '')}")
                trace_items.append({
                    "node": "executor",
                    "status": "error",
                    "detail": f"工具 {tool_name} 调用失败: {result.get('message', '')}",
                    "fallback": result.get("fallback", ""),
                })

        except Exception as e:
            errors.append(f"{tool_name}: {str(e)}")
            trace_items.append({
                "node": "executor",
                "status": "error",
                "detail": f"工具 {tool_name} 抛出异常: {str(e)}",
                "fallback": "工具执行异常，该数据缺失",
            })

    return {
        "revenue_data": revenue_data,
        "clause_data": clause_data,
        "disputes": disputes,
        "tool_errors": errors,
        "trace": trace_items,
    }


# ===========================================================================
# 五、节点：RuleCheck（调用 Phase 1 规则引擎）
# ===========================================================================

def rule_check_node(state: AgentState) -> dict:
    """
    RuleCheck 节点：调用确定性规则引擎做金额复算

    读取 state.revenue_data（来自 query_revenue 的返回），
    调用 rule_engine.audit_single 执行金额和日期校验，
    产出 issues 列表。

    这是状态机中确保"金额计算零幻觉"的关键节点。
    0 次 LLM 调用，全部确定性代码。
    """
    from src.rule_engine import audit_single, load_rules_config
    from src import rule_engine as re_mod

    merchant_id = state["merchant_id"]
    month = state["month"]
    contract = state.get("contract")
    revenue_data = state.get("revenue_data")

    # 如果营收数据缺失，无法执行规则校验
    if not revenue_data:
        config = load_rules_config()
        # 尝试获取合同
        from src.tools import load_contract
        if contract is None:
            contract = load_contract(merchant_id)

        result = {
            "merchant_id": merchant_id,
            "month": month,
            "contract_type": contract["type"] if contract else "unknown",
            "status": "error",
            "issues": [],
            "steps": [{
                "step_name": "规则引擎",
                "status": "skip",
                "detail": "营收数据缺失，跳过规则引擎复算",
                "data": {},
            }],
            "summary": f"商户 {merchant_id} {month} 因营收数据缺失，无法执行规则引擎复算",
        }

        return {
            "audit_result": result,
            "issues": [],
            "trace": [{
                "node": "rule_check",
                "status": "skip",
                "detail": "营收数据缺失，跳过规则校验",
            }],
        }

    if contract is None:
        return {
            "audit_result": None,
            "issues": [],
            "trace": [{"node": "rule_check", "status": "error", "detail": "合同数据缺失"}],
        }

    try:
        config = load_rules_config()
        result = audit_single(
            contract=contract,
            reported_revenue=revenue_data["reported_revenue"],
            paid_amount=revenue_data["paid_amount"],
            submit_date_str=revenue_data["submit_date"],
            month=month,
            merchant_id=merchant_id,
            config=config,
        )

        # 转为字典
        result_dict = result.to_dict()
        issues = result_dict.get("issues", [])

        return {
            "audit_result": result_dict,
            "issues": issues,
            "trace": [{
                "node": "rule_check",
                "status": "success",
                "detail": f"规则引擎复算完成，发现 {len(issues)} 项异常",
                "issue_count": len(issues),
                "audit_status": result_dict.get("status", ""),
            }],
        }

    except Exception as e:
        return {
            "audit_result": {
                "merchant_id": merchant_id,
                "month": month,
                "status": "error",
                "issues": [],
                "summary": f"规则引擎执行异常: {str(e)}",
            },
            "issues": [],
            "trace": [{
                "node": "rule_check",
                "status": "error",
                "detail": f"规则引擎执行异常: {str(e)}",
            }],
        }


# ===========================================================================
# 六、节点：Report（LLM 调用 #2）
# ===========================================================================

REPORT_SYSTEM_PROMPT = """你是一个机场非航收入稽核系统的报告分析师。
你的职责是：根据规则引擎的复算结果和工具收集的数据，生成一份可直接发送给机场运营管理人员的稽核报告。

要求：
1. 用清晰易懂的业务语言描述稽核发现，避免使用"amount_mismatch"等技术术语
2. 必须显式陈述四个数字：申报营业额、应缴金额、实缴金额、差额（差额 = 应缴金额 - 实缴金额）。
   无异常 case 也必须全部写出这四个数字；差额为 0 写 "¥0.00"；禁止用"实缴金额一致"这类不含数字的表述代替。
   差额必须写真实算术值：应缴与实缴不等时，即使差异在容差范围内（金额校验判定为通过/正常），
   也要写出真实差额数字，不得写成 0 —— "容差内通过"不等于"差额为 0"
3. 涉及金额的计算过程要写清楚——引用了哪条条款、合同参数是什么、怎么算的、差额多少
4. 必须同时说明"规则引擎逐道校验明细"中两道校验的结论：
   金额校验（应缴金额 vs 实缴金额是否一致）与提交日期校验（提交日期、截止日、是否按时提交）。
   即使某道校验结果为正常/无异常，也必须写明该道结论，禁止只报告出异常的那一道
5. 给出有业务价值的处理建议——涉及数字的写清楚具体金额
6. 如果有风险趋势（如连续异常、历史争议），做简要提示
7. 最终结论要明确，让读者一眼看明白"有没有问题、问题多严重、接下来怎么办"
8. 报告中出现的金额只能来自"营收数据"、"规则引擎复算结果"与"逐道校验明细"；
   规则引擎未计算滞纳金/罚款金额时，禁止自行推算（如用日费率 × 逾期天数算出滞纳金），
   也禁止出现任何来源中没有的金额；确需说明时写"滞纳金金额需另行核定"

注意：这份报告的直接读者是机场运营管理人员，非技术人员，请确保语言简洁专业。

请输出 JSON 格式（不要加 markdown 代码块标记）：
{{"summary": "稽核结论概述——必须包含申报营业额、应缴金额、实缴金额、差额四个数字（无异常 case 差额写 ¥0.00）和主要发现",
  "status": "normal/abnormal/error",
  "findings": [{{"type": "异常类型（金额不匹配/逾期提交）",
                 "severity": "high/medium/low",
                 "description": "异常描述——包含具体金额、合同参数、计算过程",
                 "evidence": "判断依据——包含条款编号和原文"}}],
  "suggestion": "处理建议——异常场景下应包含具体操作步骤和金额",
  "contract_refs": ["列出报告结论所依据的全部条款ID。
    本系统对每张账单都会执行金额与提交日期两道校验，因此 contract_refs 应同时包含
    金额校验条款 与 提交日期校验条款（如 \\"CTR-001-clause-001\\"、\\"CTR-001-clause-002\\"），
    除非某道校验因数据缺失确实未执行；
    必须使用数据中给出的条款ID、不得自造或使用条款类型名；确实没有引用则为 []"]}}"""


def report_node(state: AgentState) -> dict:
    """
    Report 节点：调用 LLM 生成结构化稽核报告

    读取 state 中所有已收集的数据和规则引擎结果，
    输出一份人类可读的稽核报告。

    这是状态机中第二次（也是最后一次）LLM 调用。
    """
    merchant_id = state["merchant_id"]
    month = state["month"]
    contract = state.get("contract", {})
    revenue_data = state.get("revenue_data", {})
    clause_data = state.get("clause_data", [])
    issues = state.get("issues", [])
    audit_result = state.get("audit_result", {})
    tool_errors = state.get("tool_errors", [])

    # 构建用户提示词
    contract_type = contract.get("type", "unknown") if contract else "unknown"
    merchant_name = contract.get("merchant_name", merchant_id) if contract else merchant_id

    issues_text = "无"
    if issues:
        issues_lines = []
        for i, issue in enumerate(issues, 1):
            issues_lines.append(
                f"  {i}. [{issue.get('severity', 'medium')}] "
                f"{issue.get('description', '')} "
                f"(条款: {issue.get('clause_id', '')})"
            )
        issues_text = "\n".join(issues_lines)

    clauses_text = "无"
    if clause_data:
        # 必须带上 clause_id（如 "CTR-001-clause-001"）—— 否则 LLM 无法在 contract_refs 中
        # 引用真实条款，只能拿 clause_type 名称编造，导致引用指标失真（评测发现）。
        clauses_text = "\n".join(
            f"  [{c.get('clause_type', '')}] 条款ID: {c.get('clause_id', '')} {c.get('description', '')[:60]}"
            for c in clause_data[:3]
        )

    revenue_text = "未获取到"
    if revenue_data:
        revenue_text = (
            f"申报营业额: ¥{revenue_data.get('reported_revenue', 0):.2f}, "
            f"实缴金额: ¥{revenue_data.get('paid_amount', 0):.2f}, "
            f"提交日期: {revenue_data.get('submit_date', '')}"
        )

    # 合同参数
    contract_params_text = ""
    if contract:
        if contract_type == "fixed":
            contract_params_text = f"固定月租金: ¥{contract.get('fixed_amount', 0):,.2f}"
        elif contract_type == "commission":
            contract_params_text = f"提成比例: {contract.get('commission_rate', 0)*100:.2f}%"
        elif contract_type == "hybrid":
            mg = contract.get('min_guarantee', 0)
            rt = contract.get('commission_rate', 0)
            th = round(mg / rt, 2) if rt > 0 else 0
            contract_params_text = f"保底额: ¥{mg:,.2f}, 提成比例: {rt*100:.2f}%, 达标线: ¥{th:,.2f}"

    # 规则引擎逐道校验明细（金额 + 提交日期）—— 两道校验对每张账单都会执行。
    # 必须暴露给 LLM：否则报告会只围绕"异常的那道"写并只引用对应条款，
    # 漏掉另一道（即使它正常），导致 Citation Completeness 虚低（评测发现）。
    check_steps = [
        s for s in (audit_result.get("steps") or [])
        if s.get("step_name") in ("金额校验", "提交日期校验")
    ]
    steps_text = "\n".join(
        f"  - {s.get('step_name')}: {s.get('detail', '')}"
        for s in check_steps
    ) if check_steps else "  无"

    user_prompt = f"""商户：{merchant_name}（{merchant_id}）
月份：{month}
合同类型：{contract_type}
合同参数：{contract_params_text}

营收数据：
{revenue_text}

合同条款（top）：
{clauses_text}

规则引擎复算结果状态：{audit_result.get('status', 'unknown')}

规则引擎逐道校验明细：
{steps_text}

发现异常（{len(issues)}项）：
{issues_text}

工具调用错误（{len(tool_errors)}项）：
{chr(10).join(f'  - {e}' for e in tool_errors) if tool_errors else '  无'}

请根据以上数据生成稽核报告。"""

    try:
        response = _call_llm([
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        report = json.loads(response)

        return {
            "report": report,
            "trace": [{
                "node": "report",
                "status": "success",
                "detail": f"报告生成完成，状态: {report.get('status', 'unknown')}",
            }],
        }

    except Exception as e:
        # LLM 失败时的降级报告
        fallback_report = {
            "summary": f"商户 {merchant_name} {month} 稽核完成（降级模式）",
            "status": audit_result.get("status", "error") if audit_result else "error",
            "findings": [{
                "type": i.get("issue_type", "unknown"),
                "severity": i.get("severity", "medium"),
                "description": i.get("description", ""),
                "evidence": f"合同条款 {i.get('clause_id', '')}",
            } for i in issues],
            "suggestion": "因报告生成服务异常，以上为规则引擎原始结果",
            "contract_refs": list(set(i.get("clause_id", "") for i in issues)),
        }

        return {
            "report": fallback_report,
            "trace": [{
                "node": "report",
                "status": "degraded",
                "detail": f"LLM 报告生成失败 ({e})，使用降级报告",
            }],
        }


# ===========================================================================
# 七、构建与编译状态机
# ===========================================================================

def build_audit_graph() -> StateGraph:
    """
    构建稽核 Agent 状态机

    四个节点一条直线路径：
    Planner(LLM) → Executor(代码) → RuleCheck(代码) → Report(LLM) → END

    Returns:
        编译好的 StateGraph，可通过 .invoke() 执行
    """
    builder = StateGraph(AgentState)

    # 注册四个节点
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("rule_check", rule_check_node)
    builder.add_node("report", report_node)

    # 连接边
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "rule_check")
    builder.add_edge("rule_check", "report")
    builder.add_edge("report", END)

    return builder.compile()


# 默认全局编译实例
_default_graph = None


def get_audit_graph() -> StateGraph:
    """获取全局编译好的状态机实例（单例）"""
    global _default_graph
    if _default_graph is None:
        _default_graph = build_audit_graph()
    return _default_graph


# ===========================================================================
# 八、入口函数
# ===========================================================================

def audit(merchant_id: str, month: str) -> dict:
    """
    完整的稽核入口函数

    调用状态机执行完整稽核流程：Plan → Execute → RuleCheck → Report。
    返回包含所有中间数据和最终报告的字典。

    Args:
        merchant_id: 商户ID (如 "M001")
        month: 月份 (如 "2024-01")

    Returns:
        dict: 包含 contract、revenue_data、clause_data、issues、report 等字段

    Raises:
        ValueError: 环境变量 DEEPSEEK_API_KEY 未设置（检查 .env 文件或系统环境变量）

    【Phase 4 拓展位】
    此函数将被 FastAPI 的 /audit 端点调用。
    audit_batch 将循环调用此函数实现批量稽核。
    """
    graph = get_audit_graph()
    result = graph.invoke({
        "merchant_id": merchant_id,
        "month": month,
        # 以下字段初始化为空，由各节点填充
        "contract": None,
        "tool_calls_plan": None,
        "planner_reasoning": None,
        "revenue_data": None,
        "clause_data": None,
        "disputes": None,
        "tool_errors": [],
        "audit_result": None,
        "issues": [],
        "report": None,
        "trace": [],
    })
    return result


def audit_mock(merchant_id: str, month: str) -> dict:
    """
    Mock 模式稽核入口（无需 API Key）

    跳过 LLM 调用的 Planner 和 Report 节点，
    使用默认规划和降级报告，仅执行代码逻辑的 Executor 和 RuleCheck。

    Args:
        merchant_id: 商户ID
        month: 月份

    Returns:
        dict: 稽核结果（不含 LLM 生成的报告）

    使用场景：
    - 开发和调试阶段，验证规则引擎和工具调用是否正确
    - 没有 DeepSeek API Key 时验证系统功能
    """
    from src.tools import load_contract, query_revenue, extract_clause

    # 1. 加载合同
    contract = load_contract(merchant_id)
    if contract is None:
        return {"error": f"商户 {merchant_id} 合同不存在"}

    # 2. 默认工具调用
    revenue_data = query_revenue(merchant_id, month)
    clause_data = extract_clause(merchant_id, "租金计算方式")

    # 3. 规则引擎
    from src.rule_engine import audit_single, load_rules_config
    config = load_rules_config()

    if revenue_data["status"] == "success":
        rd = revenue_data["data"]
        audit_result = audit_single(
            contract=contract,
            reported_revenue=rd["reported_revenue"],
            paid_amount=rd["paid_amount"],
            submit_date_str=rd["submit_date"],
            month=month,
            merchant_id=merchant_id,
            config=config,
        )
        result_dict = audit_result.to_dict()
    else:
        result_dict = {
            "status": "error",
            "issues": [],
            "summary": f"营收数据获取失败: {revenue_data.get('message', '')}",
        }

    # 4. 组装结果（不调 LLM 生成报告）
    return {
        "merchant_id": merchant_id,
        "month": month,
        "contract_type": contract.get("type", ""),
        "contract": contract,
        "revenue_data": revenue_data.get("data") if revenue_data["status"] == "success" else None,
        "clause_data": clause_data.get("data") if clause_data["status"] == "success" else [],
        "audit_result": result_dict,
        "issues": result_dict.get("issues", []),
        "summary": result_dict.get("summary", ""),
        "mode": "mock",
    }


# ===========================================================================
# 九、命令行演示
# ===========================================================================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="机场非航收入智能稽核 Agent")
    parser.add_argument("--merchant", type=str, default="M001", help="商户ID")
    parser.add_argument("--month", type=str, default="2024-01", help="月份")
    parser.add_argument("--mock", action="store_true", help="Mock 模式（无需 API Key）")
    parser.add_argument("--verbose", action="store_true", help="打印完整状态")
    args = parser.parse_args()

    print(f"🚀 机场非航收入智能稽核 Agent")
    print(f"   商户: {args.merchant} | 月份: {args.month} | 模式: {'Mock' if args.mock else 'LLM'}")
    print("=" * 50)

    if args.mock:
        result = audit_mock(args.merchant, args.month)
    else:
        try:
            result = audit(args.merchant, args.month)
        except ValueError as e:
            print(f"❌ {e}")
            print("提示: 使用 --mock 参数可运行 Mock 模式")
            return

    if "error" in result:
        print(f"❌ 稽核失败: {result['error']}")
        return

    # 打印摘要
    print(f"\n📋 稽核摘要")
    print(f"   商户: {result.get('merchant_name', args.merchant)} ({args.merchant})")
    print(f"   合同类型: {result.get('contract_type', 'unknown')}")
    print(f"   月份: {args.month}")

    audit_result = result.get("audit_result", {})
    status = audit_result.get("status", "N/A") if audit_result else "N/A"
    issues = result.get("issues", [])
    print(f"   稽核状态: {status}")
    print(f"   异常数: {len(issues)}")

    if args.verbose:
        print(f"\n📊 详细数据")
        print(f"   营收: {result.get('revenue_data', {})}")
        print(f"   条款: {result.get('clause_data', [])}")
        print(f"   异常: {issues}")
        report = result.get("report", {})
        if report:
            print(f"   报告: {json.dumps(report, ensure_ascii=False, indent=2)}")

    if issues:
        print(f"\n⚠️  异常明细:")
        for issue in issues:
            print(f"  [{issue.get('severity', 'N/A').upper()}] {issue.get('description', '')}")

    # 终端友好输出
    print(f"\n✅ 稽核完成（耗时取决于 LLM 调用）")


# ===========================================================================
# 十、自然语言解析 — 将一句话转换为结构化稽核参数
# ===========================================================================

NL_PARSE_SYSTEM_PROMPT = """你是一个机场稽核系统的自然语言解析器。
请将用户的自然语言需求解析为结构化的稽核参数。

商户列表（ID 和名称对照）：
{merchant_list}

可识别的月份格式：
- "今年1月"、"一月"、"1月" → {reference_year}年1月
- "去年12月" → 上一年12月
- "2025-01"、"2025年1月" → 指定年月
- "上个月" → 当前日期的前一月
- "这个月"、"本月" → 当前月

多月份检测：
- 如果用户提到多个时间段（"5月和6月"、"最近三个月"、"上半年"、"第一季度"），month 字段输出 "multi" 并在 error_type 标记为 "multi_month"
- 如果商户名称不在列表中，error_type 标记为 "merchant_not_found"
- 如果商户能匹配但月份超出数据范围（系统数据为2025年全年），error_type 标记为 "month_out_of_range"
- 如果商户名称模糊匹配到多个可能，error_type 标记为 "merchant_ambiguous"，并在 merchant_candidates 中列出可能商户（逗号分隔）
- 如果输入与稽核无关，error_type 标记为 "no_match"

请输出 JSON 格式（不要加 markdown 代码块标记）：
{{"merchant_id": "匹配到的商户ID",
  "merchant_name": "匹配到的商户名称",
  "month": "解析出的月份 YYYY-MM，多月份或无法解析时填对应说明",
  "intent": "用户意图",
  "confidence": 置信度 0~1,
  "error_type": "成功时填空字符串，失败时按上面规则标记",
  "error_reason": "面向终端用户的失败原因说明，成功时为空字符串",
  "merchant_candidates": "商户名称模糊匹配时的候选列表，无则空字符串"}}"""


def parse_nl_query(nl_text: str, reference_year: Optional[int] = None) -> dict:
    """
    自然语言解析：将一句话转为稽核参数

    调用 DeepSeek API 从自然语言中提取商户ID和月份。
    例如 "帮我看一下云松咖啡今年1月的租金" → {"merchant_id": "M001", "month": "2024-01"}

    Args:
        nl_text: 用户的自然语言输入
        reference_year: 参考年份，默认当前年份

    Returns:
        dict: {
            "merchant_id": str | None,   # 解析出的商户ID
            "merchant_name": str | None, # 解析出的商户名称
            "month": str | None,         # 解析出的月份 (YYYY-MM)
            "intent": str,               # 用户意图
            "confidence": float,          # 置信度 0~1
            "error_reason": str,          # 失败原因
            "raw_query": str,             # 原始输入
        }
    """
    from datetime import datetime

    if reference_year is None:
        reference_year = datetime.now().year

    # 构建商户列表文本（从 contracts.json 动态加载，不硬编码商户ID）
    try:
        from src.tools import load_contract
        import json as _json
        _contracts_path = os.path.join(PROJECT_ROOT, "data", "contracts.json")
        with open(_contracts_path, "r", encoding="utf-8") as _f:
            _all_contracts = _json.load(_f)
        merchant_lines = [
            f"  {c['merchant_id']} - {c['merchant_name']}（{c['type']}合同）"
            for c in sorted(_all_contracts, key=lambda x: x["merchant_id"])
        ]
        merchant_text = "\n".join(merchant_lines)
    except Exception:
        merchant_text = "  M001 - 云松咖啡\n  M002 - 麦香餐厅\n  M003 - 星野小馆\n  M004 - 青竹咖啡"

    system_prompt = NL_PARSE_SYSTEM_PROMPT.format(
        merchant_list=merchant_text,
        reference_year=reference_year,
    )

    # 空/过短输入预处理
    if not nl_text or len(nl_text.strip()) < 3:
        return {
            "merchant_id": None, "merchant_name": None, "month": None,
            "intent": "audit", "confidence": 0.0,
            "error_type": "empty_input",
            "error_reason": "请输入查询内容",
            "raw_query": nl_text or "",
        }

    user_prompt = f"参考年份: {reference_year}\n用户输入: {nl_text}"

    try:
        response = _call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.1, max_tokens=300)

        result = json.loads(response)

        # 补充字段
        result["raw_query"] = nl_text
        if "confidence" not in result:
            result["confidence"] = 0.5
        if "error_type" not in result or not result["error_type"]:
            result["error_type"] = ""
        if "merchant_candidates" not in result:
            result["merchant_candidates"] = ""

        return result

    except Exception as e:
        return {
            "merchant_id": None, "merchant_name": None, "month": None,
            "intent": "audit", "confidence": 0.0,
            "error_type": "no_match",
            "error_reason": "暂时无法理解您的查询，请尝试使用下拉选择或重新描述",
            "raw_query": nl_text,
        }


def is_deepseek_available() -> bool:
    """检查 DeepSeek API Key 是否已配置"""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    return len(key) > 0 and key.startswith("sk-")


if __name__ == "__main__":
    main()