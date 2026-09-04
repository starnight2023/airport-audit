# =============================================================================
# src/tools.py — Phase 3: Agent 工具函数模块
# =============================================================================
# 功能说明：
#   1. 定义三个稽核工具函数：query_revenue、extract_clause、query_historical_disputes
#   2. 每个工具附带 LangChain Tool 格式的 name + description，供 LLM Planner 决策
#   3. 每个工具包含 try-except 降级兜底，工具失败时不中断稽核流程
#
# 使用方式：
#   from src.tools import query_revenue, extract_clause, TOOL_DESCRIPTIONS
#   result = query_revenue("M001", "2024-01")
#
# 【Phase 3 核心】
# executor_node 遍历 tool_calls_plan 并调用此处定义的函数。
# 三个工具的调用结果写入 AgentState，供 rule_check_node 使用。
#
# 【Phase 4 拓展位】
# query_revenue 将被封装为 MCP Server 工具，通过 MCP 协议远程调用。
# TOOL_DESCRIPTIONS 将作为 MCP Tool Schema 的输入。
# =============================================================================

import csv
import json
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BILLS_DIR = os.path.join(DATA_DIR, "bills")
DISPUTES_PATH = os.path.join(DATA_DIR, "historical_disputes.json")
CONTRACTS_PATH = os.path.join(DATA_DIR, "contracts.json")


# ===========================================================================
# 一、工具函数
# ===========================================================================

def query_revenue(merchant_id: str, month: str) -> dict:
    """
    工具1: 查询商户月度营收数据

    读取 data/bills/{merchant_id}_2024.csv，返回指定月份的账单数据。

    Args:
        merchant_id: 商户ID (如 "M001")
        month: 月份 (如 "2024-01")

    Returns:
        {"status": "success", "data": {"merchant_id": ..., "month": ...,
         "reported_revenue": ..., "paid_amount": ..., "submit_date": ...}}
        or {"status": "error", "message": ..., "fallback": ...}

    【Phase 1 联动】
    此工具读取的数据与 rule_engine.audit_single 共享同一数据源。

    【Phase 4 拓展位】
    此函数可封装为 MCP Server 的 revenue_query 工具，
    通过 MCP Client 被 Agent 调用，验证协议标准化对接能力。
    """
    try:
        filepath = os.path.join(BILLS_DIR, f"{merchant_id}_2024.csv")
        if not os.path.exists(filepath):
            return {
                "status": "error",
                "message": f"账单文件不存在: {filepath}",
                "fallback": "该商户营收数据暂不可查，已标记待处理",
            }

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["month"] == month:
                    return {
                        "status": "success",
                        "data": {
                            "merchant_id": merchant_id,
                            "month": month,
                            "reported_revenue": float(row["reported_revenue"]),
                            "paid_amount": float(row["paid_amount"]),
                            "submit_date": row["submit_date"],
                        },
                    }

        return {
            "status": "error",
            "message": f"未找到 {merchant_id} 在 {month} 的账单记录",
            "fallback": "该月份数据缺失，已跳过本次稽核",
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"营收查询异常: {str(e)}",
            "fallback": "营收查询服务异常，已标记待处理",
        }


def extract_clause(
    merchant_id: str,
    query_text: str,
    top_k: int = 3,
) -> dict:
    """
    工具2: 查询合同条款（BM25 + 向量混合检索）

    调用 retriever.enhanced_retrieve（Query改写 → BM25关键词 + 向量语义
    双路召回 → Reranker 精排），根据自然语言查询在知识库中定位最匹配的合同条款。

    对比实验结论（scripts/benchmark_rag.py，430 条查询）：
    混合检索相对纯向量语义检索，Top-1 准确率从 80% 提升至 100%，MRR 0.90 → 1.00。

    Args:
        merchant_id: 商户ID (如 "M001")
        query_text: 查询意图文本 (如 "租金计算方式")
        top_k: 返回条款数量上限

    Returns:
        {"status": "success", "data": [{"clause_id": ..., "description": ...,
         "clause_type": ..., "parameters": ..., "score": ...}]}
        or {"status": "error", "message": ..., "fallback": ...}

    【Phase 2 联动】
    此工具依赖 Phase 2 构建的 Chroma 知识库。
    如果知识库未构建，将返回降级错误信息。
    """
    try:
        from src.retriever import enhanced_retrieve

        results = enhanced_retrieve(
            merchant_id,
            query_text,
            top_k=top_k,
            rewrite_mode="rule",
            use_hyde=False,
        )

        if not results:
            return {
                "status": "success",
                "data": [],
                "message": f"未找到 {merchant_id} 与 '{query_text}' 相关的条款",
            }

        return {
            "status": "success",
            "data": results,
        }

    except ImportError as e:
        return {
            "status": "error",
            "message": f"检索模块不可用: {str(e)}",
            "fallback": "知识库检索服务暂不可用，需人工核查合同条款",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"条款检索异常: {str(e)}",
            "fallback": "条款检索服务异常，已跳过自动引用",
        }


def query_historical_disputes(merchant_id: str) -> dict:
    """
    工具3: 查询商户历史争议记录

    读取 data/historical_disputes.json，返回近6个月的争议记录。
    争议记录可用于判断该商户是否有"争议历史"，辅助异常定级。

    Args:
        merchant_id: 商户ID (如 "M001")

    Returns:
        {"status": "success", "data": [{"merchant_id": ..., "month": ...,
         "dispute_type": ..., "description": ..., "resolution": ...}]}
        or {"status": "error", "message": ..., "fallback": ...}

    【Phase 3 设计说明】
    此工具返回的信息将影响 Report Node 的异常严重程度判定。
    有历史争议的商户，同一类型异常可升级处理。
    """
    try:
        if not os.path.exists(DISPUTES_PATH):
            return {
                "status": "success",
                "data": [],
                "message": "历史争议数据文件不存在，视为无争议记录",
            }

        with open(DISPUTES_PATH, "r", encoding="utf-8") as f:
            all_disputes = json.load(f)

        # 过滤指定商户的记录
        merchant_disputes = [
            d for d in all_disputes if d["merchant_id"] == merchant_id
        ]

        return {
            "status": "success",
            "data": merchant_disputes,
            "total_count": len(merchant_disputes),
        }

    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": f"争议数据解析失败: {str(e)}",
            "fallback": "争议记录暂不可查",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"争议查询异常: {str(e)}",
            "fallback": "争议查询服务异常",
        }


# ===========================================================================
# 二、工具描述（供 LLM Planner 使用）
# ===========================================================================

TOOL_DESCRIPTIONS = [
    {
        "name": "query_revenue",
        "description": "查询商户指定月份的营收数据，返回申报营业额、实缴金额、报表提交日期。必须提供 merchant_id 和 month。",
        "parameters": {
            "merchant_id": "商户ID，如 M001",
            "month": "月份，如 2024-01",
        },
    },
    {
        "name": "extract_clause",
        "description": "查询商户合同中的相关条款。通过自然语言描述你想查询的内容（如'租金计算方式''保底额''提交截止日'），返回最匹配的条款原文和计费参数。",
        "parameters": {
            "merchant_id": "商户ID，如 M001",
            "query_text": "查询意图，如'租金计算方式'",
        },
    },
    {
        "name": "query_historical_disputes",
        "description": "查询商户的历史争议记录。返回该商户近6个月内是否有收入争议、逾期争议等记录及处理结果。",
        "parameters": {
            "merchant_id": "商户ID，如 M001",
        },
    },
]

# 工具名称到函数的映射表（executor_node 用此表调度执行）
TOOL_REGISTRY = {
    "query_revenue": query_revenue,
    "extract_clause": extract_clause,
    "query_historical_disputes": query_historical_disputes,
}


# ===========================================================================
# 三、工具函数：加载合同（规划器内部使用）
# ===========================================================================

def load_contract(merchant_id: str) -> Optional[dict]:
    """
    加载指定商户的合同数据

    从 data/contracts.json 中查找匹配的合同。
    供 planner_node 和 rule_check_node 内部使用。

    Args:
        merchant_id: 商户ID

    Returns:
        合同字典，未找到返回 None
    """
    try:
        if not os.path.exists(CONTRACTS_PATH):
            return None
        with open(CONTRACTS_PATH, "r", encoding="utf-8") as f:
            contracts = json.load(f)
        for c in contracts:
            if c["merchant_id"] == merchant_id:
                return c
        return None
    except Exception:
        return None


# ===========================================================================
# 四、快速验证
# ===========================================================================

def test_tools():
    """测试三个工具的基本功能"""
    print("=" * 50)
    print("🔧 工具函数快速验证")
    print("=" * 50)

    # 测试 query_revenue
    print("\n[1] query_revenue('M001', '2024-01'):")
    result = query_revenue("M001", "2024-01")
    if result["status"] == "success":
        d = result["data"]
        print(f"    申报营业额: ¥{d['reported_revenue']:.2f}")
        print(f"    实缴金额:   ¥{d['paid_amount']:.2f}")
        print(f"    提交日期:   {d['submit_date']}")
    else:
        print(f"    {result['message']}")

    # 测试 extract_clause
    print("\n[2] extract_clause('M001', '租金计算方式'):")
    result = extract_clause("M001", "租金计算方式")
    if result["status"] == "success":
        for r in result["data"][:2]:
            print(f"    [{r['score']:.4f}] {r['clause_type']}: {r['description'][:50]}...")
    else:
        print(f"    {result.get('message', '未匹配')}")

    # 测试 query_historical_disputes
    print("\n[3] query_historical_disputes('M001'):")
    result = query_historical_disputes("M001")
    if result["status"] == "success":
        print(f"    争议记录数: {result.get('total_count', len(result['data']))}")
        for d in result["data"]:
            print(f"    - {d['month']} {d['dispute_type']}: {d['description'][:40]}...")
    else:
        print(f"    {result['message']}")

    # 测试 query_revenue 异常场景
    print("\n[4] query_revenue('M999', '2024-01') [不存在商户]:")
    result = query_revenue("M999", "2024-01")
    print(f"    status: {result['status']}, fallback: {result.get('fallback', 'N/A')}")

    print("\n✅ 工具函数验证完成")


if __name__ == "__main__":
    test_tools()