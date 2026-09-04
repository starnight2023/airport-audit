# =============================================================================
# src/rule_engine.py — 第一阶段：确定性规则校验引擎
# =============================================================================
# 核心思想：一个"大号计算器"，用代码写死的公式做金额复算，
#           不依赖 LLM，同一个输入永远得到同一个结果。
#
# 功能说明：
#   1. 加载合同 JSON 和账单 CSV
#   2. 逐行校验每张账单：
#      - fixed:      paid_amount == fixed_amount ?
#      - commission: paid_amount == reported_revenue × commission_rate ?
#      - hybrid:     paid_amount == max(min_guarantee, reported_revenue × commission_rate) ?
#      - 所有类型：submit_date 是否在截止日前 ?
#   3. 输出结构化稽核报告（每项异常关联合同条款编号）
#   4. 统计并打印分布
#
# 运行方式：
#   python src/rule_engine.py                          # 对所有商户执行稽核
#   python src/rule_engine.py --merchant M001          # 仅对单个商户执行
#   python src/rule_engine.py --month 2024-01          # 仅对指定月份
#
# 拓展接口（供后续阶段调用）：
#   - audit_single(contract, bill) → AuditResult        # Phase 3 Agent 的 rule_check_node
#   - audit_merchant(merchant_id) → AuditResult          # Phase 4 FastAPI
#   - batch_audit() → list[AuditResult]                  # Phase 5 批量评测
#
# 面试问答准备：
#   Q: "为什么把规则引擎和LLM解耦？"
#   A: 金额计算要求 100% 准确，LLM 有幻觉风险。
#      规则引擎用代码写死公式，保证每次复算可追溯、可验证。
#      LLM 只做决策调度和异常解释，不碰数字计算。
# =============================================================================

import csv
import json
import os
import sys
from datetime import datetime, date
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# 项目路径设置（兼容各种运行方式）
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 引入项目内模块
try:
    from models import (
        ContractType,
        AuditResult,
        AuditIssue,
        CheckStep,
        AuditStatus,
        IssueSeverity,
        month_to_datetime_str,
        next_month,
    )
except ImportError:
    # 备用：当 models 模块不可用时使用的内联定义
    AuditResult = dict
    AuditIssue = dict
    CheckStep = dict

# 路径常量
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BILLS_DIR = os.path.join(DATA_DIR, "bills")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
CONTRACTS_PATH = os.path.join(DATA_DIR, "contracts.json")
RULES_CONFIG_PATH = os.path.join(CONFIG_DIR, "rules.yaml")


# ===========================================================================
# 一、配置加载
# ===========================================================================

def load_rules_config(path: str = RULES_CONFIG_PATH) -> dict:
    """
    加载 YAML 规则配置

    【为什么用 YAML？】
    - 修改阈值（如逾期天数、滞纳金比例）只需改 YAML，不改代码
    - 非技术人员也能理解和维护
    - 配合环境变量可做到"配置即代码"

    Returns:
        dict: 规则配置字典
    """
    if not os.path.exists(path):
        print(f"⚠ 规则配置文件不存在: {path}，将使用默认配置")
        return {
            "submission": {"deadline_day": 5, "grace_days": 3},
            "validation": {"amount_tolerance": 0.01, "min_anomaly_amount": 0.01},
        }
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


CONFIG = load_rules_config()


# ===========================================================================
# 二、核心校验函数
# ===========================================================================

def _get_tolerance(config: dict) -> float:
    """获取金额比对的浮点容差"""
    return config.get("validation", {}).get("amount_tolerance", 0.01)


def _get_min_anomaly(config: dict) -> float:
    """获取最小异常金额阈值"""
    return config.get("validation", {}).get("min_anomaly_amount", 0.01)


def _get_deadline_day(config: dict) -> int:
    """获取提交截止日"""
    return config.get("submission", {}).get("deadline_day", 5)


def _get_grace_days(config: dict) -> int:
    """获取宽限期"""
    return config.get("submission", {}).get("grace_days", 3)


def check_paid_amount(
    contract: dict,
    reported_revenue: float,
    paid_amount: float,
    tolerance: Optional[float] = None,
    min_anomaly: Optional[float] = None,
    config: Optional[dict] = None,
) -> tuple[bool, float, str]:
    """
    核心函数：校验 paid_amount 是否与合同条款一致

    Args:
        contract: 合同字典
        reported_revenue: 商户申报营业额
        paid_amount: 商户实缴金额
        tolerance: 容差比例（默认从配置加载）
        min_anomaly: 最小异常金额（默认从配置加载）
        config: 规则配置字典（用于容差取值）

    Returns:
        (is_match, expected_amount, clause_id)
        is_match: True 表示金额一致（正常）
        expected_amount: 合同计算的应有金额
        clause_id: 引用的条款编号

    公式说明：
        fixed:      expected = fixed_amount
        commission: expected = reported_revenue × commission_rate
        hybrid:     expected = max(min_guarantee, reported_revenue × commission_rate)

    【Phase 2 拓展位】
    此处的 clause_id 关联到合同条款的 clause_id，
    未来 Agent 拿到这个 ID 可以从向量库检索条款原文进行解释。
    """
    if config is None:
        config = CONFIG
    if tolerance is None:
        tolerance = _get_tolerance(config)
    if min_anomaly is None:
        min_anomaly = _get_min_anomaly(config)

    ctype = contract["type"]
    clause_id = f"{contract['contract_id']}-clause-001"

    # ---- 根据合同类型计算应有金额 ----
    if ctype == "fixed":
        expected_amount = contract.get("fixed_amount", 0.0)
    elif ctype == "commission":
        rate = contract.get("commission_rate", 0.0)
        expected_amount = round(reported_revenue * rate, 2)
    elif ctype == "hybrid":
        rate = contract.get("commission_rate", 0.0)
        min_guarantee = contract.get("min_guarantee", 0.0)
        commission_part = round(reported_revenue * rate, 2)
        expected_amount = max(min_guarantee, commission_part)
    else:
        return False, 0.0, clause_id

    # ---- 比对实缴金额与应有金额 ----
    diff = abs(paid_amount - expected_amount)
    if expected_amount > 0:
        diff_ratio = diff / expected_amount
    else:
        diff_ratio = 0.0 if diff == 0 else 1.0

    is_match = diff_ratio <= tolerance or diff <= min_anomaly

    return is_match, expected_amount, clause_id


def check_submission_date(
    submit_date_str: str,
    month: str,
    deadline_day: Optional[int] = None,
    grace_days: Optional[int] = None,
    config: Optional[dict] = None,
) -> tuple[bool, str, str]:
    """
    校验报表是否按时提交

    Args:
        submit_date_str: 提交日期 ("2024-02-03")
        month: 账单月份 ("2024-01")
        deadline_day: 截止日（默认从配置加载）
        grace_days: 宽限期（默认从配置加载）
        config: 规则配置字典

    Returns:
        (is_on_time, deadline_str, clause_id)
        is_on_time: True 表示按时提交
        deadline_str: 截止日期原文（含宽限期）
        clause_id: 引用的条款编号

    【Phase 3 拓展位】
    Agent 的 rule_check_node 将调用此函数。
    逾期信息将写入 AuditIssue 的 description。
    【Phase 5 拓展位】
    Streamlit 将展示 deadline_str 助用户理解。
    """
    if config is None:
        config = CONFIG
    if deadline_day is None:
        deadline_day = _get_deadline_day(config)
    if grace_days is None:
        grace_days = _get_grace_days(config)

    # 从 month 推断截止日（下月 deadline_day 日）
    year_s, month_s = month.split("-")
    bill_year = int(year_s)
    bill_month = int(month_s)
    next_y, next_m = next_month(bill_year, bill_month)

    deadline = date(next_y, next_m, deadline_day)
    deadline_with_grace = date(next_y, next_m, min(28, deadline_day + grace_days))

    try:
        submit_date = datetime.strptime(submit_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False, str(deadline), "clause-002"

    is_on_time = submit_date <= deadline_with_grace
    clause_id = "clause-002"

    return is_on_time, str(deadline_with_grace), clause_id


def audit_single(
    contract: dict,
    reported_revenue: float,
    paid_amount: float,
    submit_date_str: str,
    month: str,
    merchant_id: str,
    config: Optional[dict] = None,
) -> AuditResult:
    """
    对单张账单执行稽核校验（一条完整规则校验流水线）

    Args:
        contract: 合同字典
        reported_revenue: 申报营业额
        paid_amount: 实缴金额
        submit_date_str: 提交日期
        month: 账单月份
        merchant_id: 商户 ID
        config: 规则配置（可选）

    Returns:
        AuditResult 对象

    【Phase 3 核心拓展位】
    此函数将作为 LangGraph 的 rule_check_node 被调用：
    state = {
        "merchant_id": ...,
        "month": ...,
        "contract_type": ...,
        "revenue_data": {...},    # 来自 executor_node
        "clause_data": {...},     # 来自 executor_node
    }
    rule_check_node 调用此函数产生产出 issues 列表。

    返回的 AuditResult 中的 steps 列表记录每个检查步骤，
    Phase 5 的 Streamlit 将其渲染为分步展示面板。
    """
    if config is None:
        config = CONFIG

    issues = []
    steps = []
    status = AuditStatus.NORMAL

    # ===== Step 1: 识别合同类型 =====
    ctype = contract["type"]
    steps.append(CheckStep(
        step_name="合同类型识别",
        status="success",
        detail=f"合同 {contract['contract_id']} | 类型: {ctype} | "
               f"商户: {contract['merchant_name']} ({merchant_id})",
        data={
            "contract_id": contract["contract_id"],
            "contract_type": ctype,
            "merchant_name": contract["merchant_name"],
            "merchant_id": merchant_id,
        },
    ))

    # ===== Step 2: 金额校验 =====
    is_match, expected_amount, amount_clause_id = check_paid_amount(
        contract, reported_revenue, paid_amount, config=config
    )
    if is_match:
        steps.append(CheckStep(
            step_name="金额校验",
            status="success",
            detail=f"实缴金额 ¥{paid_amount:.2f} = 应有金额 ¥{expected_amount:.2f} ✓",
            data={
                "reported_revenue": reported_revenue,
                "paid_amount": paid_amount,
                "expected_amount": expected_amount,
                "match": True,
            },
        ))
    else:
        diff = abs(paid_amount - expected_amount)
        steps.append(CheckStep(
            step_name="金额校验",
            status="error",
            detail=(
                f"实缴金额 ¥{paid_amount:.2f} ≠ 应有金额 ¥{expected_amount:.2f} "
                f"(差异: ¥{diff:.2f}) ✗"
            ),
            data={
                "reported_revenue": reported_revenue,
                "paid_amount": paid_amount,
                "expected_amount": expected_amount,
                "match": False,
                "diff": round(diff, 2),
            },
        ))
        issues.append(AuditIssue(
            merchant_id=merchant_id,
            month=month,
            issue_type="amount_mismatch",
            clause_id=amount_clause_id,
            expected_value=expected_amount,
            actual_value=paid_amount,
            description=(
                f"金额不匹配：实缴 ¥{paid_amount:.2f}，"
                f"合同计算应有 ¥{expected_amount:.2f}"
            ),
            severity="high",
        ))

    # ===== Step 3: 提交日期校验 =====
    is_on_time, deadline, date_clause_id = check_submission_date(
        submit_date_str, month, config=config
    )
    if is_on_time:
        steps.append(CheckStep(
            step_name="提交日期校验",
            status="success",
            detail=f"提交日期 {submit_date_str} ≤ 截止日 {deadline} ✓",
            data={
                "submit_date": submit_date_str,
                "deadline": deadline,
                "on_time": True,
            },
        ))
    else:
        steps.append(CheckStep(
            step_name="提交日期校验",
            status="error",
            detail=f"提交日期 {submit_date_str} > 截止日 {deadline} (含宽限期) ✗",
            data={
                "submit_date": submit_date_str,
                "deadline": deadline,
                "on_time": False,
            },
        ))
        issues.append(AuditIssue(
            merchant_id=merchant_id,
            month=month,
            issue_type="late_submission",
            clause_id=f"{contract['contract_id']}-{date_clause_id}",
            expected_value=0.0,
            actual_value=0.0,
            description=f"逾期提交：提交日 {submit_date_str}，截止日（含宽限期）{deadline}",
            severity="medium",
        ))

    # ===== Step 4: 营业额波动检查（辅助检测少报） =====
    # 注：Phase 1 仅做简单提示，Phase 3 会通过历史数据做更精确的检测
    # 当前跳过这一步，保留接口

    # ===== 汇总 =====
    status = AuditStatus.ABNORMAL if issues else AuditStatus.NORMAL
    issue_count = len(issues)
    issue_types = set(i.issue_type for i in issues)
    summary_parts = [f"商户 {merchant_id} {month} 稽核完成。"]

    if status == AuditStatus.NORMAL:
        summary_parts.append("所有校验通过，无异常。")
    else:
        summary_parts.append(f"发现 {issue_count} 项异常：{', '.join(issue_types)}。")

    return AuditResult(
        merchant_id=merchant_id,
        month=month,
        contract_type=ctype,
        status=status,
        issues=issues,
        steps=steps,
        summary=" ".join(summary_parts),
    )


# ===========================================================================
# 三、批量稽核
# ===========================================================================

def audit_merchant(
    merchant_id: str,
    contracts: list[dict],
    bills_dir: str = BILLS_DIR,
    config: Optional[dict] = None,
) -> list[AuditResult]:
    """
    对指定商户的全部月份执行稽核

    【Phase 3 拓展位】
    Agent 的 audit() 函数调用此函数（或包装后暴露）。
    """
    # 查找合同
    contract = None
    for c in contracts:
        if c["merchant_id"] == merchant_id:
            contract = c
            break
    if contract is None:
        print(f"⚠ 未找到商户 {merchant_id} 的合同")
        return []

    # 加载账单
    bills_path = os.path.join(bills_dir, f"{merchant_id}_2024.csv")
    if not os.path.exists(bills_path):
        print(f"⚠ 未找到账单文件: {bills_path}")
        return []

    results = []
    with open(bills_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = audit_single(
                contract=contract,
                reported_revenue=float(row["reported_revenue"]),
                paid_amount=float(row["paid_amount"]),
                submit_date_str=row["submit_date"],
                month=row["month"],
                merchant_id=merchant_id,
                config=config,
            )
            results.append(result)

    return results


def batch_audit(
    contracts: list[dict],
    bills_dir: str = BILLS_DIR,
    merchant_id: Optional[str] = None,
    month: Optional[str] = None,
    config: Optional[dict] = None,
) -> list[AuditResult]:
    """
    批量稽核全部（或按条件过滤）账单

    Args:
        contracts: 合同列表
        bills_dir: 账单文件目录
        merchant_id: 可选，仅稽核指定商户
        month: 可选，仅稽核指定月份
        config: 规则配置

    Returns:
        list[AuditResult]: 稽核结果列表

    【Phase 4 拓展位】
    此函数将作为 FastAPI /audit 接口的底层实现。
    """
    results = []

    # 获取要稽核的商户列表
    merchant_ids = set()
    for c in contracts:
        mid = c["merchant_id"]
        if merchant_id and mid != merchant_id:
            continue
        merchant_ids.add(mid)

    for mid in sorted(merchant_ids):
        merchant_results = audit_merchant(mid, contracts, bills_dir, config)
        for r in merchant_results:
            if month and r.month != month:
                continue
            results.append(r)

    return results


# ===========================================================================
# 四、统计与报告
# ===========================================================================

def print_audit_summary(results: list[AuditResult]):
    """打印稽核结果统计摘要"""
    total = len(results)
    abnormal = sum(1 for r in results if r.status == AuditStatus.ABNORMAL)
    normal = total - abnormal

    # 异常类型分布
    issue_type_counts = {}
    for r in results:
        for i in r.issues:
            issue_type_counts[i.issue_type] = issue_type_counts.get(i.issue_type, 0) + 1

    # 异常商户列表
    abnormal_merchants = sorted(set(
        r.merchant_id for r in results if r.status == AuditStatus.ABNORMAL
    ))

    print("\n" + "=" * 60)
    print("📋 稽核结果统计")
    print("=" * 60)
    print(f"  总账单数:      {total}")
    print(f"  正常账单:      {normal} ({normal/total*100:.1f}% 通过率)")
    print(f"  异常账单:      {abnormal} ({abnormal/total*100:.1f}% 异常率)")
    print(f"  异常类型分布: {issue_type_counts}")
    print(f"  涉及异常商户: {abnormal_merchants}")
    print("=" * 60)


def export_audit_report(
    results: list[AuditResult],
    output_path: Optional[str] = None,
) -> str:
    """
    导出完整的稽核报告 JSON

    Args:
        results: 稽核结果列表
        output_path: 保存路径（可选）

    Returns:
        报告 JSON 字符串

    【Phase 5 拓展位】
    Streamlit 的"导出报告"按钮将调用此函数。
    报告的格式与 Phase 3 Agent 的输出格式保持一致。
    """
    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "total_bills": len(results),
            "normal_count": sum(1 for r in results if r.status == AuditStatus.NORMAL),
            "abnormal_count": sum(1 for r in results if r.status == AuditStatus.ABNORMAL),
        },
        "results": [r.to_dict() for r in results],
    }

    json_str = json.dumps(report, ensure_ascii=False, indent=2)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"✓ 稽核报告已导出: {output_path}")

    return json_str


# ===========================================================================
# 五、命令行入口
# ===========================================================================

def main():
    """命令行主入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="机场非航收入智能稽核系统 — 规则引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python src/rule_engine.py                           # 全量稽核
  python src/rule_engine.py --merchant M001           # 单个商户
  python src/rule_engine.py --month 2024-01           # 指定月份
  python src/rule_engine.py --export report.json      # 导出报告
        """,
    )
    parser.add_argument("--merchant", type=str, help="仅稽核指定商户ID (如 M001)")
    parser.add_argument("--month", type=str, help="仅稽核指定月份 (如 2024-01)")
    parser.add_argument("--export", type=str, help="导出报告到指定路径 (如 report.json)")
    parser.add_argument("--config", type=str, default=RULES_CONFIG_PATH, help="规则配置文件路径")

    args = parser.parse_args()

    # 加载配置和合同
    config = load_rules_config(args.config)
    contracts = json.load(open(CONTRACTS_PATH, "r", encoding="utf-8"))

    print(f"🔍 规则引擎启动 | 商户过滤: {args.merchant or '全部'} | 月份过滤: {args.month or '全部'}")

    # 执行稽核
    results = batch_audit(
        contracts=contracts,
        bills_dir=BILLS_DIR,
        merchant_id=args.merchant,
        month=args.month,
        config=config,
    )

    # 打印统计
    print_audit_summary(results)

    # 打印详细异常
    print("\n📝 异常明细:")
    for r in results:
        if r.status == AuditStatus.ABNORMAL:
            for issue in r.issues:
                print(f"  [{issue.severity.upper()}] {issue.description}")

    # 导出报告
    if args.export:
        export_audit_report(results, args.export)

    # 返回结果列表（供编程调用）
    return results


if __name__ == "__main__":
    main()
