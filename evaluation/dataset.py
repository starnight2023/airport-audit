# =============================================================================
# evaluation/dataset.py — 评测数据集构建
# =============================================================================
# 目标：
#   自动构建 600 条 case 的 Ground Truth 与 required_evidence，不人工标注。
#
# 事实源（权威，确定性）：
#   1. audit_mock() 输出的 audit_result（规则引擎复算）→ status / issues
#   2. bill CSV / query_revenue → reported_revenue / paid_amount
#   3. contracts.json → 合同参数
#   4. truth_labels.json → 数据生成时注入的异常标注（仅用于确定性评测指标）
#
# 关键假设（与规则引擎口径一致）：
#   - payable（应缴金额）= 规则引擎复算的 expected_amount
#   - difference = payable - paid（应缴 - 实缴，与 LLM 报告口径一致）
#   - required_evidence 仅含条款证据（clause_id）：本项目无 bill_id / payment_id
#     命名空间（营收与实缴同属一条账单 CSV 记录），故 Completeness 在条款维度计算。
# =============================================================================

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRUTH_LABELS_PATH = os.path.join(DATA_DIR, "truth_labels.json")
CONTRACTS_PATH = os.path.join(DATA_DIR, "contracts.json")

# 边界补充集（合成账单，覆盖主数据集缺失的边界维度，见 data/ 下 *_boundary 文件）
CONTRACTS_BOUNDARY_PATH = os.path.join(DATA_DIR, "contracts_boundary.json")
BILLS_BOUNDARY_PATH = os.path.join(DATA_DIR, "bills_boundary.csv")

# 账单月份：50 商户 × 12 月 = 600 条（与 evaluate.run_full_evaluation 口径一致）
DEFAULT_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]

# 异常类型映射（truth_label → 中文）
ANOMALY_TYPE_MAP = {
    "金额不符": "amount_mismatch",
    "逾期提交": "late_submission",
    "少报营业额": "underreported_revenue",
}

ISSUE_TYPE_ZH = {
    "amount_mismatch": "金额不符",
    "late_submission": "逾期提交",
    "underreported_revenue": "少报营业额",
}


def load_contracts() -> list[dict]:
    with open(CONTRACTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_truth_labels() -> list[dict]:
    if not os.path.exists(TRUTH_LABELS_PATH):
        return []
    with open(TRUTH_LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_truth_label_index(truth_labels: list[dict]) -> dict:
    """(merchant_id, month) -> anomaly_type"""
    idx = {}
    for lbl in truth_labels:
        idx[(lbl["merchant_id"], lbl["month"])] = lbl.get("anomaly_type", "")
    return idx


def _compute_payable(contract: dict, reported_revenue: float, paid_amount: float) -> float:
    """
    计算规则引擎复算的应缴金额。
    优先取审计 issue 中的 expected_value（已有金额异常时），否则用 check_paid_amount 复算。
    纯确定性，不调用 LLM。
    """
    from src.rule_engine import check_paid_amount
    _, expected, _ = check_paid_amount(contract, reported_revenue, paid_amount)
    return round(float(expected), 2)


def build_ground_truth(audit_mock_result: dict) -> dict:
    """
    从 audit_mock 输出构建 ground truth（规则引擎为权威事实源）。

    Returns:
        dict: {
            "status": normal/abnormal/error,
            "revenue": 申报营业额 | None,
            "payable": 应缴金额 | None,
            "paid":    实缴金额 | None,
            "difference": payable - paid | None,
            "reason": ["金额不符", ...],
            "used_clause_ids": [规则引擎实际引用的条款ID],
            "source": "rule_engine",
        }
    """
    contract = audit_mock_result.get("contract") or {}
    revenue_data = audit_mock_result.get("revenue_data")
    ar = audit_mock_result.get("audit_result") or {}
    issues = ar.get("issues", []) or []
    status = ar.get("status", "error")

    revenue = paid = payable = diff = None
    if revenue_data:
        revenue = round(float(revenue_data.get("reported_revenue", 0)), 2)
        paid = round(float(revenue_data.get("paid_amount", 0)), 2)

    if paid is not None and contract:
        am_issue = next((i for i in issues if i.get("issue_type") == "amount_mismatch"), None)
        if am_issue and am_issue.get("expected_value") is not None:
            payable = round(float(am_issue["expected_value"]), 2)
        else:
            payable = _compute_payable(contract, revenue or 0.0, paid)
        diff = round(payable - paid, 2)  # 应缴 - 实缴

    # 审计证据条款 = 规则引擎实际核实的条款（金额 + 提交日期），而非仅被标记异常的条款
    used_clause_ids = _audit_evidence_clause_ids(contract, revenue_data, audit_mock_result.get("month", ""))
    reason = [ISSUE_TYPE_ZH.get(i.get("issue_type", ""), i.get("issue_type", ""))
              for i in issues if i.get("issue_type")]

    return {
        "status": status,
        "revenue": revenue,
        "payable": payable,
        "paid": paid,
        "difference": diff,
        "reason": reason,
        "used_clause_ids": used_clause_ids,
        "source": "rule_engine",
    }


def _default_clause_ids(contract: dict) -> list[str]:
    """规则引擎对任意 case 的金额校验都会引用 clause-001（见 rule_engine.check_paid_amount）。"""
    cid = contract.get("contract_id", "CTR")
    return [f"{cid}-clause-001"]


def _audit_evidence_clause_ids(contract: dict, revenue_data: dict, month: str) -> list[str]:
    """
    规则引擎"实际核实的条款"（审计证据依据），而非仅被标记异常的条款。

    规则引擎对每个 case 固定执行两道校验，各返回一个 clause_id：
      1. 金额校验   → check_paid_amount   返回 {contract_id}-clause-001
      2. 提交日期校验 → check_submission_date 返回 clause-002（拼接为 {contract_id}-clause-002）

    报告结论（含"无异常"）依据这两道校验得出，故 required_evidence 应为二者的并集；
    否则报告引用日期条款（支撑"未逾期"结论）会被误判为过度引用（评测发现，precision 虚低）。
    """
    cid = contract.get("contract_id", "CTR")
    if not revenue_data:
        return _default_clause_ids(contract)

    from src.rule_engine import check_paid_amount, check_submission_date
    try:
        _, _, amount_cid = check_paid_amount(
            contract,
            float(revenue_data.get("reported_revenue", 0) or 0),
            float(revenue_data.get("paid_amount", 0) or 0),
        )
        submit = revenue_data.get("submit_date")
        if submit:
            _, _, date_cid = check_submission_date(submit, month)
            return sorted({amount_cid, f"{cid}-{date_cid}"})
        return [amount_cid]
    except Exception:  # noqa: BLE001 — 推导失败回退金额条款
        return _default_clause_ids(contract)


def build_cases(
    max_cases: int | None = None,
    merchants: list[str] | None = None,
    months: list[str] | None = None,
    verbose: bool = False,
) -> list[dict]:
    """
    批量构建评测 case。

    Args:
        max_cases: 限制 case 数量（默认全部）
        merchants: 限制商户
        months: 限制月份
        verbose: 打印进度

    Returns:
        list[dict]: 每个 case 包含 input / ground_truth / required_evidence /
                    contract / revenue_data / clause_data / truth_label。
    """
    from src.agent_graph import audit_mock

    contracts = load_contracts()
    truth_index = build_truth_label_index(load_truth_labels())
    months = months or DEFAULT_MONTHS

    merchant_ids = [c["merchant_id"] for c in sorted(contracts, key=lambda x: x["merchant_id"])]
    if merchants:
        merchant_ids = [m for m in merchant_ids if m in merchants]

    cases = []
    count = 0
    for mid in merchant_ids:
        for month in months:
            if max_cases and count >= max_cases:
                return cases
            count += 1

            try:
                result = audit_mock(mid, month)
            except Exception as e:
                if verbose:
                    print(f"  ⚠ audit_mock 失败: {mid} {month}: {e}")
                cases.append({
                    "case_id": f"C{count:03d}",
                    "input": {"merchant_id": mid, "month": month},
                    "status": "error",
                    "error": f"audit_mock 异常: {e}",
                })
                continue

            if "error" in result:
                cases.append({
                    "case_id": f"C{count:03d}",
                    "input": {"merchant_id": mid, "month": month},
                    "status": "error",
                    "error": result["error"],
                })
                continue

            contract = result.get("contract") or {}
            gt = build_ground_truth(result)
            used_ids = gt["used_clause_ids"] or _default_clause_ids(contract)

            truth_type = truth_index.get((mid, month), "")

            cases.append({
                "case_id": f"C{count:03d}",
                "input": {"merchant_id": mid, "month": month},
                "contract_type": result.get("contract_type", "unknown"),
                "contract": contract,
                "revenue_data": result.get("revenue_data"),
                "clause_data": result.get("clause_data", []),
                "issues": result.get("issues", []),          # report_node 需要的原始异常
                "audit_result": result.get("audit_result"),  # report_node 需要的规则引擎结果
                "ground_truth": gt,
                "required_evidence": used_ids,
                "truth_label": {
                    "anomaly": bool(truth_type),
                    "anomaly_type": truth_type,
                },
                "status": "ok",
            })

    return cases


def get_single_case(merchant_id: str, month: str) -> dict | None:
    """单 case 调试：构建指定 (商户, 月份) 的一个 case。"""
    cases = build_cases(merchants=[merchant_id], months=[month])
    return cases[0] if cases else None


# ---------------------------------------------------------------------------
# 边界补充集（合成账单）
# ---------------------------------------------------------------------------
# 目的：主数据集 600 条在边界维度上存在覆盖缺口（金额容差 1% 附近、提交宽限期
# 1~3 天、零营业额、缺失提交日期、多缴）。边界集用合成账单补齐这些维度，供评测
# 检验规则引擎在阈值附近的判定是否正确。
#
# 口径与主集一致：
#   - ground_truth 由规则引擎（audit_single）实算，不人工硬编码；
#   - truth_label 取自 CSV 的 expected_status（领域业务语义真值，独立于规则引擎）
#     —— 这正是边界集的价值：对照"业务语义预期"检验规则引擎在边界处的判定。
#   若规则引擎在阈值附近把边界判错，确定性指标（anomaly 混淆矩阵）会如实暴露。

BOUNDARY_SCENARIO_TYPE = {
    "amount": "金额不符",
    "submit": "逾期提交",
    "zero_revenue": "金额不符",
    "missing_submit": "逾期提交",
    "overpaid": "金额不符",
    "fixed": "金额不符",
    "hybrid": "金额不符",
}


def _boundary_anomaly_type(scenario: str) -> str:
    """按 scenario 前缀映射 anomaly_type（与主集 truth_label 的异常类型口径一致）。"""
    for key, zh in BOUNDARY_SCENARIO_TYPE.items():
        if scenario.startswith(key):
            return zh
    return ""


def build_boundary_cases(verbose: bool = False) -> list[dict]:
    """
    构建边界补充集 case（200 条合成账单，覆盖主数据集缺失的边界维度）。

    数据源：data/contracts_boundary.json + data/bills_boundary.csv。
    事实源：audit_single（规则引擎）实算 ground truth；
    truth_label 来自 CSV 的 expected_status（领域真值，独立于规则引擎）。
    输出 case 结构与 build_cases 完全一致，可无缝走 evaluate_case / aggregate。
    """
    import csv

    from src.rule_engine import CONFIG as RULE_CONFIG
    from src.rule_engine import audit_single

    if not os.path.exists(BILLS_BOUNDARY_PATH):
        print(f"  ⚠ 边界补充集数据缺失（未找到 {BILLS_BOUNDARY_PATH}），返回空集")
        return []

    contracts = {}
    if os.path.exists(CONTRACTS_BOUNDARY_PATH):
        with open(CONTRACTS_BOUNDARY_PATH, "r", encoding="utf-8") as f:
            contracts = {c["merchant_id"]: c for c in json.load(f)}

    with open(BILLS_BOUNDARY_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cases = []
    for idx, row in enumerate(rows, 1):
        mid = row["merchant_id"]
        month = row["month"]
        contract = contracts.get(mid) or {}
        revenue_data = {
            "merchant_id": mid,
            "month": month,
            "reported_revenue": float(row["reported_revenue"] or 0),
            "paid_amount": float(row["paid_amount"] or 0),
            "submit_date": row["submit_date"].strip() or None,  # 空串 → None（缺失提交日期）
        }

        try:
            ar = audit_single(
                contract,
                revenue_data["reported_revenue"],
                revenue_data["paid_amount"],
                revenue_data["submit_date"] or "",
                month,
                mid,
                RULE_CONFIG,
            )
        except Exception as e:  # noqa: BLE001
            cases.append({
                "case_id": f"B{idx:03d}",
                "input": {"merchant_id": mid, "month": month},
                "status": "error",
                "error": f"audit_single 异常: {e}",
            })
            continue

        result_dict = ar.to_dict()
        # clause_data 用边界合同自身的条款列表填充：与主集 extract_clause(top_k=3) 返回的
        # 条款一致（clause_id/clause_type/description 均具备），保证 report_node 把全部
        # 3 个条款 ID 暴露给 LLM 供 contract_refs 引用（空列表会让 LLM 无从引用日期条款）。
        clause_data = contract.get("clauses", []) or []
        mock_result = {
            "merchant_id": mid,
            "month": month,
            "contract_type": contract.get("type", ""),
            "contract": contract,
            "revenue_data": revenue_data,
            "clause_data": clause_data,
            "audit_result": result_dict,
            "issues": result_dict.get("issues", []),
            "summary": result_dict.get("summary", ""),
        }
        gt = build_ground_truth(mock_result)
        used_ids = gt["used_clause_ids"] or _default_clause_ids(contract)

        scenario = row.get("scenario", "")
        is_abnormal = row["expected_status"] == "abnormal"
        # anomaly_type：正常账单一律 ""（金额核对指标据此判断无金额异常）；
        # abnormal 账单优先取 CSV 显式标注，缺失时按 scenario 前缀兜底推断。
        anomaly_type = row.get("expected_anomaly_type", "") or (
            _boundary_anomaly_type(scenario) if is_abnormal else ""
        )
        cases.append({
            "case_id": f"B{idx:03d}",
            "input": {"merchant_id": mid, "month": month},
            "contract_type": contract.get("type", "unknown"),
            "contract": contract,
            "revenue_data": revenue_data,
            "clause_data": clause_data,
            "issues": result_dict.get("issues", []),
            "audit_result": result_dict,
            "ground_truth": gt,
            "required_evidence": used_ids,
            "truth_label": {
                "anomaly": is_abnormal,
                "anomaly_type": anomaly_type,
            },
            "boundary_scenario": scenario,
            "status": "ok",
        })

    if verbose:
        print(f"  🏗  构建边界补充集: {len(cases)} cases")
    return cases


if __name__ == "__main__":
    cases = build_cases(max_cases=3, verbose=True)
    for c in cases:
        print(json.dumps({
            "case_id": c.get("case_id"),
            "input": c.get("input"),
            "gt": c.get("ground_truth"),
            "required_evidence": c.get("required_evidence"),
            "truth_label": c.get("truth_label"),
        }, ensure_ascii=False, indent=2))
