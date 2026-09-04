# =============================================================================
# evaluation/consistency.py — Report Consistency（报告与规则引擎一致性）
# =============================================================================
# 原则：LLM 负责解释，Rule Engine 负责事实。规则引擎输出视为权威 ground truth。
#
# 对每个 case 枚举 5 个关键事实，逐项核对报告：
#   status / revenue / payable / paid / difference
# 每个事实的状态：
#   correct  —— 报告陈述且与 ground truth 一致
#   wrong    —— 报告陈述但与 ground truth 矛盾（contradiction）
#   missing  —— 报告未陈述该事实
#
# report_consistency = 1 - wrong / 事实总数
#   （一致性只惩罚"矛盾"，不惩罚"遗漏"；遗漏由 claim_recall 度量）
#
# 数字事实比较使用与规则引擎一致的容差口径（ratio 容差 + 最小金额）。
# 差额按绝对值比对（不惩罚"多缴/少缴"方向措辞，属已知局限）。
# =============================================================================

from typing import Optional

NUMERIC_FACTS = ("revenue", "payable", "paid", "difference")


def _within_tolerance(a: float, b: float, tolerance: float, min_amt: float) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(tolerance * abs(b), min_amt)


def check_report_consistency(
    ground_truth: dict,
    extracted: dict,
    tolerance: float = 0.01,
    min_anomaly_amount: float = 0.01,
) -> dict:
    """
    核对报告与 ground truth 的关键事实一致性。

    Args:
        ground_truth: dataset.build_ground_truth 的输出
        extracted: claim_extractor.extract_claims 的输出
        tolerance / min_anomaly_amount: 金额比对容差

    Returns:
        dict: {
            "facts": [{"fact", "expected", "reported", "state", "contradiction"}],
            "correct": int, "wrong": int, "missing": int,
            "report_consistency": float,   # 1 - wrong/total
            "coverage": float,             # correct/(correct+wrong+missing) = claim_recall 贡献
        }
    """
    facts = []

    # ---- status 事实 ----
    expected_status = ground_truth.get("status")
    reported_status = extracted.get("status")
    if reported_status is None:
        state = "missing"
    elif reported_status == expected_status:
        state = "correct"
    else:
        state = "wrong"
    facts.append({
        "fact": "status",
        "expected": expected_status,
        "reported": reported_status,
        "state": state,
        "contradiction": state == "wrong",
    })

    # ---- 金额类事实 ----
    claim_values = {c["claim_type"]: c["value"] for c in extracted.get("claims", [])}
    for fact in NUMERIC_FACTS:
        expected = ground_truth.get(fact)
        reported = claim_values.get(fact)
        if expected is None:
            # ground truth 缺失（如营收数据缺失），不参与核对
            facts.append({
                "fact": fact, "expected": None, "reported": reported,
                "state": "n/a", "contradiction": False,
            })
            continue
        if reported is None:
            state = "missing"
        elif _within_tolerance(
            # 差额按绝对值比对：direction 已由锚定关键词（少缴/多缴）表达，
            # 提取值不含方向，带符号比较会把方向措辞差异误判为矛盾（评测发现 C018）
            abs(float(reported)) if fact == "difference" else float(reported),
            abs(float(expected)) if fact == "difference" else float(expected),
            tolerance, min_anomaly_amount,
        ):
            state = "correct"
        else:
            state = "wrong"
        facts.append({
            "fact": fact,
            "expected": expected,
            "reported": reported,
            "state": state,
            "contradiction": state == "wrong",
        })

    # 过滤掉 n/a 的事实
    scored = [f for f in facts if f["state"] != "n/a"]
    total = len(scored)
    correct = sum(1 for f in scored if f["state"] == "correct")
    wrong = sum(1 for f in scored if f["state"] == "wrong")
    missing = sum(1 for f in scored if f["state"] == "missing")

    return {
        "facts": facts,
        "correct": correct,
        "wrong": wrong,
        "missing": missing,
        "report_consistency": round(1 - wrong / total, 4) if total > 0 else 0.0,
        "coverage": round(correct / total, 4) if total > 0 else 0.0,
    }
