# =============================================================================
# evaluation/metrics.py — 指标聚合
# =============================================================================
# 两类指标：
#   deterministic —— 规则引擎自身的确定性表现（对照 truth_labels 注入异常）
#        amount_accuracy：应缴金额核对准确率（排除"少报营业额"，口径同 evaluate.py）
#        anomaly_f1 / precision / recall / accuracy：异常识别（对照注入标注）
#   generative   —— LLM 报告质量（仅在真实 LLM 报告生成的 case 上聚合）
#        report_consistency / claim_* / citation_* / hallucination_rate
#
# 明确口径：
#   - claim_accuracy = correct / total_expected（faithfully stated 的比例）
#   - claim_precision = correct / (correct + wrong)（陈述中正确的比例）
#   - claim_recall = correct / total_expected（= accuracy，二者重合，因每个期望事实
#     只会 correct/wrong/missing；precision 在有额外陈述时才与 recall 分化）
#   - citation_completeness 与 citation_recall 同值（completeness 即 recall 型指标）
# =============================================================================

def _is_abnormal(gt_status: str) -> bool:
    return gt_status == "abnormal"


def _has_amount_issue(ground_truth: dict, tolerance: float, min_amt: float) -> bool:
    payable, paid = ground_truth.get("payable"), ground_truth.get("paid")
    if payable is None or paid is None:
        return False
    return abs(payable - paid) > max(tolerance * abs(payable), min_amt)


def compute_deterministic(cases: list[dict], tolerance: float = 0.01, min_amt: float = 0.01) -> dict:
    """
    确定性指标：规则引擎异常识别 vs 注入标注（truth_labels）+ 金额核对准确率。
    """
    tp = fp = fn = tn = 0
    amount_ok = amount_checked = 0

    for case in cases:
        if case.get("status") != "ok":
            continue
        gt = case.get("ground_truth") or {}
        label = case.get("truth_label") or {}
        predicted = _is_abnormal(gt.get("status", ""))
        actual = bool(label.get("anomaly"))

        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif actual and not predicted:
            fn += 1
        else:
            tn += 1

        # 应缴金额核对：排除"少报营业额"（其应缴基准被扭曲，不属于金额核对范畴）
        if label.get("anomaly_type") != "少报营业额":
            amount_checked += 1
            expected_amount_anomaly = (label.get("anomaly_type") == "金额不符")
            if _has_amount_issue(gt, tolerance, min_amt) == expected_amount_anomaly:
                amount_ok += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "dataset_size": len(cases),
        "amount_accuracy": round(amount_ok / amount_checked, 4) if amount_checked > 0 else 0.0,
        "amount_checked_bills": amount_checked,
        "anomaly_precision": round(precision, 4),
        "anomaly_recall": round(recall, 4),
        "anomaly_f1": round(f1, 4),
        "anomaly_accuracy": round(accuracy, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _generative_cases(case_results: list[dict]) -> list[dict]:
    """只在真实 LLM 报告生成的 case 上聚合生成式指标。"""
    return [r for r in case_results if r.get("report_mode") == "llm"]


def compute_generative(case_results: list[dict]) -> dict:
    """
    生成式指标聚合（LLM 报告质量）。

    聚合口径采用"求和再除"，避免少数超短报告扭曲均值。
    report_consistency 例外（每 case 已是 0~1 比例，取均值）。
    """
    evals = _generative_cases(case_results)

    def _sum(attr, sub):
        return sum((r.get(sub) or {}).get(attr, 0) for r in evals)

    n = len(evals)
    if n == 0:
        return {
            "evaluated_cases": 0,
            "report_consistency": None,
            "claim_accuracy": None, "claim_precision": None, "claim_recall": None,
            "citation_truthfulness": None, "citation_completeness": None,
            "citation_precision": None, "citation_recall": None,
            "hallucination_rate": None,
            "judge_error_rate": None,
        }

    # claim
    correct = _sum("correct", "consistency")
    wrong = _sum("wrong", "consistency")
    missing = _sum("missing", "consistency")
    total_expected = correct + wrong + missing

    # citation（required 分母由 citation_verifier 的 required_count 提供）
    cit_total = _sum("total", "citation")
    cit_valid = _sum("valid", "citation")
    cit_recalled = _sum("recalled", "citation")
    required_sum = _sum("required_count", "citation")

    # hallucination
    hall_total = _sum("total_claims", "hallucination")
    hall_count = _sum("hallucinated", "hallucination")

    return {
        "evaluated_cases": n,
        "report_consistency": round(sum(r.get("metrics", {}).get("report_consistency", 0) for r in evals) / n, 4),
        "claim_accuracy": round(correct / total_expected, 4) if total_expected > 0 else None,
        "claim_precision": round(correct / (correct + wrong), 4) if (correct + wrong) > 0 else None,
        "claim_recall": round(correct / total_expected, 4) if total_expected > 0 else None,
        "citation_truthfulness": round(cit_valid / cit_total, 4) if cit_total > 0 else None,
        "citation_completeness": round(cit_recalled / required_sum, 4) if required_sum > 0 else None,
        "citation_precision": round(cit_recalled / cit_total, 4) if cit_total > 0 else None,
        "citation_recall": round(cit_recalled / required_sum, 4) if required_sum > 0 else None,
        "hallucination_rate": round(hall_count / hall_total, 4) if hall_total > 0 else None,
        "judge_error_rate": None,  # LLM Judge 默认关闭（确定性路径），启用后由 runner 回填
        "_claim_breakdown": {"correct": correct, "wrong": wrong, "missing": missing,
                             "total_expected": total_expected},
        "_citation_breakdown": {"total_cited": cit_total, "valid": cit_valid,
                                "recalled": cit_recalled, "required": required_sum},
        "_hallucination_breakdown": {"events": hall_count, "total_claims": hall_total},
    }


def aggregate(cases: list[dict], case_results: list[dict]) -> dict:
    """
    汇总完整评测结果：确定性 + 生成式 + 运行统计。

    Returns:
        dict: {
            "dataset_size", "deterministic": {...}, "generative": {...},
            "runtime": {"report_llm": n, "report_degraded": n, "case_errors": n},
            "n_evaluated": n,
        }
    """
    deterministic = compute_deterministic(cases)
    generative = compute_generative(case_results)

    runtime = {
        "report_llm": sum(1 for r in case_results if r.get("report_mode") == "llm"),
        "report_degraded": sum(1 for r in case_results if r.get("report_mode") == "degraded"),
        "report_none": sum(1 for r in case_results if r.get("report_mode") == "none"),
        "case_errors": sum(1 for r in case_results if r.get("status") == "error"),
    }

    return {
        "dataset_size": len(cases),
        "deterministic": deterministic,
        "generative": generative,
        "runtime": runtime,
        "n_evaluated": generative.get("evaluated_cases", 0),
    }
