# =============================================================================
# evaluation/hallucination.py — Hallucination 检测（确定性，白名单化）
# =============================================================================
# 定义：报告中存在、但 Ground Truth / 规则引擎 / 账单 / 合同均不能支持的事实。
#
# 判定为幻觉的三类确定性事件（类别互斥，不与其他指标重复计数）：
#   1. nonexistent_citation —— 引用了一个不在商户合同条款集合里的 clause_id
#   2. invalid_status       —— 报告 status 不是 normal/abnormal/error 之一
#   3. hallucinated_number  —— 报告里出现的金额，既不对应任何来源金额，
#                              也不属于已识别的（revenue/payable/paid/difference）Claim
#                              （已被识别的错误金额归入 consistency 的矛盾类，不计幻觉）
#
# 允许的推论（不判幻觉）：
#   - 金额派生结论（"存在少缴情况"）可由 difference>0 推出；
#   - 报告的 key 事实若与来源金额一致，属正常陈述。
#
# 指标：hallucination_rate = hallucinated_claims / total_claims
#   total_claims = 金额断言数 + 1(status) + 引用条款数
# =============================================================================

def build_source_amounts(ground_truth: dict, contract: dict) -> list[float]:
    """
    该 case 报告"应当出现"的金额集合：
      账单事实（revenue/paid/payable/difference）+ 合同参数（固定额/保底/达标线）。
    """
    sources = []
    for key in ("revenue", "paid", "payable", "difference"):
        v = ground_truth.get(key)
        if v is not None:
            sources.append(float(v))

    c = contract or {}
    if c.get("fixed_amount") is not None:
        sources.append(float(c["fixed_amount"]))
    if c.get("min_guarantee") is not None:
        sources.append(float(c["min_guarantee"]))
        rate = c.get("commission_rate") or 0
        if rate > 0:
            sources.append(round(float(c["min_guarantee"]) / float(rate), 2))  # 达标线

    return sources


def _near_any(value: float, sources: list[float], tolerance: float, min_amt: float) -> bool:
    for s in sources:
        if abs(value - s) <= max(tolerance * abs(s), min_amt):
            return True
    return False


def detect_hallucinations(
    extracted: dict,
    ground_truth: dict,
    contract: dict,
    valid_clause_ids: set,
    tolerance: float = 0.01,
    min_anomaly_amount: float = 0.01,
) -> dict:
    """
    检测报告中的幻觉事件。

    Args:
        extracted: claim_extractor.extract_claims 的输出
        ground_truth / contract / valid_clause_ids: 事实源
        tolerance / min_anomaly_amount: 金额容差

    Returns:
        dict: {
            "events": [{"type", "claim", "evidence", "severity"}],
            "hallucinated": int,
            "total_claims": int,          # 金额断言 + status + 引用数
            "hallucination_rate": float,
        }
    """
    events = []
    sources = build_source_amounts(ground_truth, contract)
    attributed_values = {float(c["value"]) for c in extracted.get("claims", [])}

    # ---- 1. 不存在的引用条款 ----
    for cid in extracted.get("citations", []):
        if cid not in valid_clause_ids:
            events.append({
                "type": "nonexistent_citation",
                "claim": f"引用条款 {cid}",
                "evidence": f"该条款 ID 不在商户合同条款集合中（{len(valid_clause_ids)} 条真实条款）",
                "severity": "high",
            })

    # ---- 2. 非法 status ----
    if extracted.get("invalid_status"):
        events.append({
            "type": "invalid_status",
            "claim": f"报告状态为 {extracted.get('raw_status')!r}",
            "evidence": "status 只能是 normal / abnormal / error",
            "severity": "high",
        })

    # ---- 3. 无来源的金额 ----
    for amt in extracted.get("amounts", []):
        value = float(amt["value"])
        # 已识别的 Claim 金额（即使是错的）归一致性矛盾类，不计幻觉
        if value in attributed_values:
            continue
        if not _near_any(value, sources, tolerance, min_anomaly_amount):
            events.append({
                "type": "hallucinated_number",
                "claim": f"金额 ¥{value:,.2f}（上下文：{amt['text']}）",
                "evidence": f"该金额与账单/合同/规则引擎中的任何来源金额均不一致（来源: {[f'{s:,.2f}' for s in sources]}）",
                "severity": "medium",
            })

    # ---- 汇总 ----
    total_claims = len(extracted.get("amounts", [])) + 1 + len(extracted.get("citations", []))
    hallucinated = len(events)

    return {
        "events": events,
        "hallucinated": hallucinated,
        "total_claims": total_claims,
        "hallucination_rate": round(hallucinated / total_claims, 4) if total_claims > 0 else 0.0,
    }
