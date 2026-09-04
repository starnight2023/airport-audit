# =============================================================================
# evaluation/citation_verifier.py — 引用校验（确定性集合比对）
# =============================================================================
# 本项目的引用是 clause_id（字符串），而非引用原文段落。因此 Citation
# Truthfulness / Completeness / Precision / Recall 全部退化为确定性集合运算：
#
#   有效性(valid)    = 引用的 clause_id ∈ 商户合同真实条款集合
#   支持性(supported)= 引用的 clause_id ∈ 规则引擎本次实际使用的条款集合
#   required         = 规则引擎本次实际使用的条款（ground truth 引用）
#
# 指标定义（逐 case）：
#   citation_truthfulness  = |cited ∩ valid| / |cited|        （引用是否指向真实条款）
#   citation_completeness  = |required ∩ cited| / |required|  （是否引用了应引用的证据，= recall）
#   citation_recall        = 同 completeness
#   citation_precision     = |required ∩ cited| / |cited|     （引用里有多少是必需证据）
#
# 说明：
#   - 本项目无 bill_id / payment_id 命名空间（营收与实缴同属一条账单记录），
#     故 Completeness 仅在条款证据维度计算。
# =============================================================================


def valid_clause_ids(contract: dict) -> set:
    """商户合同真实存在的条款 ID 集合。"""
    ids = set()
    for c in (contract or {}).get("clauses", []) or []:
        if c.get("clause_id"):
            ids.add(c["clause_id"])
    return ids


def verify_citations(
    cited_ids: list[str],
    contract: dict,
    used_clause_ids: list[str],
) -> dict:
    """
    校验一批引用条款，返回逐条结果与聚合指标。

    Returns:
        dict: {
            "results": [{"citation", "exists", "supported", "relevant"}],
            "total": int, "valid": int, "supported": int,
            "recalled": int,
            "truthfulness": float, "completeness": float,
            "precision": float, "recall": float,
        }
    """
    valid = valid_clause_ids(contract)
    used = set(used_clause_ids)

    results = []
    for cid in cited_ids:
        exists = cid in valid
        supported = exists and cid in used
        results.append({
            "citation": cid,
            "exists": exists,
            "supported": supported,
            "relevant": cid in used,
        })

    total = len(cited_ids)
    n_valid = sum(1 for r in results if r["exists"])
    n_supported = sum(1 for r in results if r["supported"])
    recalled = len(used & set(cited_ids))
    n_required = len(used)

    return {
        "results": results,
        "total": total,
        "valid": n_valid,
        "supported": n_supported,
        "recalled": recalled,
        "required_count": n_required,
        # 未引用任何条款时：truthfulness=1.0（无虚假引用），precision/completeness 视为 0
        "truthfulness": round(n_valid / total, 4) if total > 0 else 1.0,
        "completeness": round(recalled / n_required, 4) if n_required > 0 else 1.0,
        "precision": round(recalled / total, 4) if total > 0 else 0.0,
        "recall": round(recalled / n_required, 4) if n_required > 0 else 1.0,
    }
