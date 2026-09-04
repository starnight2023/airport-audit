# =============================================================================
# scripts/benchmark_rag.py — RAG 增强效果对比评测
# =============================================================================
# 功能说明：
#   1. 构造一批测试查询（每查询标注期望命中的条款类型）
#   2. 对比不同检索方案的效果：
#      - 基础方案：单路向量检索（retrieve_clause）
#      - 增强方案：多路召回 + Reranker（enhanced_retrieve）
#      - Query 改写方案：规则改写 + 多路召回 + Reranker
#   3. 计算指标：
#      - Recall@K：前 K 条结果中命中期望条款类型的比例
#      - MRR：第一条命中结果排名的倒数均值
#      - Top-1 准确率：第一条结果是否命中
#
# 运行方式：
#   python scripts/benchmark_rag.py
#
# 面试阐述点：
#   - 基础向量检索 vs 增强检索的对比数据
#   - Query 改写对口语查询的提升
#   - 多路召回+精排相对单路召回的提升
# =============================================================================

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ===========================================================================
# 测试查询集（标注期望条款类型）
# ===========================================================================

def _load_test_queries() -> list[dict]:
    """从 contracts.json 动态选取评测商户，避免硬编码商户ID"""
    import json as _json
    _path = os.path.join(PROJECT_ROOT, "data", "contracts.json")
    with open(_path, "r", encoding="utf-8") as f:
        contracts = _json.load(f)

    def find_first(ctype):
        for c in contracts:
            if c["type"] == ctype:
                return c["merchant_id"]
        return "M001"

    comm = find_first("commission")    # 有提成比例的商户
    hybrid = find_first("hybrid")       # 有保底额+达标线的商户
    any_merchant = contracts[0]["merchant_id"] if contracts else "M001"

    return [
        {"query": "租金怎么算", "merchant": any_merchant, "expected": "rent_calculation"},
        {"query": "云松咖啡怎么收租", "merchant": any_merchant, "expected": "rent_calculation"},
        {"query": "保底额是多少", "merchant": hybrid, "expected": "rent_calculation"},
        {"query": "提成比例多少", "merchant": comm, "expected": "rent_calculation"},
        {"query": "报表最晚什么时候交", "merchant": any_merchant, "expected": "submission_deadline"},
        {"query": "提交截止日期", "merchant": any_merchant, "expected": "submission_deadline"},
        {"query": "逾期不交会怎么样", "merchant": any_merchant, "expected": "late_fee"},
        {"query": "滞纳金怎么收", "merchant": any_merchant, "expected": "late_fee"},
        {"query": "营业额达到多少按提成", "merchant": hybrid, "expected": "revenue_threshold"},
        {"query": "达标线在哪", "merchant": hybrid, "expected": "revenue_threshold"},
    ]


TEST_QUERIES = _load_test_queries()


# ===========================================================================
# 评测函数
# ===========================================================================

def evaluate_method(retrieve_fn, top_k=3) -> dict:
    """
    评测一种检索方案

    Args:
        retrieve_fn: 检索函数，签名 (merchant_id, query) -> list[dict]
        top_k: 评估前 K 条

    Returns:
        dict: {"recall@k": float, "mrr": float, "top1": float}
    """
    hits = 0
    mrr_sum = 0.0
    top1_hits = 0

    for case in TEST_QUERIES:
        results = retrieve_fn(case["merchant"], case["query"], top_k=top_k)
        expected = case["expected"]

        # 前 top_k 条里有没有命中期类型
        found_rank = None
        for i, r in enumerate(results):
            if r.get("clause_type") == expected:
                found_rank = i
                break

        if found_rank is not None:
            hits += 1
            mrr_sum += 1.0 / (found_rank + 1)
            if found_rank == 0:
                top1_hits += 1

    n = len(TEST_QUERIES)
    return {
        f"recall@{top_k}": round(hits / n, 4),
        "mrr": round(mrr_sum / n, 4),
        "top1": round(top1_hits / n, 4),
    }


# ===========================================================================
# 主入口
# ===========================================================================

def main():
    print("=" * 70)
    print("📊 RAG 增强效果对比评测")
    print("=" * 70)
    print(f"\n测试查询数: {len(TEST_QUERIES)}")
    print("查询类型覆盖: 租金计算 / 提成比例 / 保底额 / 提交截止日 / 滞纳金 / 达标线")
    print()

    from src.retriever import retrieve_clause, retrieve_multi_path, enhanced_retrieve, rerank

    def multi_path_rerank(mid, q, top_k):
        """多路召回（BM25+向量）+ Reranker 精排，不经 Query 改写"""
        recalled = retrieve_multi_path(mid, q, top_k=top_k * 3)
        return rerank(recalled, q, top_k=top_k)

    methods = {
        "基础单路向量检索": lambda mid, q, top_k: retrieve_clause(mid, q, top_k=top_k),
        "多路召回+精排(无改写)": multi_path_rerank,
        "Query改写+多路+精排": lambda mid, q, top_k: enhanced_retrieve(mid, q, top_k=top_k, rewrite_mode="rule", use_hyde=False),
    }

    print(f"{'方案':<22} {'Recall@3':<12} {'MRR':<10} {'Top-1':<10}")
    print("-" * 56)

    results = {}
    for name, fn in methods.items():
        metrics = evaluate_method(fn)
        results[name] = metrics
        print(f"{name:<22} {metrics['recall@3']:.1%}        {metrics['mrr']:.3f}    {metrics['top1']:.1%}")

    # 对比总结
    base = results.get("基础单路向量检索", {})
    enhanced = results.get("Query改写+多路+精排", {})
    print("\n" + "=" * 70)
    print("对比结论")
    print("=" * 70)
    if base and enhanced:
        print(f"  Query改写+多路+精排 vs 基础向量检索:")
        print(f"    Recall@3: {base['recall@3']:.1%} → {enhanced['recall@3']:.1%} "
              f"({enhanced['recall@3'] - base['recall@3']:+.1%})")
        print(f"    MRR:      {base['mrr']:.3f} → {enhanced['mrr']:.3f} "
              f"({enhanced['mrr'] - base['mrr']:+.3f})")
        print(f"    Top-1:    {base['top1']:.1%} → {enhanced['top1']:.1%} "
              f"({enhanced['top1'] - base['top1']:+.1%})")

    print("\n✅ 评测完成")


if __name__ == "__main__":
    main()