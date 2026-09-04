# =============================================================================
# src/hyde.py — RAG 增强：HyDE（假设文档嵌入）
# =============================================================================
# 原理：
#   HyDE (Hypothetical Document Embeddings) 的核心思想是：
#   先用 LLM 生成一个"假设的回答文档"（针对查询的假设条款文本），
#   再用这个假设文档的向量去向量库检索，而不是直接用查询向量。
#
#   为什么有效？
#   查询（口语）和文档（书面条款）的词汇/表述差异大，直接匹配相似度低。
#   假设文档是"模拟的条款文本"，和真实条款的表述更接近，
#   所以用假设文档向量检索，命中相关条款的概率更高。
#
# 使用方式：
#   from src.hyde import hyde_retrieve
#   results = hyde_retrieve("M001", "租金怎么算", top_k=3)
# =============================================================================

import os
import sys
import json
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ===========================================================================
# 一、LLM 生成假设条款
# ===========================================================================

HYDE_PROMPT = """你是一个机场合同条款生成器。你的任务是：针对用户提出的合同查询问题，生成一段"假设的合同条款原文"。

生成的条款要：
1. 以合同条款的口吻书写（如"租金计算方式：""提交截止日：""滞纳金规则："）
2. 使用与真实合同条款一致的用词（如"申报营业额""保底额""提成比例""截止日"）
3. 不要写具体数字，用占位符或范围代替（因为真实条款的数字在知识库中）
4. 只输出条款原文，不要解释

示例：
查询：云松咖啡怎么收租
输出：租金计算方式：月租金取保底额与申报营业额乘以提成比例的较高值

查询：报表最晚什么时候交
输出：商户需在每月截止日前提交上月营业报表，逾期将产生滞纳金

查询：逾期不交会怎么样
输出：逾期提交按日收取滞纳金，宽限期后可加收罚款

请生成以下查询对应的假设条款：
{query}"""


def generate_hypothetical_clause(query: str) -> str:
    """
    调用 LLM 生成假设的合同条款文本

    失败时降级为原始查询，保证不中断。

    Args:
        query: 用户查询

    Returns:
        假设条款文本（降级时为原始查询）
    """
    from agent_graph import _call_llm
    try:
        response = _call_llm([
            {"role": "system", "content": HYDE_PROMPT},
            {"role": "user", "content": f"查询：{query}"},
        ], temperature=0.3, max_tokens=100)
        result = response.strip().strip('"').strip("'")
        if result and len(result) > 5:
            return result
        return query
    except Exception:
        return query


# ===========================================================================
# 二、HyDE 检索
# ===========================================================================

def hyde_retrieve(
    merchant_id: str,
    query: str,
    top_k: int = 3,
    use_llm: bool = True,
) -> list[dict]:
    """
    HyDE 检索：生成假设文档 → 向量检索

    Args:
        merchant_id: 商户ID
        query: 用户查询
        top_k: 返回条数
        use_llm: True 用 LLM 生成假设文档（慢但准）；False 直接用原查询

    Returns:
        list[dict]: 与 retrieve_clause 相同格式的结果
    """
    # 生成假设条款
    if use_llm:
        hypothetical = generate_hypothetical_clause(query)
    else:
        hypothetical = query

    # 用假设条款检索
    from src.retriever import retrieve_clause
    results = retrieve_clause(merchant_id, hypothetical, top_k=top_k)

    # 记录用的是什么查询
    for r in results:
        r["_hypothetical"] = hypothetical

    return results


# ===========================================================================
# 三、演示
# ===========================================================================

def demo():
    """演示 HyDE 检索"""
    print("=" * 50)
    print("🧠 HyDE 检索演示")
    print("=" * 50)

    tests = ["租金怎么算", "提交截止日", "滞纳金"]
    for q in tests:
        # 不调 LLM，直接用原查询，只展示流程
        results = hyde_retrieve("M001", q, top_k=2, use_llm=False)
        print(f"\n查询: {q}")
        for r in results:
            print(f"  [{r['score']:.4f}] {r['clause_type']}: {r['description'][:40]}...")


if __name__ == "__main__":
    demo()