# =============================================================================
# src/query_rewriter.py — RAG 增强：Query 改写
# =============================================================================
# 功能说明：
#   1. 规则模式：基于同义词词典对用户查询做关键词扩展，不改写原始查询
#   2. LLM 模式：调用 DeepSeek 把口语化查询改写成更适合向量检索的书面语
#   3. 两种模式输出均可用于后续向量检索
#
# 为什么需要 Query 改写？
#   用户输入是口语（"云松咖啡怎么收租"），合同条款是书面语（"租金计算方式：
#   月租金 = max(保底额..., 申报营业额 × ...)"）。直接用口语向量去匹配，
#   相似度可能偏低。改写后让查询与文档在词汇和语义上更接近。
#
# 使用方式：
#   from src.query_rewriter import rewrite_query
#   rewritten = rewrite_query("云松咖啡怎么收租", mode="rule")
#   rewritten = rewrite_query("云松咖啡怎么收租", mode="llm")
# =============================================================================

import os
import sys
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ===========================================================================
# 一、规则模式：同义词词典
# ===========================================================================

# 面向机场计费场景的同义词/相关词映射
SYNONYM_DICT = {
    # 租金相关
    "租金": ["租金", "租费", "月租", "租赁费", "租金计算", "收费", "应缴", "收租", "交租"],
    "怎么算": ["计算方式", "计算方法", "公式", "如何计算"],
    "保底": ["保底额", "最低租金", "底租", "最低消费", "保底租金"],
    "提成": ["提成比例", "分成", "抽成", "佣金比例", "佣金"],
    "达标线": ["达标线", "营业额阈值", "阈值", "分界线"],
    # 提交相关
    "提交": ["提交", "申报", "报送", "上交", "提交报表"],
    "截止": ["截止日", "最后期限", "deadline", "最晚"],
    "逾期": ["逾期", "过期", "超期", "滞纳金"],
    "滞纳金": ["滞纳金", "罚款", "违约金", "罚金"],
    # 报表相关
    "报表": ["报表", "营业额报表", "营业收入报表", "账单"],
    "营业额": ["营业额", "营业收入", "营收", "销售额", "收入"],
    # 其他
    "多少": ["金额", "数额", "费用", "金额多少"],
    "扣款": ["扣款", "缴纳", "缴费", "实缴", "实付"],
}

# 停用词（去除噪音）
STOPWORDS = {"的", "了", "吗", "呢", "啊", "呀", "是", "有", "和", "与", "跟", "及",
             "我", "你", "他", "它", "我们", "你们", "他们", "请", "帮", "查一下", "看看"}


def _tokenize_chinese(text: str) -> list[str]:
    """简单中文分词：按标点/空白切分 + 提取关键词子串"""
    import re
    # 按非中文/数字/字母的字符切分
    tokens = re.findall(r'[一-鿿]+|[a-zA-Z0-9]+', text)
    result = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        result.append(tok)
    return result


def rewrite_query_rule(query: str) -> str:
    """
    规则模式改写：基于同义词词典扩展关键词，返回改写后的查询

    策略：
    - 从查询中提取所有命中的同义词组
    - 将命中的词扩展为同义词组拼接，保留原文
    - 输出为"原文 + 扩展词"的形式

    例："云松咖啡怎么收租" → "云松咖啡怎么收租 租金 租费 月租 租赁费 计算方式"
    """
    if not query or not query.strip():
        return query

    expansions = []
    for keyword, synonyms in SYNONYM_DICT.items():
        # 双向匹配：词根 或 任一同义词 出现在查询中即命中该组
        group_hit = (keyword in query) or any(syn in query for syn in synonyms if syn)
        if group_hit:
            # 把该组所有词（词根+同义词）中未出现在查询里的都加入扩展
            all_terms = [keyword] + [s for s in synonyms if s]
            for term in all_terms:
                if term != keyword and term not in query and term not in expansions:
                    expansions.append(term)

    if not expansions:
        return query.strip()

    return f"{query.strip()} {' '.join(expansions)}"


# ===========================================================================
# 二、LLM 模式：调用 DeepSeek 改写
# ===========================================================================

REWRITE_LLM_PROMPT = """你是一个检索查询改写器。你的任务是：将用户的自然语言查询改写成更适合向量检索的书面表达。

改写要求：
1. 保留查询中的关键业务信息（商户、合同类型、条款类型等）
2. 将口语表达（"怎么收租""查一下""有没有问题"）转成书面语（"租金计算方式""提交截止日""滞纳金规则"）
3. 补充可能缺失的关键词（如查询只说"租金"，可补充"租金计算方式"）
4. 输出简短的一句话，不要解释，不要加引号

示例：
输入：云松咖啡怎么收租
输出：云松咖啡租金计算方式

输入：报表最晚什么时候交
输出：商户提交营业报表截止日期

输入：逾期不交会怎么样
输出：逾期提交滞纳金收取规则

请改写以下查询：
{query}"""


def rewrite_query_llm(query: str) -> str:
    """
    LLM 模式改写：调用 DeepSeek 将口语转书面语

    失败时降级为规则模式改写，保证不中断。
    """
    from agent_graph import _call_llm
    try:
        response = _call_llm([
            {"role": "system", "content": REWRITE_LLM_PROMPT},
            {"role": "user", "content": f"查询：{query}"},
        ], temperature=0.1, max_tokens=100)
        result = response.strip().strip('"').strip("'")
        if result and len(result) > 2:
            return result
        return rewrite_query_rule(query)
    except Exception:
        # LLM 不可用时降级为规则改写
        return rewrite_query_rule(query)


# ===========================================================================
# 三、统一入口
# ===========================================================================

def rewrite_query(query: str, mode: str = "rule", use_llm: bool = False) -> str:
    """
    Query 改写统一入口

    Args:
        query: 原始查询
        mode: "rule" 规则模式 / "llm" LLM 模式
        use_llm: True 时即使 mode="rule" 也尝试 LLM（LLM 优先，失败降级规则）

    Returns:
        改写后的查询字符串
    """
    if not query or not query.strip():
        return query

    if mode == "llm" or use_llm:
        # 先 LLM，失败自动降级规则
        rewritten = rewrite_query_llm(query)
        return rewritten
    return rewrite_query_rule(query)


# ===========================================================================
# 四、快速验证
# ===========================================================================

def demo():
    """演示 Query 改写"""
    print("=" * 50)
    print("✍️  Query 改写演示")
    print("=" * 50)

    tests = [
        "云松咖啡怎么收租",
        "报表最晚什么时候交",
        "逾期不交会怎么样",
        "保底额是多少",
        "星野小馆提成比例",
    ]

    for q in tests:
        rule_result = rewrite_query_rule(q)
        print(f"\n原始: {q}")
        print(f"规则: {rule_result}")


if __name__ == "__main__":
    demo()