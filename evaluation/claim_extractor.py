# =============================================================================
# evaluation/claim_extractor.py — 报告原子 Claim 抽取（确定性优先）
# =============================================================================
# 目标：
#   把 LLM 生成的半结构化报告拆解为可判定的原子 Claim：
#     - status（稽核状态）
#     - revenue（申报营业额）
#     - payable（应缴金额）
#     - paid（实缴金额）
#     - difference（差额）
#     - citation（引用的条款 ID）
#
# 设计：
#   1. 报告本身是结构化 JSON（report_node 输出 {summary, status, findings,
#      suggestion, contract_refs}），status / contract_refs 直接可读，无需 LLM；
#   2. 金额用"锚定关键词 + 金额"正则抽取（如"应缴金额为¥7,632.47"），
#      并用公式片段预处理消除 "¥52,893.10×14.43%=¥7,632.47" 这类干扰；
#   3. 不默认调用 LLM（可用 use_llm_judge 仅处理无法判定的自由文本）。
#
# 抽取对象（claim_type → 来源字段/文本）：
#   status      → report["status"]
#   revenue     → 含"申报营业额/营业额"等关键词附近的金额
#   payable     → 含"应缴/应有/应交"等关键词附近的金额
#   paid        → 含"实缴/实际缴纳"等关键词附近的金额
#   difference  → 含"差额/少缴/多缴"等关键词附近的金额
# =============================================================================

import re
from typing import Optional

# 金额：¥ 前缀或"元"后缀（支持千分位，如 ¥7,632.47）
_MONEY_NUM = r"[\d,]+(?:\.\d{1,2})?"
_MONEY_RE = re.compile(
    rf"(?:¥|￥)\s*({_MONEY_NUM})|({_MONEY_NUM})\s*元"
)

# 公式片段预处理：把 "¥52,893.10×14.43%=¥7,632.47" 折叠为 "¥7,632.47"
# （避免公式左侧的营业额被误判为应缴金额）
_FORMULA_RE = re.compile(
    rf"¥?\s*{_MONEY_NUM}\s*[×*xX÷/]\s*\d+(?:\.\d+)?\s*%?\s*=\s*¥?\s*({_MONEY_NUM})"
)


def _preprocess(text: str) -> str:
    """折叠内联公式，使关键词锚定更可靠。"""
    return _FORMULA_RE.sub(r"¥\1", text)


# 锚定关键词模式：某类事实的关键词 → 紧随其后的金额
_ANCHORED = [
    ("revenue",   re.compile(r"(?:申报营业额|营业额|申报营收|营收)(?:为|是)?[:：]?\s*(?:约)?\s*¥?\s*(%s)" % _MONEY_NUM)),
    ("payable",   re.compile(r"(?:应缴金额|应缴|应交|应有金额|应有|应付|应收|合同计算|计算应缴|应收租金|应付租金)(?:为|是)?[:：]?\s*(?:约)?\s*¥?\s*(%s)" % _MONEY_NUM)),
    ("paid",      re.compile(r"(?:实际缴纳|实际缴费|实缴金额|实收|缴纳金额|实际支付|已缴|缴费)(?:为|是)?[:：]?\s*(?:约)?\s*¥?\s*(%s)" % _MONEY_NUM)),
    ("difference", re.compile(r"(?:差额|少缴|少交|多缴|多交|欠缴|少付|差异)(?:为|是)?[:：]?\s*(?:约)?\s*¥?\s*(%s)" % _MONEY_NUM)),
]

# 条款 ID 模式：如 CTR-001-clause-001（须以字母开头，避免切出 "001-clause-001" 子串）
_CLAUSE_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*-\d+-clause-\d+")

_VALID_STATUS = {"normal", "abnormal", "error"}


def _normalize(value: str) -> float:
    return float(value.replace(",", ""))


def _report_texts(report: dict) -> list[tuple[str, str]]:
    """返回 [(来源名, 文本)]：summary / 各 findings.description / suggestion。"""
    texts = []
    if report.get("summary"):
        texts.append(("summary", report["summary"]))
    for i, f in enumerate(report.get("findings") or []):
        if f.get("description"):
            texts.append((f"findings[{i}]", f["description"]))
    if report.get("suggestion"):
        texts.append(("suggestion", report["suggestion"]))
    return texts


def _first_tagged_amount(anchor_type: str, texts: list[tuple[str, str]]) -> Optional[dict]:
    """在全部文本中按锚定关键词找第一个匹配金额。

    同样先做公式折叠（_FORMULA_RE），避免 "应缴金额为¥52,893.10×14.43%=¥7,632.47"
    这类内联公式把左侧营业额误判为应缴金额。
    """
    for source, raw in texts:
        text = _preprocess(raw)
        for m in _ANCHORED:
            if m[0] != anchor_type:
                continue
            for mm in m[1].finditer(text):
                return {
                    "value": _normalize(mm.group(1)),
                    "text": mm.group(0),
                    "source": source,
                }
    return None


def extract_all_amounts(texts: list[tuple[str, str]]) -> list[dict]:
    """抽取所有金额（含公式折叠后），供幻觉检测使用。"""
    amounts = []
    for source, raw in texts:
        text = _preprocess(raw)
        for m in _MONEY_RE.finditer(text):
            val = m.group(1) or m.group(2)
            amounts.append({
                "value": _normalize(val),
                "text": text[max(0, m.start() - 15): m.end() + 15],
                "source": source,
            })
    return amounts


def extract_citations(report: dict) -> list[str]:
    """
    抽取报告引用的条款 ID：
      1. report["contract_refs"]（结构化字段）
      2. 全文正则匹配（findings[].evidence、summary 等自由文本）
    去重、保序。
    """
    cited: list[str] = []
    seen = set()
    for ref in (report.get("contract_refs") or []):
        if isinstance(ref, str) and ref not in seen:
            seen.add(ref)
            cited.append(ref)
    for _, raw in _report_texts(report):
        text = _preprocess(raw)
        for m in _CLAUSE_ID_RE.finditer(text):
            cid = m.group(0)
            if cid not in seen:
                seen.add(cid)
                cited.append(cid)
    return cited


def extract_claims(report: dict) -> dict:
    """
    从报告抽取结构化 Claim。

    Returns:
        dict: {
            "status": str | None,          # 报告声明的稽核状态
            "claims": [                    # 已识别的金额类 Claim
                {"claim_id", "claim_type", "claim", "value", "source"}
            ],
            "amounts": [...],              # 全部金额（幻觉检测用）
            "citations": [...],            # 引用条款 ID
            "invalid_status": bool,        # status 是否非法（幻觉候选）
        }
    """
    texts = _report_texts(report)
    status = report.get("status")
    invalid_status = status is not None and status not in _VALID_STATUS

    claims = []
    for anchor_type in ("revenue", "payable", "paid", "difference"):
        found = _first_tagged_amount(anchor_type, texts)
        if found:
            claims.append({
                "claim_id": f"CLAIM-{anchor_type.upper()}",
                "claim_type": anchor_type,
                "claim": found["text"],
                "value": found["value"],
                "source": found["source"],
            })

    return {
        "status": status if status in _VALID_STATUS else None,
        "raw_status": status,
        "invalid_status": invalid_status,
        "claims": claims,
        "amounts": extract_all_amounts(texts),
        "citations": extract_citations(report),
    }
