# =============================================================================
# evaluation/llm_judge.py — 独立 LLM Judge（可选第二层语义判定）
# =============================================================================
# 用途：
#   仅当确定性集合比对无法判定"引用是否支撑 Claim"时才启用（本数据下默认关闭）。
#   与报告生成解耦：不使用生成报告的同一调用，Judge 有独立 prompt 与解析。
#
# 接口：
#   judge_entailment(premise, hypothesis, retries=2) -> dict
#     返回 {"label": "entailment"|"contradiction"|"unknown",
#           "confidence": float, "reason": str}
#     解析失败自动重试；最终失败返回 {"label":"unknown", "judge_error": true}，
#     绝不让评测整体崩溃。
# =============================================================================

import json
import os
import re

JUDGE_LABELS = ("entailment", "contradiction", "unknown")

JUDGE_SYSTEM_PROMPT = """你是一个严格的语义蕴含判定器。给定"前提(premise)"和"假设(hypothesis)"，
判断 hypothesis 是否被 premise 支持。

- entailment：premise 支持或蕴含 hypothesis
- contradiction：premise 与 hypothesis 直接矛盾
- unknown：无法从 premise 判定

只输出严格 JSON（不要 markdown、不要多余文字）：
{"label": "entailment|contradiction|unknown", "confidence": 0.0~1.0, "reason": "一句话理由"}"""


def _parse_judge_response(text: str) -> dict:
    """解析 Judge 输出，容忍少量噪声（截取首个 JSON 对象）。"""
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"无法从 Judge 输出中解析 JSON: {text[:200]!r}")
        data = json.loads(m.group(0))

    label = str(data.get("label", "")).strip().lower()
    if label not in JUDGE_LABELS:
        raise ValueError(f"非法 label: {label!r}")
    return {
        "label": label,
        "confidence": float(data.get("confidence", 0.5)),
        "reason": str(data.get("reason", "")),
        "judge_error": False,
    }


def judge_entailment(
    premise: str,
    hypothesis: str,
    retries: int = 2,
    temperature: float = 0.0,
    model: str | None = None,
) -> dict:
    """
    调用 LLM Judge 判定语义蕴含。

    Args:
        premise: 引用/证据文本
        hypothesis: 待判定 Claim
        retries: 解析失败后的重试次数
        temperature / model: LLM 参数（model 缺省用 DEEPSEEK_MODEL）

    Returns:
        {"label", "confidence", "reason", "judge_error"}
    """
    from src.agent_graph import _call_llm  # 复用现有 LLM 客户端封装

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = _call_llm([
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"前提：{premise}\n假设：{hypothesis}"},
            ], temperature=temperature, max_tokens=300)
            return _parse_judge_response(resp)
        except Exception as e:  # noqa: BLE001
            last_err = e

    return {
        "label": "unknown",
        "confidence": 0.0,
        "reason": f"Judge 调用/解析最终失败: {last_err}",
        "judge_error": True,
    }
