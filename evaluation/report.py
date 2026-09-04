# =============================================================================
# evaluation/report.py — 评测结果输出（JSON / HTML / 逐 case / Top 失败案例）
# =============================================================================

import html
import json
import os
from datetime import datetime


def _fmt(value, digits=4):
    """指标格式化：None → "n/a"。"""
    if value is None:
        return "n/a"
    return f"{value:.{digits}%}" if isinstance(value, float) else str(value)


def _fmt_num(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


# =============================================================================
# 一、保存机器可读结果
# =============================================================================

def save_json(metrics: dict, output_dir: str) -> str:
    path = os.path.join(output_dir, "evaluation_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def save_cases(case_results: list[dict], output_dir: str) -> str:
    """逐 case 详细结果（含 report / claims / citations / 判定结果）。"""
    path = os.path.join(output_dir, "evaluation_cases.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(case_results, f, ensure_ascii=False, indent=2)
    return path


# =============================================================================
# 二、Top 失败案例
# =============================================================================

def _case_failure_weight(result: dict) -> tuple[int, int, int]:
    """(矛盾事实数, 幻觉事件数, 无效引用数) 用于失败案例排序。"""
    cons = result.get("consistency") or {}
    hall = result.get("hallucination") or {}
    cit = result.get("citation") or {}
    contradiction = cons.get("wrong", 0)
    hallucination = hall.get("hallucinated", 0)
    invalid_citation = sum(1 for r in (cit.get("results") or []) if not r.get("exists"))
    return (contradiction + hallucination + invalid_citation, hallucination, contradiction)


def build_top_failures(case_results: list[dict], top_n: int = 10) -> list[dict]:
    """
    从失败案例中挑选 Top-N（矛盾/幻觉/无效引用最多者），
    每个案例附带可读的失败原因摘要。
    """
    ranked = []
    for r in case_results:
        if r.get("report_mode") != "llm":
            continue
        weight = _case_failure_weight(r)
        if weight[0] == 0:
            continue
        ranked.append((weight, r))
    ranked.sort(key=lambda x: x[0], reverse=True)

    failures = []
    for weight, r in ranked[:top_n]:
        reasons = []

        # 矛盾事实
        for f in (r.get("consistency") or {}).get("facts", []):
            if f["state"] == "wrong":
                reasons.append({
                    "type": "fact_contradiction",
                    "detail": (
                        f"事实[{f['fact']}]：规则引擎={_fmt_num(f['expected'])}，"
                        f"报告={_fmt_num(f['reported'])}"
                    ),
                })
        # 幻觉
        for ev in (r.get("hallucination") or {}).get("events", []):
            reasons.append({
                "type": ev["type"],
                "detail": f"{ev['claim']} —— {ev['evidence']}",
            })
        # 无效引用
        for res in (r.get("citation") or {}).get("results", []):
            if not res.get("exists"):
                reasons.append({
                    "type": "nonexistent_citation",
                    "detail": f"引用条款 {res['citation']} 不存在于商户合同",
                })

        failures.append({
            "case_id": r.get("case_id"),
            "merchant_id": (r.get("input") or {}).get("merchant_id"),
            "month": (r.get("input") or {}).get("month"),
            "weight": weight[0],
            "reasons": reasons,
        })
    return failures


def print_top_failures(failures: list[dict]) -> None:
    """终端友好输出 Top 失败案例。"""
    if not failures:
        print("\n✅ 无失败案例")
        return
    print("\n" + "=" * 70)
    print(f"🔴 Top {len(failures)} 失败案例")
    print("=" * 70)
    for f in failures:
        print(f"\nCase {f['case_id']} ({f['merchant_id']} {f['month']})")
        for reason in f["reasons"]:
            print(f"  Issue: {reason['type']}")
            print(f"    {reason['detail']}")
        print("  " + "-" * 60)


# =============================================================================
# 三、HTML 报告
# =============================================================================

def _metrics_table(title: str, mapping: dict, keys: list[tuple[str, str]]) -> str:
    rows = "".join(
        f"<tr><td class='k'>{name}</td><td>{_fmt(mapping.get(key))}</td></tr>"
        for key, name in keys
    )
    return f"<h3>{html.escape(title)}</h3><table>{rows}</table>"


def save_html(metrics: dict, case_results: list[dict], failures: list[dict], output_dir: str) -> str:
    det = metrics.get("deterministic", {})
    gen = metrics.get("generative", {})
    rt = metrics.get("runtime", {})

    deterministic_html = _metrics_table("确定性指标（规则引擎 vs 注入标注）", det, [
        ("amount_accuracy", "应缴金额核对准确率"),
        ("anomaly_precision", "异常识别精确率"),
        ("anomaly_recall", "异常识别召回率"),
        ("anomaly_f1", "异常识别 F1"),
        ("anomaly_accuracy", "异常识别准确率"),
    ])

    generative_html = _metrics_table("生成式指标（LLM 报告质量，仅在真实报告 case 上聚合）", gen, [
        ("report_consistency", "Report Consistency"),
        ("claim_accuracy", "Claim Accuracy"),
        ("claim_precision", "Claim Precision"),
        ("claim_recall", "Claim Recall"),
        ("citation_truthfulness", "Citation Truthfulness"),
        ("citation_completeness", "Citation Completeness"),
        ("citation_precision", "Citation Precision"),
        ("citation_recall", "Citation Recall"),
        ("hallucination_rate", "Hallucination Rate"),
        ("judge_error_rate", "Judge Error Rate"),
    ])

    # 逐 case 表格
    case_rows = []
    for r in case_results:
        cons = r.get("consistency") or {}
        cit = r.get("citation") or {}
        hall = r.get("hallucination") or {}
        m = r.get("metrics") or {}
        case_rows.append(
            "<tr>"
            f"<td>{html.escape(r.get('case_id', ''))}</td>"
            f"<td>{html.escape((r.get('input') or {}).get('merchant_id', ''))}</td>"
            f"<td>{html.escape((r.get('input') or {}).get('month', ''))}</td>"
            f"<td>{html.escape(r.get('report_mode', ''))}</td>"
            f"<td>{_fmt(m.get('report_consistency'))}</td>"
            f"<td>{_fmt(m.get('claim_recall_case'))}</td>"
            f"<td>{_fmt(cit.get('truthfulness'))}</td>"
            f"<td>{_fmt(cit.get('completeness'))}</td>"
            f"<td>{hall.get('hallucinated', 0)}</td>"
            "</tr>"
        )
    case_table = (
        "<table><tr><th>case</th><th>商户</th><th>月份</th><th>模式</th>"
        "<th>一致性</th><th>claim覆盖</th><th>引用真实率</th><th>引用完整率</th><th>幻觉数</th></tr>"
        + "".join(case_rows) + "</table>"
    )

    # 失败案例 HTML
    failure_blocks = []
    for f in failures:
        reasons = "<br>".join(
            f"<b>[{html.escape(r['type'])}]</b> {html.escape(r['detail'])}"
            for r in f["reasons"]
        )
        failure_blocks.append(
            f"<div class='fail'><b>Case {html.escape(f['case_id'])}"
            f" ({f['merchant_id']} {f['month']})</b><br>{reasons}</div>"
        )
    failure_html = "".join(failure_blocks) or "<p>无</p>"

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>生成式评测报告</title>
<style>
  body {{ font-family: -apple-system, 'PingFang SC', sans-serif; margin: 24px; color: #222; }}
  h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 28px; border-bottom: 1px solid #ddd; }}
  h3 {{ font-size: 14px; margin: 16px 0 6px; }}
  table {{ border-collapse: collapse; margin: 8px 0 20px; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 5px 10px; text-align: left; }}
  th {{ background: #f5f5f5; }}
  td.k {{ color: #555; }}
  .fail {{ background: #fff3f3; border: 1px solid #f2c2c2; padding: 8px 12px; margin: 6px 0; border-radius: 4px; font-size: 13px; }}
  .meta {{ color: #777; font-size: 13px; }}
</style></head><body>
<h1>机场非航收入智能稽核系统 — 生成式自动化评测报告</h1>
<p class="meta">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 · 数据集: {metrics.get('dataset_size')} cases
 · LLM 报告: {rt.get('report_llm', 0)} · 降级报告: {rt.get('report_degraded', 0)} · 失败 case: {rt.get('case_errors', 0)}
 · 参与生成式聚合: {gen.get('evaluated_cases')}</p>
<h2>一、确定性指标</h2>{deterministic_html}
<h2>二、生成式指标</h2>{generative_html}
<h2>三、逐 case 结果（{len(case_rows)}）</h2>{case_table}
<h2>四、Top 失败案例</h2>{failure_html}
</body></html>"""

    path = os.path.join(output_dir, "evaluation_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


# =============================================================================
# 四、汇总落盘
# =============================================================================

def write_outputs(
    metrics: dict,
    case_results: list[dict],
    output_dir: str,
    top_failures: int = 10,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    failures = build_top_failures(case_results, top_n=top_failures)
    paths = {
        "json": save_json(metrics, output_dir),
        "cases": save_cases(case_results, output_dir),
        "html": save_html(metrics, case_results, failures, output_dir),
    }
    return {"paths": paths, "failures": failures}
