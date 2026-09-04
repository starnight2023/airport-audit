# =============================================================================
# evaluation/runner.py — 生成式自动化评测主入口
# =============================================================================
# 管线：
#   build_cases (600) → 逐 case 生成报告（缓存） → Claim/Citation 抽取（确定性）
#   → 一致性 / 引用 / 幻觉判定 → 指标聚合 → JSON/HTML/逐 case 输出 → Top 失败案例
#
# 运行：
#   python -m evaluation.runner                    # 全量 600
#   python -m evaluation.runner --cases 20         # 前 20 条
#   python -m evaluation.runner --merchant M001    # 单商户 12 月
#   python -m evaluation.runner --merchant M001 --month 2025-01   # 单 case 调试
#   python -m evaluation.runner --no-report        # 确定性降级报告跑通链路（不调 LLM）
#
# 工程要求：
#   - 某 case 失败不中断整体；失败记入结果并继续
#   - LLM 报告结果落盘缓存（evaluation/cache/），支持断点续跑与重跑
#   - 报告生成走 audit_mock + report_node，不调用与评测无关的 Planner LLM
#     （LLM 调用从 2 次/case 压缩为 1 次/case）
# =============================================================================

import argparse
import contextlib
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from evaluation.cache import CacheStore
from evaluation.dataset import build_boundary_cases, build_cases
from evaluation.claim_extractor import extract_claims
from evaluation.citation_verifier import verify_citations, valid_clause_ids
from evaluation.consistency import check_report_consistency
from evaluation.hallucination import detect_hallucinations
from evaluation.metrics import aggregate
from evaluation import report as report_mod

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("evaluation", {})


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _disable_llm():
    """临时禁用 agent_graph._call_llm，使 report_node 走确定性降级报告。"""
    import src.agent_graph as ag
    original = ag._call_llm

    def _raise(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("LLM 已禁用（确定性模式）")

    ag._call_llm = _raise
    try:
        yield
    finally:
        ag._call_llm = original


def _state_for_case(case: dict) -> dict:
    return {
        "merchant_id": case["input"]["merchant_id"],
        "month": case["input"]["month"],
        "contract": case.get("contract"),
        "revenue_data": case.get("revenue_data"),
        "clause_data": case.get("clause_data", []),
        "issues": case.get("issues", []),
        "audit_result": case.get("audit_result"),
        "tool_errors": [],
    }


def generate_report(case: dict, use_llm: bool, cache: CacheStore) -> tuple[dict, str]:
    """
    生成报告并返回 (report, mode)。mode ∈ {"llm", "degraded"}。
    优先读缓存（use_llm 时），避免重复调用昂贵 LLM。
    """
    mid = case["input"]["merchant_id"]
    month = case["input"]["month"]
    key = f"report_{mid}_{month}"

    if use_llm:
        cached = cache.get(key)
        if cached is not None and cached.get("mode") == "llm":
            return cached["report"], "llm"

    from src.agent_graph import report_node

    if use_llm:
        out = report_node(_state_for_case(case))
    else:
        with _disable_llm():
            out = report_node(_state_for_case(case))

    report_obj = out.get("report") or {}
    trace = (out.get("trace") or [{}])[0]
    mode = "degraded" if trace.get("status") == "degraded" else "llm"

    if use_llm:
        cache.set(key, {"mode": mode, "report": report_obj})

    return report_obj, mode


# ---------------------------------------------------------------------------
# 单 case 评测
# ---------------------------------------------------------------------------

def evaluate_case(case: dict, config: dict, cache: CacheStore) -> dict:
    """对单个 case 执行完整评测链路，失败不抛异常（记入结果）。"""
    if case.get("status") != "ok":
        return {
            "case_id": case.get("case_id"),
            "input": case.get("input"),
            "status": "error",
            "report_mode": "none",
            "error": case.get("error", "case 构建失败"),
        }

    result = {
        "case_id": case["case_id"],
        "input": case["input"],
        "contract_type": case.get("contract_type"),
        "ground_truth": case.get("ground_truth"),
        "required_evidence": case.get("required_evidence"),
        "truth_label": case.get("truth_label"),
        "status": "ok",
    }

    # 1. 生成报告
    try:
        report_obj, mode = generate_report(
            case,
            use_llm=config.get("use_llm_report", True),
            cache=cache,
        )
    except Exception as e:  # noqa: BLE001
        result.update({
            "status": "error",
            "report_mode": "none",
            "error": f"报告生成异常: {e}",
        })
        return result

    result["report_mode"] = mode
    result["report"] = report_obj

    # 2. 抽取 + 判定（纯确定性）
    try:
        extracted = extract_claims(report_obj)
        contract = case.get("contract") or {}
        gt = case.get("ground_truth") or {}

        consistency = check_report_consistency(
            gt, extracted,
            tolerance=config.get("amount_tolerance", 0.01),
            min_anomaly_amount=config.get("min_anomaly_amount", 0.01),
        )
        citation = verify_citations(
            extracted.get("citations", []),
            contract,
            case.get("required_evidence", []),
        )
        hallucination = detect_hallucinations(
            extracted, gt, contract,
            valid_clause_ids=valid_clause_ids(contract),
            tolerance=config.get("amount_tolerance", 0.01),
            min_anomaly_amount=config.get("min_anomaly_amount", 0.01),
        )
    except Exception as e:  # noqa: BLE001
        result.update({
            "status": "error",
            "report_mode": mode,
            "error": f"抽取/判定异常: {e}",
        })
        return result

    # 3. 逐 case 指标
    result.update({
        "claims": extracted.get("claims", []),
        "citations": extracted.get("citations", []),
        "claim_results": consistency.get("facts", []),
        "citation_results": citation.get("results", []),
        "hallucination_events": hallucination.get("events", []),
        "consistency": consistency,
        "citation": citation,
        "hallucination": hallucination,
        "metrics": {
            "report_consistency": consistency.get("report_consistency"),
            "claim_recall_case": consistency.get("coverage"),
            "citation_truthfulness": citation.get("truthfulness"),
            "citation_completeness": citation.get("completeness"),
            "hallucination_count": hallucination.get("hallucinated"),
        },
    })
    return result


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="生成式部分自动化评测")
    parser.add_argument("--cases", type=int, default=None, help="限制评测 case 数（默认全部）")
    parser.add_argument("--merchant", type=str, default=None, help="只评测指定商户")
    parser.add_argument("--month", type=str, default=None, help="只评测指定月份")
    parser.add_argument("--no-report", action="store_true", help="确定性降级报告跑通链路（不调 LLM）")
    parser.add_argument("--boundary", action="store_true",
                        help="评测边界补充集（25 条合成账单：金额容差/提交宽限期/零营业额/缺失提交日期/多缴）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认 config.output_dir）")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="评测配置文件")
    parser.add_argument("--refresh-cache", action="store_true", help="忽略报告缓存，重新生成")
    parser.add_argument("--verbose", action="store_true", help="打印逐 case 详情")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_report:
        config["use_llm_report"] = False
    if args.output_dir:
        config["output_dir"] = args.output_dir

    cache_dir = config.get("cache_dir", "evaluation/cache")
    output_dir = config.get("output_dir", "evaluation/output")
    cache = CacheStore(cache_dir, enabled=config.get("cache_enabled", True))
    if args.refresh_cache:
        cache.clear()

    print("=" * 70)
    print("📊 生成式部分自动化评测")
    print("=" * 70)
    mode_label = "边界补充集（合成账单）" if args.boundary else "全量/过滤数据集"
    print(f"  模式: {mode_label}")
    print(f"  数据缓存: {cache_dir} | 输出: {output_dir} | LLM 报告: {'ON' if config.get('use_llm_report', True) else 'OFF'}")

    # ---- 构建数据集（可落盘缓存，重跑无需重载模型/重检索） ----
    # 仅当无任何过滤条件且不限制条数时才复用 600 条全量缓存
    data_cache_key = "dataset_cases.json"
    t0 = time.time()
    if args.boundary:
        # 边界补充集：数据来自 data/contracts_boundary.json + bills_boundary.csv，
        # 独立于主 600 条，不复用主集缓存（报告走 report_{mid}_{month} 键，无冲突）。
        cases = build_boundary_cases(verbose=args.verbose)
        print(f"  🏗  构建边界补充集: {len(cases)} cases（{time.time()-t0:.1f}s）")
    else:
        use_data_cache = (
            config.get("cache_enabled", True)
            and not args.refresh_cache
            and args.cases is None
            and not args.merchant
            and not args.month
        )
        cached_cases = cache.get(data_cache_key) if use_data_cache else None

        if cached_cases is not None:
            cases = cached_cases
            print(f"  📥 从缓存加载数据集: {len(cases)} cases")
        else:
            cases = build_cases(
                max_cases=args.cases,
                merchants=[args.merchant] if args.merchant else None,
                months=[args.month] if args.month else None,
                verbose=args.verbose,
            )
            print(f"  🏗  构建数据集: {len(cases)} cases（{time.time()-t0:.1f}s）")
            if use_data_cache:
                cache.set(data_cache_key, cases)

    # ---- 逐 case 评测 ----
    print("\n  🔍 逐 case 评测（报告生成 + 抽取 + 判定）...")
    t1 = time.time()
    case_results = []
    total = len(cases)
    for i, case in enumerate(cases, 1):
        if args.verbose:
            print(f"    [{i}/{total}] {case['case_id']} {case.get('input')}")
        case_results.append(evaluate_case(case, config, cache))
        if i % 100 == 0:
            print(f"    已处理 {i}/{total}（{time.time()-t1:.1f}s）")

    # ---- 聚合 + 输出 ----
    metrics = aggregate(cases, case_results)
    output = report_mod.write_outputs(metrics, case_results, output_dir, top_failures=10)

    # ---- 终端摘要 ----
    det = metrics["deterministic"]
    gen = metrics["generative"]
    rt = metrics["runtime"]
    print("\n" + "=" * 70)
    print("📈 评测摘要")
    print("=" * 70)
    print(f"  数据集: {metrics['dataset_size']} | LLM 报告: {rt['report_llm']} | "
          f"降级: {rt['report_degraded']} | 失败: {rt['case_errors']} | 生成式聚合: {gen.get('evaluated_cases')}")
    print(f"\n  【确定性】")
    print(f"    应缴金额核对准确率: {det['amount_accuracy']:.2%} ({det['amount_checked_bills']} 条)")
    print(f"    异常识别 F1: {det['anomaly_f1']:.2%} (P={det['anomaly_precision']:.2%} R={det['anomaly_recall']:.2%})")
    print(f"  【生成式】")
    print(f"    Report Consistency:  {_pct(gen.get('report_consistency'))}")
    print(f"    Claim Accuracy:      {_pct(gen.get('claim_accuracy'))}")
    print(f"    Claim Precision:     {_pct(gen.get('claim_precision'))}")
    print(f"    Claim Recall:        {_pct(gen.get('claim_recall'))}")
    print(f"    Citation Truthfulness:{_pct(gen.get('citation_truthfulness'))}")
    print(f"    Citation Completeness:{_pct(gen.get('citation_completeness'))}")
    print(f"    Citation Precision:  {_pct(gen.get('citation_precision'))}")
    print(f"    Citation Recall:     {_pct(gen.get('citation_recall'))}")
    print(f"    Hallucination Rate:  {_pct(gen.get('hallucination_rate'))}")

    print("\n  📁 输出文件:")
    for name, path in output["paths"].items():
        print(f"    {name}: {path}")

    report_mod.print_top_failures(output["failures"])
    print(f"\n✅ 评测完成（总耗时 {time.time()-t0:.1f}s）")


def _pct(value):
    return "n/a" if value is None else f"{value:.2%}"


if __name__ == "__main__":
    main()
