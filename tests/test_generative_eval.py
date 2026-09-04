# =============================================================================
# tests/test_generative_eval.py — 生成式评测模块单元/集成测试
# =============================================================================
# 覆盖（对应验收标准）：
#   - Claim 提取（金额解析、锚定抽取、公式折叠、条款 ID 正则）
#   - 数字事实比较（容差）
#   - Citation 比较（存在性 / 支持性）
#   - Citation Completeness / Precision / Recall
#   - Hallucination 判定（不存在引用 / 非法 status / 无来源金额）
#   - Metric aggregation（确定性 + 生成式）
#   - LLM Judge 解析（严格 JSON、非法 label、judge_error 兜底）
#   - 集成：Agent→Report→Extraction→Verification→Metrics 全链路
#
# 运行：
#   pytest tests/test_generative_eval.py -v          # 快速（含降级报告集成）
#   pytest tests/test_generative_eval.py -v --runslow  # 含真实数据链路
# =============================================================================

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.claim_extractor import extract_claims, extract_citations
from evaluation.citation_verifier import verify_citations, valid_clause_ids
from evaluation.consistency import check_report_consistency, _within_tolerance
from evaluation.hallucination import detect_hallucinations, build_source_amounts
from evaluation.metrics import aggregate, compute_deterministic
from evaluation import llm_judge


# ===========================================================================
# 一、Claim 提取
# ===========================================================================

class TestClaimExtractor:
    def test_parse_amounts_in_summary(self):
        report = {
            "summary": "应缴金额为¥7,632.47，实际缴纳¥6,487.60，差额为¥1,144.87，申报营业额为¥52,893.10",
            "status": "abnormal",
            "findings": [], "suggestion": "", "contract_refs": [],
        }
        out = extract_claims(report)
        vals = {c["claim_type"]: c["value"] for c in out["claims"]}
        assert vals["payable"] == pytest.approx(7632.47)
        assert vals["paid"] == pytest.approx(6487.60)
        assert vals["difference"] == pytest.approx(1144.87)
        assert vals["revenue"] == pytest.approx(52893.10)
        assert out["status"] == "abnormal"

    def test_formula_folding(self):
        # 内联公式 "¥52,893.10×14.43%=¥7,632.47" 应折叠，避免营业额被误判为应缴金额
        report = {
            "summary": "计算应缴金额为¥52,893.10×14.43%=¥7,632.47。",
            "status": "abnormal", "findings": [], "suggestion": "", "contract_refs": [],
        }
        out = extract_claims(report)
        payable = next((c for c in out["claims"] if c["claim_type"] == "payable"), None)
        assert payable is not None
        assert payable["value"] == pytest.approx(7632.47)

    def test_status_invalid(self):
        out = extract_claims({"status": "可疑", "summary": "", "findings": [], "suggestion": ""})
        assert out["invalid_status"] is True
        assert out["status"] is None

    def test_citation_regex_no_substring(self):
        # "条款CTR-001-clause-001" 不应切出 "001-clause-001" 子串
        report = {
            "summary": "依据条款CTR-001-clause-001计算",
            "status": "normal",
            "findings": [{"description": "", "evidence": "条款CTR-001-clause-001：租金计算方式"}],
            "suggestion": "", "contract_refs": ["CTR-001-clause-001"],
        }
        assert extract_citations(report) == ["CTR-001-clause-001"]

    def test_missing_amounts(self):
        out = extract_claims({"summary": "无异常", "status": "normal", "findings": [], "suggestion": ""})
        assert out["claims"] == []


# ===========================================================================
# 二、数字比较 / 一致性
# ===========================================================================

class TestConsistency:
    def test_within_tolerance(self):
        assert _within_tolerance(7632.47, 7632.47, 0.01, 0.01)
        assert _within_tolerance(7630, 7632.47, 0.01, 0.01)   # 0.03% 差异 < 1%
        assert not _within_tolerance(8000, 7632.47, 0.01, 0.01)
        assert not _within_tolerance(7632.47, None, 0.01, 0.01)

    def test_all_correct(self):
        gt = {"status": "abnormal", "revenue": 52893.10, "payable": 7632.47,
              "paid": 6487.60, "difference": 1144.87}
        extracted = {
            "status": "abnormal",
            "claims": [
                {"claim_type": "revenue", "value": 52893.10},
                {"claim_type": "payable", "value": 7632.47},
                {"claim_type": "paid", "value": 6487.60},
                {"claim_type": "difference", "value": 1144.87},
            ],
        }
        r = check_report_consistency(gt, extracted)
        assert r["correct"] == 5 and r["wrong"] == 0 and r["missing"] == 0
        assert r["report_consistency"] == 1.0

    def test_wrong_and_missing(self):
        gt = {"status": "abnormal", "revenue": 52893.10, "payable": 7632.47,
              "paid": 6487.60, "difference": 1144.87}
        # 报告把应缴金额说成 8000（矛盾），且不陈述差额（缺失）
        extracted = {
            "status": "abnormal",
            "claims": [
                {"claim_type": "revenue", "value": 52893.10},
                {"claim_type": "payable", "value": 8000},
                {"claim_type": "paid", "value": 6487.60},
            ],
        }
        r = check_report_consistency(gt, extracted)
        states = {f["fact"]: f["state"] for f in r["facts"] if f["state"] != "n/a"}
        assert states["payable"] == "wrong"
        assert states["difference"] == "missing"
        assert states["status"] == "correct"
        # 5 个事实中 1 个矛盾 → 一致性 0.8
        assert r["report_consistency"] == pytest.approx(0.8)

    def test_difference_sign_flip_is_correct(self):
        """差额按绝对值比对：报告 -3899.52 与规则引擎 +3899.52 方向措辞差异不算矛盾。"""
        gt = {"status": "abnormal", "revenue": 1.0, "payable": 2.0, "paid": 3.0,
              "difference": -3899.52}
        extracted = {
            "status": "abnormal",
            "claims": [{"claim_type": "difference", "value": 3899.52}],
        }
        r = check_report_consistency(gt, extracted)
        diff_fact = next(f for f in r["facts"] if f["fact"] == "difference")
        assert diff_fact["state"] == "correct"
        assert r["wrong"] == 0


# ===========================================================================
# 三、Citation 校验
# ===========================================================================

CONTRACT = {
    "contract_id": "CTR-001",
    "merchant_id": "M001",
    "clauses": [
        {"clause_id": "CTR-001-clause-001", "clause_type": "rent_calculation"},
        {"clause_id": "CTR-001-clause-002", "clause_type": "submission_deadline"},
    ],
}


class TestCitationVerifier:
    def test_valid_and_supported(self):
        r = verify_citations(["CTR-001-clause-001"], CONTRACT, used_clause_ids=["CTR-001-clause-001"])
        assert r["total"] == 1 and r["valid"] == 1 and r["supported"] == 1
        assert r["truthfulness"] == 1.0 and r["completeness"] == 1.0

    def test_nonexistent_citation(self):
        r = verify_citations(["CTR-001-clause-999"], CONTRACT, used_clause_ids=["CTR-001-clause-001"])
        assert r["valid"] == 0
        assert r["truthfulness"] == 0.0
        assert not r["results"][0]["exists"]

    def test_completeness_precision_recall(self):
        # 必需证据 = {clause-001, clause-002}；只引用了 clause-001
        r = verify_citations(
            ["CTR-001-clause-001"],
            CONTRACT,
            used_clause_ids=["CTR-001-clause-001", "CTR-001-clause-002"],
        )
        assert r["completeness"] == pytest.approx(0.5)   # 1/2
        assert r["recall"] == pytest.approx(0.5)
        assert r["precision"] == pytest.approx(1.0)      # 引用 1 条且必需

    def test_no_citations(self):
        r = verify_citations([], CONTRACT, used_clause_ids=["CTR-001-clause-001"])
        assert r["total"] == 0
        assert r["truthfulness"] == 1.0
        assert r["completeness"] == 0.0
        assert r["precision"] == 0.0

    def test_valid_clause_ids(self):
        assert valid_clause_ids(CONTRACT) == {"CTR-001-clause-001", "CTR-001-clause-002"}


# ===========================================================================
# 四、Hallucination 判定
# ===========================================================================

class TestHallucination:
    GT = {"status": "abnormal", "revenue": 52893.10, "payable": 7632.47,
          "paid": 6487.60, "difference": 1144.87}

    def test_nonexistent_citation(self):
        extracted = {
            "citations": ["CTR-001-clause-999"],
            "amounts": [], "claims": [], "invalid_status": False,
        }
        r = detect_hallucinations(extracted, self.GT, CONTRACT,
                                  valid_clause_ids=valid_clause_ids(CONTRACT))
        assert r["hallucinated"] == 1
        assert r["events"][0]["type"] == "nonexistent_citation"

    def test_invalid_status(self):
        extracted = {
            "citations": [], "amounts": [], "claims": [],
            "invalid_status": True, "raw_status": "可疑",
        }
        r = detect_hallucinations(extracted, self.GT, CONTRACT,
                                  valid_clause_ids=valid_clause_ids(CONTRACT))
        assert any(e["type"] == "invalid_status" for e in r["events"])

    def test_extra_amount_without_source(self):
        extracted = {
            "citations": [], "claims": [],
            "amounts": [{"value": 5000.0, "text": "额外费用¥5,000.00", "source": "summary"}],
            "invalid_status": False,
        }
        r = detect_hallucinations(extracted, self.GT, CONTRACT,
                                  valid_clause_ids=valid_clause_ids(CONTRACT))
        assert any(e["type"] == "hallucinated_number" for e in r["events"])

    def test_known_amounts_not_hallucination(self):
        # 与来源一致的金额不判幻觉
        extracted = {
            "citations": ["CTR-001-clause-001"], "claims": [
                {"claim_type": "payable", "value": 7632.47},
            ],
            "amounts": [{"value": 7632.47, "text": "应缴¥7,632.47", "source": "summary"},
                        {"value": 1144.87, "text": "差额¥1,144.87", "source": "suggestion"}],
            "invalid_status": False,
        }
        r = detect_hallucinations(extracted, self.GT, CONTRACT,
                                  valid_clause_ids=valid_clause_ids(CONTRACT))
        assert r["hallucinated"] == 0

    def test_source_amounts_include_contract_params(self):
        hybrid = {"contract_id": "CTR-002", "min_guarantee": 10000.0, "commission_rate": 0.1}
        gt = {"revenue": 50000, "paid": 10000, "payable": 10000, "difference": 0}
        sources = build_source_amounts(gt, hybrid)
        assert 10000.0 in sources          # min_guarantee
        assert 100000.0 in sources         # 达标线 10000/0.1


# ===========================================================================
# 五、指标聚合
# ===========================================================================

class TestMetrics:
    def test_deterministic(self):
        cases = [
            {"status": "ok", "ground_truth": {"status": "abnormal", "payable": 100, "paid": 80},
             "truth_label": {"anomaly": True, "anomaly_type": "金额不符"}},
            {"status": "ok", "ground_truth": {"status": "normal", "payable": 100, "paid": 100},
             "truth_label": {"anomaly": False, "anomaly_type": ""}},
            {"status": "ok", "ground_truth": {"status": "normal", "payable": 100, "paid": 100},
             "truth_label": {"anomaly": True, "anomaly_type": "少报营业额"}},  # FN（规则引擎无法检出）
        ]
        r = compute_deterministic(cases)
        assert r["anomaly_recall"] == pytest.approx(0.5)     # 2 个异常检出 1
        assert r["amount_accuracy"] == pytest.approx(1.0)    # 少报排除后 2 条全对

    def test_generative_aggregate(self):
        def mk_case(correct, wrong, missing, cit_total, cit_valid, cit_recalled, cit_req, hall, hall_total):
            return {
                "report_mode": "llm",
                "consistency": {"correct": correct, "wrong": wrong, "missing": missing},
                "citation": {"total": cit_total, "valid": cit_valid, "recalled": cit_recalled,
                             "required_count": cit_req},
                "hallucination": {"hallucinated": hall, "total_claims": hall_total},
                "metrics": {"report_consistency": 0.9},
            }
        results = [
            mk_case(correct=4, wrong=0, missing=1, cit_total=2, cit_valid=2,
                    cit_recalled=1, cit_req=2, hall=0, hall_total=5),
            mk_case(correct=3, wrong=1, missing=1, cit_total=1, cit_valid=0,
                    cit_recalled=0, cit_req=1, hall=1, hall_total=4),
        ]
        gen = aggregate([], results)["generative"]
        assert gen["evaluated_cases"] == 2
        # claim: correct=7, wrong=1, missing=2, total=10
        assert gen["claim_accuracy"] == pytest.approx(0.7)
        assert gen["claim_precision"] == pytest.approx(7 / 8)
        assert gen["claim_recall"] == pytest.approx(0.7)
        # citation: total=3, valid=2, recalled=1, required=3
        # （指标在 metrics.py 内 round 到 4 位小数，故用 abs 容差）
        assert gen["citation_truthfulness"] == pytest.approx(2 / 3, abs=1e-4)
        assert gen["citation_completeness"] == pytest.approx(1 / 3, abs=1e-4)
        assert gen["citation_precision"] == pytest.approx(1 / 3, abs=1e-4)
        assert gen["citation_recall"] == pytest.approx(1 / 3, abs=1e-4)
        # hallucination: 1/9
        assert gen["hallucination_rate"] == pytest.approx(1 / 9, abs=1e-4)


# ===========================================================================
# 六、LLM Judge 解析（无网络）
# ===========================================================================

class TestLlmJudge:
    def test_parse_valid(self):
        r = llm_judge._parse_judge_response(
            '{"label": "entailment", "confidence": 0.95, "reason": "一致"}')
        assert r["label"] == "entailment"
        assert r["confidence"] == 0.95

    def test_parse_markdown_wrapped(self):
        r = llm_judge._parse_judge_response(
            '```json\n{"label": "contradiction", "confidence": 0.8, "reason": "矛盾"}\n```')
        assert r["label"] == "contradiction"

    def test_parse_invalid_label(self):
        with pytest.raises(ValueError):
            llm_judge._parse_judge_response('{"label": "maybe", "confidence": 0.5}')

    def test_judge_error_fallback(self, monkeypatch):
        def _raise(*a, **k):
            raise RuntimeError("LLM down")
        # judge_entailment 在调用时 `from src.agent_graph import _call_llm`，
        # 因此补丁模块属性即可使重试全部失败 → judge_error 兜底
        import src.agent_graph as ag
        monkeypatch.setattr(ag, "_call_llm", _raise)
        r = llm_judge.judge_entailment("前提", "假设", retries=1)
        assert r["judge_error"] is True
        assert r["label"] == "unknown"


# ===========================================================================
# 七、集成测试
# ===========================================================================

def _fake_case(merchant="M001", month="2025-01"):
    """构造一个与 dataset.build_cases 同构的 case（Hermetic，不依赖真实数据/模型）。"""
    return {
        "case_id": "C001",
        "input": {"merchant_id": merchant, "month": month},
        "contract_type": "commission",
        "contract": {
            "contract_id": "CTR-001", "type": "commission",
            "merchant_name": "云松咖啡", "commission_rate": 0.1443,
            "clauses": [
                {"clause_id": "CTR-001-clause-001", "clause_type": "rent_calculation",
                 "description": "租金计算方式：月租金 = 申报营业额 × 14.43%"},
                {"clause_id": "CTR-001-clause-002", "clause_type": "submission_deadline",
                 "description": "商户需在每月5日前提交上月营业报表"},
            ],
        },
        "revenue_data": {"merchant_id": merchant, "month": month,
                         "reported_revenue": 52893.10, "paid_amount": 6487.60,
                         "submit_date": "2025-02-02"},
        "clause_data": [],
        "issues": [{"merchant_id": merchant, "month": month, "issue_type": "amount_mismatch",
                    "clause_id": "CTR-001-clause-001", "expected_value": 7632.47,
                    "actual_value": 6487.60,
                    "description": "金额不匹配：实缴 ¥6487.60，合同计算应有 ¥7632.47",
                    "severity": "high"}],
        "audit_result": {"merchant_id": merchant, "month": month, "contract_type": "commission",
                         "status": "abnormal",
                         "issues": [{"issue_type": "amount_mismatch",
                                     "clause_id": "CTR-001-clause-001",
                                     "expected_value": 7632.47, "actual_value": 6487.60}],
                         "steps": [], "summary": "发现异常"},
        "ground_truth": {"status": "abnormal", "revenue": 52893.10, "payable": 7632.47,
                         "paid": 6487.60, "difference": 1144.87,
                         "reason": ["金额不符"], "used_clause_ids": ["CTR-001-clause-001"],
                         "source": "rule_engine"},
        # 审计证据条款 = 规则引擎实际核实的条款（金额 + 提交日期）
        "required_evidence": ["CTR-001-clause-001", "CTR-001-clause-002"],
        "truth_label": {"anomaly": True, "anomaly_type": "金额不符"},
        "status": "ok",
    }


class TestIntegration:
    def test_full_pipeline_deterministic_report(self):
        """降级报告链路：Agent→Report→Extraction→Verification→Metrics（无 LLM、无模型）。"""
        from evaluation.runner import evaluate_case
        from evaluation.cache import CacheStore
        from evaluation.metrics import aggregate

        cache = CacheStore("/tmp/eval_test_cache", enabled=False)
        config = {"use_llm_report": False, "amount_tolerance": 0.01, "min_anomaly_amount": 0.01}

        case_results = []
        for case in [_fake_case()]:
            r = evaluate_case(case, config, cache)
            assert r["status"] == "ok", r.get("error")
            assert r["report_mode"] in ("llm", "degraded")
            assert r["report"] is not None
            # 降级报告：status/payable 正确，金额事实被抽取
            assert r["metrics"]["report_consistency"] == 1.0
            assert r["citation"]["truthfulness"] == 1.0
            case_results.append(r)

        metrics = aggregate([_fake_case()], case_results)
        assert metrics["deterministic"]["anomaly_f1"] == 1.0

    @pytest.mark.slow
    def test_full_pipeline_real_data(self):
        """真实数据链路（需加载 BGE 模型）：build_cases → evaluate_case → aggregate。"""
        from evaluation.runner import evaluate_case
        from evaluation.cache import CacheStore
        from evaluation.dataset import build_cases
        from evaluation.metrics import aggregate

        cache = CacheStore("/tmp/eval_test_cache_real", enabled=False)
        config = {"use_llm_report": False, "amount_tolerance": 0.01, "min_anomaly_amount": 0.01}

        cases = build_cases(max_cases=3)
        assert len(cases) == 3
        case_results = [evaluate_case(c, config, cache) for c in cases]
        assert all(r["status"] == "ok" for r in case_results)
        metrics = aggregate(cases, case_results)
        assert metrics["dataset_size"] == 3
        assert "deterministic" in metrics and "generative" in metrics


# =============================================================================
# TestBoundary —— 边界补充集（合成账单，覆盖主数据集缺失的边界维度）
# =============================================================================
class TestBoundary:
    def test_build_200_cases_all_ok(self):
        """边界集应恰好 200 条且全部构建成功。"""
        from evaluation.dataset import build_boundary_cases
        cases = build_boundary_cases()
        assert len(cases) == 200
        assert all(c["status"] == "ok" for c in cases)

    def test_rule_engine_matches_expected_status(self):
        """规则引擎对全部 200 条边界账单的判定应与领域预期一致（容差两侧/宽限两侧等）。"""
        from evaluation.dataset import build_boundary_cases
        cases = build_boundary_cases()
        for c in cases:
            gt_status = c["ground_truth"]["status"]
            expected = "abnormal" if c["truth_label"]["anomaly"] else "normal"
            assert gt_status == expected, f"{c['case_id']} {c['boundary_scenario']}: {gt_status} != {expected}"

    def test_amount_accuracy_perfect_on_boundary(self):
        """确定性指标：边界集应缴金额核对准确率 100%、异常识别 F1 100%。"""
        from evaluation.dataset import build_boundary_cases
        from evaluation.metrics import compute_deterministic
        cases = build_boundary_cases()
        det = compute_deterministic(cases, tolerance=0.01, min_amt=0.01)
        assert det["amount_accuracy"] == 1.0
        assert det["anomaly_f1"] == 1.0
        assert det["confusion_matrix"] == {"tp": 95, "fp": 0, "fn": 0, "tn": 105}

    def test_boundary_scenarios_cover_gaps(self):
        """边界集应覆盖此前缺失的维度：近容差金额、宽限期提交、零营业额、缺失提交日期、多缴。"""
        from evaluation.dataset import build_boundary_cases
        scenarios = [c.get("boundary_scenario", "") for c in build_boundary_cases()]
        assert any("amount_under" in s and "just" in s for s in scenarios)     # 近容差两侧
        assert any("submit_late" in s and "grace" in s for s in scenarios)      # 宽限内提交
        assert any("zero_revenue" in s for s in scenarios)                      # 零营业额
        assert any("missing_submit" in s for s in scenarios)                    # 缺失提交日期
        assert any("overpaid" in s for s in scenarios)                          # 多缴

    def test_clause_data_exposes_all_clause_ids(self):
        """边界 case 的 clause_data 必须带全部 3 个条款 ID，否则 LLM 无法引用日期条款。"""
        from evaluation.dataset import build_boundary_cases
        cid = next(c for c in build_boundary_cases() if c["case_id"] == "B005")
        clause_ids = {cl["clause_id"] for cl in cid["clause_data"]}
        assert "CTR-BM001-clause-001" in clause_ids
        assert "CTR-BM001-clause-002" in clause_ids

    def test_missing_submit_has_single_evidence(self):
        """缺失提交日期的账单，required_evidence 仅含金额条款（无日期校验）。"""
        from evaluation.dataset import build_boundary_cases
        cases = build_boundary_cases()
        miss = [c for c in cases if "missing_submit" in c.get("boundary_scenario", "")]
        assert len(miss) == 3
        assert all(c["required_evidence"] == [c["contract"]["contract_id"] + "-clause-001"]
                   for c in miss)
