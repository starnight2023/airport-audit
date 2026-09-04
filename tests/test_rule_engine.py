# =============================================================================
# tests/test_rule_engine.py — 规则引擎单元测试
# =============================================================================
# 运行方式：cd 到项目根目录，执行 pytest tests/ -v
# =============================================================================
#
# 【测试策略】
# Phase 1 的测试分三层：
#   1. 单元测试（本文件）—— 对单个校验函数做边界条件测试
#   2. 集成测试（test_integration）—— 模拟生成数据→规则引擎全链路
#   3. 评测脚本（scripts/quick_eval.py）—— 对标注数据做定量指标计算

import os
import sys
import json
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

try:
    from models import (
        ContractType, AuditResult, AuditIssue, AuditStatus, IssueSeverity,
    )
    from rule_engine import (
        load_rules_config,
        check_paid_amount,
        check_submission_date,
        audit_single,
    )
except ImportError as e:
    print(f"⚠ 导入错误: {e}")
    print("请确保在项目根目录运行 pytest")
    sys.exit(1)


# ===========================================================================
# 测试夹具（Fixture）
# ===========================================================================

@pytest.fixture
def sample_contracts() -> dict[str, dict]:
    """创建三个合同类型的样例合同"""
    return {
        "fixed": {
            "contract_id": "CTR-TEST-001",
            "merchant_id": "M-TEST",
            "merchant_name": "测试商户A",
            "type": "fixed",
            "fixed_amount": 10000.0,
            "commission_rate": 0.0,
            "min_guarantee": None,
            "submit_deadline_day": 5,
        },
        "commission": {
            "contract_id": "CTR-TEST-002",
            "merchant_id": "M-TEST",
            "merchant_name": "测试商户B",
            "type": "commission",
            "fixed_amount": None,
            "commission_rate": 0.12,
            "min_guarantee": None,
            "submit_deadline_day": 5,
        },
        "hybrid": {
            "contract_id": "CTR-TEST-003",
            "merchant_id": "M-TEST",
            "merchant_name": "测试商户C",
            "type": "hybrid",
            "fixed_amount": None,
            "commission_rate": 0.15,
            "min_guarantee": 8000.0,
            "submit_deadline_day": 5,
        },
    }


@pytest.fixture
def default_config() -> dict:
    """加载默认规则配置"""
    return load_rules_config(
        os.path.join(PROJECT_ROOT, "config", "rules.yaml")
    )


# ===========================================================================
# 一、check_paid_amount 单元测试
# ===========================================================================

class TestCheckPaidAmount:
    """测试金额校验核心函数"""

    def test_fixed_match(self, sample_contracts):
        """固定租金：金额一致 → 通过"""
        match, expected, clause_id = check_paid_amount(
            sample_contracts["fixed"], 0, 10000.0
        )
        assert match is True
        assert expected == 10000.0

    def test_fixed_mismatch(self, sample_contracts):
        """固定租金：金额不一致 → 不通过"""
        match, expected, clause_id = check_paid_amount(
            sample_contracts["fixed"], 0, 9500.0
        )
        assert match is False
        assert expected == 10000.0

    def test_commission_match(self, sample_contracts):
        """提成合同：金额一致 → 通过"""
        # reported_revenue=50000, rate=0.12 → expected=6000
        match, expected, clause_id = check_paid_amount(
            sample_contracts["commission"], 50000.0, 6000.0
        )
        assert match is True
        assert expected == 6000.0

    def test_commission_mismatch(self, sample_contracts):
        """提成合同：金额不一致 → 不通过"""
        match, expected, clause_id = check_paid_amount(
            sample_contracts["commission"], 50000.0, 5500.0
        )
        assert match is False
        assert expected == 6000.0

    def test_hybrid_min_guarantee_applied(self, sample_contracts):
        """
        保底+提成：营业额低，保底额生效
        revenue=30000, rate=0.15 → commission=4500
        min_guarantee=8000 > 4500 → expected=8000
        """
        match, expected, clause_id = check_paid_amount(
            sample_contracts["hybrid"], 30000.0, 8000.0
        )
        assert match is True
        assert expected == 8000.0

    def test_hybrid_commission_applied(self, sample_contracts):
        """
        保底+提成：营业额高，提成生效
        revenue=80000, rate=0.15 → commission=12000
        min_guarantee=8000 < 12000 → expected=12000
        """
        match, expected, clause_id = check_paid_amount(
            sample_contracts["hybrid"], 80000.0, 12000.0
        )
        assert match is True
        assert expected == 12000.0

    def test_hybrid_commission_mismatch(self, sample_contracts):
        """
        保底+提成：应付提成12000，实缴10000 → 不通过
        """
        match, expected, clause_id = check_paid_amount(
            sample_contracts["hybrid"], 80000.0, 10000.0
        )
        assert match is False
        assert expected == 12000.0

    def test_tolerance_allows_small_diff(self, sample_contracts):
        """
        容差测试：1%以内的差异应通过
        fixed_amount=10000, paid=10050, diff=0.5% → 通过
        """
        match, expected, clause_id = check_paid_amount(
            sample_contracts["fixed"], 0, 10050.0
        )
        assert match is True, "0.5% 差异应在容差范围内"

    def test_tolerance_rejects_large_diff(self, sample_contracts):
        """
        容差测试：超过1%的差异应不通过
        fixed_amount=10000, paid=10200, diff=2% → 不通过
        """
        match, expected, clause_id = check_paid_amount(
            sample_contracts["fixed"], 0, 10200.0
        )
        assert match is False, "2% 差异应超出容差范围"


# ===========================================================================
# 二、check_submission_date 单元测试
# ===========================================================================

class TestCheckSubmissionDate:
    """测试提交日期校验函数"""

    def test_on_time_submission(self):
        """正常提交：下月3号 → 通过"""
        on_time, deadline, clause_id = check_submission_date(
            "2024-02-03", "2024-01"
        )
        assert on_time is True

    def test_deadline_day_submission(self):
        """截止日当天提交 → 通过"""
        on_time, deadline, clause_id = check_submission_date(
            "2024-02-05", "2024-01"
        )
        assert on_time is True

    def test_grace_period_submission(self):
        """宽限期内提交 → 通过（deadline=5, grace=3, 即8号前）"""
        on_time, deadline, clause_id = check_submission_date(
            "2024-02-08", "2024-01"
        )
        assert on_time is True

    def test_late_submission(self):
        """超期提交：下月10号 → 不通过"""
        on_time, deadline, clause_id = check_submission_date(
            "2024-02-10", "2024-01"
        )
        assert on_time is False

    def test_december_cross_year(self):
        """跨年测试：12月账单，截止日在下年1月"""
        on_time, deadline, clause_id = check_submission_date(
            "2025-01-04", "2024-12"
        )
        assert on_time is True

        on_time, deadline, clause_id = check_submission_date(
            "2025-01-10", "2024-12"
        )
        assert on_time is False

    def test_invalid_date_format(self):
        """异常格式：日期格式错误 → 视为逾期"""
        on_time, deadline, clause_id = check_submission_date(
            "invalid", "2024-01"
        )
        assert on_time is False


# ===========================================================================
# 三、audit_single 集成测试
# ===========================================================================

class TestAuditSingle:
    """测试完整的单账单稽核流水线"""

    def test_fixed_normal(self, sample_contracts):
        """固定租金：正常账单 → 状态 normal"""
        result = audit_single(
            contract=sample_contracts["fixed"],
            reported_revenue=0,
            paid_amount=10000.0,
            submit_date_str="2024-02-03",
            month="2024-01",
            merchant_id="M-TEST",
        )
        assert result.status == AuditStatus.NORMAL
        assert len(result.steps) >= 2  # 至少合同识别+金额+日期
        assert len(result.issues) == 0

    def test_fixed_amount_mismatch(self, sample_contracts):
        """固定租金：金额错误 → 状态 abnormal"""
        result = audit_single(
            contract=sample_contracts["fixed"],
            reported_revenue=0,
            paid_amount=8000.0,  # 少交了2000
            submit_date_str="2024-02-03",
            month="2024-01",
            merchant_id="M-TEST",
        )
        assert result.status == AuditStatus.ABNORMAL
        issue_types = [i.issue_type for i in result.issues]
        assert "amount_mismatch" in issue_types

    def test_fixed_late_submission(self, sample_contracts):
        """固定租金：逾期提交 → 状态 abnormal"""
        result = audit_single(
            contract=sample_contracts["fixed"],
            reported_revenue=0,
            paid_amount=10000.0,
            submit_date_str="2024-02-10",  # 逾期
            month="2024-01",
            merchant_id="M-TEST",
        )
        assert result.status == AuditStatus.ABNORMAL
        issue_types = [i.issue_type for i in result.issues]
        assert "late_submission" in issue_types

    def test_commission_revenue_underreport(self, sample_contracts):
        """
        提成合同：少报营业额 → 金额不匹配
        实际营业额80000，少报为64000（80%），
        实际应付=80000×0.12=9600，少报应付=64000×0.12=7680，
        如果实缴9600（按真实额），则与基于少报额的计算不匹配
        """
        result = audit_single(
            contract=sample_contracts["commission"],
            reported_revenue=64000.0,  # 少报
            paid_amount=9600.0,        # 按真实额缴纳
            submit_date_str="2024-02-03",
            month="2024-01",
            merchant_id="M-TEST",
        )
        # 基于 reported_revenue=64000, rate=0.12 → 应有=7680
        # 实际 paid=9600 != 7680 → amount_mismatch
        assert result.status == AuditStatus.ABNORMAL
        issue_types = set(i.issue_type for i in result.issues)
        assert "amount_mismatch" in issue_types

    def test_hybrid_min_guarantee_exceeded(self, sample_contracts):
        """
        保底合同：营业额未达标时仍按保底收取
        revenue=20000, rate=0.15 → commission=3000
        min_guarantee=8000 → 应付8000
        """
        result = audit_single(
            contract=sample_contracts["hybrid"],
            reported_revenue=20000.0,
            paid_amount=8000.0,
            submit_date_str="2024-02-03",
            month="2024-01",
            merchant_id="M-TEST",
        )
        assert result.status == AuditStatus.NORMAL

    def test_hybrid_below_min_guarantee_payment(self, sample_contracts):
        """
        保底合同：营业额未达标但商户只交了提成部分
        revenue=20000, rate=0.15 → commission=3000
        应付=8000, 实缴=3000 → 异常
        """
        result = audit_single(
            contract=sample_contracts["hybrid"],
            reported_revenue=20000.0,
            paid_amount=3000.0,  # 少交了5000
            submit_date_str="2024-02-03",
            month="2024-01",
            merchant_id="M-TEST",
        )
        assert result.status == AuditStatus.ABNORMAL

    def test_check_steps_order(self, sample_contracts):
        """验证检查步骤的顺序和完整性"""
        result = audit_single(
            contract=sample_contracts["fixed"],
            reported_revenue=0,
            paid_amount=10000.0,
            submit_date_str="2024-02-03",
            month="2024-01",
            merchant_id="M-TEST",
        )
        step_names = [s.step_name for s in result.steps]
        assert "合同类型识别" in step_names
        assert "金额校验" in step_names
        assert "提交日期校验" in step_names
        # 确保顺序
        type_idx = step_names.index("合同类型识别")
        amount_idx = step_names.index("金额校验")
        date_idx = step_names.index("提交日期校验")
        assert type_idx < amount_idx < date_idx, "步骤顺序错误"


# ===========================================================================
# 四、边界条件测试
# ===========================================================================

class TestEdgeCases:
    """边界条件测试"""

    def test_zero_revenue(self, sample_contracts):
        """零营业额：commission 合同应付0"""
        match, expected, clause_id = check_paid_amount(
            sample_contracts["commission"], 0.0, 0.0
        )
        assert match is True
        assert expected == 0.0

    def test_zero_revenue_hybrid(self, sample_contracts):
        """零营业额：hybrid 合同应付保底额"""
        match, expected, clause_id = check_paid_amount(
            sample_contracts["hybrid"], 0.0, 8000.0
        )
        assert match is True
        assert expected == 8000.0

    def test_very_large_revenue(self, sample_contracts):
        """
        超大营业额：验证浮点计算正确性
        round(999999.99 * 0.12, 2) = round(119999.9988, 2) = 120000.0
        """
        match, expected, clause_id = check_paid_amount(
            sample_contracts["commission"], 999999.99, 120000.00
        )
        # expected = 999999.99 * 0.12 = 119999.9988 → round → 120000.0
        assert expected == 120000.0
        assert match is True

    def test_negative_paid_amount(self, sample_contracts):
        """负数实缴金额（退款场景）：应标记为异常"""
        match, expected, clause_id = check_paid_amount(
            sample_contracts["fixed"], 0, -1000.0
        )
        assert match is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
