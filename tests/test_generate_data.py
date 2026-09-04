# =============================================================================
# tests/test_generate_data.py — 数据生成单元测试
# =============================================================================
# 运行方式：cd 到项目根目录，执行 pytest tests/ -v
# =============================================================================

import os
import sys
import json
import pytest

# 添加 src 到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from generate_data import (
    assign_contract_types,
    build_contracts_with_clauses,
    generate_monthly_bills,
    set_random_seed,
    CONTRACT_TYPE_DISTRIBUTION,
    MERCHANT_TEMPLATES,
)

SEED = 42


class TestMerchantAndContract:
    """测试商户与合同数据生成"""

    def test_merchant_count(self):
        """验证生成了50家商户"""
        merchants = assign_contract_types(SEED)
        assert len(merchants) == 50, f"期望50家商户，实际{len(merchants)}家"

    def test_contract_type_distribution(self):
        """验证合同类型分布符合预期（20 fixed / 15 commission / 15 hybrid）"""
        merchants = assign_contract_types(SEED)
        type_counts = {}
        for m in merchants:
            type_counts[m["type"]] = type_counts.get(m["type"], 0) + 1
        assert type_counts.get("fixed", 0) == 20, f"fixed 期望20, 实际{type_counts.get('fixed', 0)}"
        assert type_counts.get("commission", 0) == 15, f"commission 期望15, 实际{type_counts.get('commission', 0)}"
        assert type_counts.get("hybrid", 0) == 15, f"hybrid 期望15, 实际{type_counts.get('hybrid', 0)}"

    def test_contract_params_by_type(self):
        """验证不同合同类型的参数正确性"""
        merchants = assign_contract_types(SEED)
        for m in merchants:
            if m["type"] == "fixed":
                assert m["fixed_amount"] is not None and m["fixed_amount"] > 0
                assert m["commission_rate"] == 0.0
                assert m["min_guarantee"] is None
            elif m["type"] == "commission":
                assert m["fixed_amount"] is None
                assert 0.08 <= m["commission_rate"] <= 0.20
                assert m["min_guarantee"] is None
            elif m["type"] == "hybrid":
                assert m["fixed_amount"] is None
                assert 0.10 <= m["commission_rate"] <= 0.18
                assert m["min_guarantee"] is not None and m["min_guarantee"] > 0

    def test_contract_clauses(self):
        """验证合同条款构建"""
        merchants = assign_contract_types(SEED)
        contracts = build_contracts_with_clauses(merchants)
        assert len(contracts) == 50
        for c in contracts:
            assert "contract_id" in c
            assert "clauses" in c
            # 每份合同至少要有3条款（rent_calculation, submission_deadline, late_fee）
            assert len(c["clauses"]) >= 3
            # hybrid 类型多一条 revenue_threshold
            if c["type"] == "hybrid":
                assert len(c["clauses"]) == 4


class TestBillGeneration:
    """测试账单生成"""

    def test_bill_count_and_format(self):
        """验证账单数量和格式"""
        merchants = assign_contract_types(SEED)
        bills, labels = generate_monthly_bills(merchants, year=2025, seed=SEED)
        # 50 商户 × 12 月 = 600 条账单
        assert len(bills) == 600, f"期望600条账单，实际{len(bills)}条"
        # 逐条验证格式
        for bill in bills:
            assert "merchant_id" in bill and bill["merchant_id"].startswith("M")
            assert "month" in bill and "-" in bill["month"]
            assert bill["reported_revenue"] > 0
            assert bill["paid_amount"] > 0
            assert bill["submit_date"] and "-" in bill["submit_date"]

    def test_anomaly_rate(self):
        """
        验证异常比例约为10%（允许±5%浮动）
        注：因随机种子差异，实际异常率在5%~15%均属正常统计波动
        """
        merchants = assign_contract_types(SEED)
        # 使用偏离种子避免合同分配阶段的随机序列碰撞
        bills_seed = SEED + 1000
        bills, labels = generate_monthly_bills(
            merchants, year=2025, anomaly_rate=0.10, seed=bills_seed
        )
        rate = len(labels) / len(bills)
        # 10% ± 5% 浮动范围（样本量240，统计波动可接受）
        assert 0.05 <= rate <= 0.15, f"异常率{rate:.3f}不在预期范围0.05~0.15"

    def test_anomaly_types(self):
        """验证三种异常类型都有出现"""
        merchants = assign_contract_types(SEED)
        bills, labels = generate_monthly_bills(
            merchants, year=2025, anomaly_rate=0.10, seed=SEED
        )
        types_found = set(l["anomaly_type"] for l in labels)
        expected_types = {"少报营业额", "金额不符", "逾期提交"}
        assert types_found == expected_types, f"缺少异常类型: {expected_types - types_found}"

    def test_fixed_contract_bills(self):
        """验证固定租金合同的金额正确性"""
        merchants = assign_contract_types(SEED)
        fixed_merchants = [m for m in merchants if m["type"] == "fixed"]
        bills, labels = generate_monthly_bills(fixed_merchants, seed=SEED)
        # 非异常的fixed账单，paid_amount应等于fixed_amount
        for bill in bills:
            if not bill["_is_anomaly"]:
                expected = next(
                    m["fixed_amount"] for m in fixed_merchants
                    if m["merchant_id"] == bill["merchant_id"]
                )
                # 允许浮点精度差异
                assert abs(bill["paid_amount"] - expected) < 0.01, (
                    f"{bill['merchant_id']} {bill['month']}: "
                    f"期望{expected}, 实际{bill['paid_amount']}"
                )

    def test_seed_reproducibility(self):
        """验证随机种子保证可复现"""
        merchants1 = assign_contract_types(SEED)
        merchants2 = assign_contract_types(SEED)
        for m1, m2 in zip(merchants1, merchants2):
            assert m1["type"] == m2["type"]
            assert m1.get("fixed_amount") == m2.get("fixed_amount")
            assert m1.get("commission_rate") == m2.get("commission_rate")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
