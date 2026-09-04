# =============================================================================
# tests/conftest.py — pytest 插件配置
# =============================================================================
# 定义 --runslow 标志，用于控制包含模型加载的慢速测试
# =============================================================================

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """公开仓库不含 data/ 数据，若缺失则自动生成，保证 clone 后测试可直接运行"""
    if not os.path.exists(os.path.join(PROJECT_ROOT, "data", "contracts.json")):
        sys.path.insert(0, PROJECT_ROOT)
        from src.generate_data import main as gen_main

        gen_main(seed=42, year=2025, anomaly_rate=0.10)

    # historical_disputes.json 由 generate_data 不生成，这里合成最小样本
    disputes_path = os.path.join(PROJECT_ROOT, "data", "historical_disputes.json")
    if not os.path.exists(disputes_path):
        os.makedirs(os.path.dirname(disputes_path), exist_ok=True)
        sample = [
            {
                "merchant_id": "M001",
                "merchant_name": "云松咖啡",
                "month": "2025-03",
                "dispute_type": "revenue_dispute",
                "description": "商户声称3月申报营业额因系统故障少记，要求按实际营业额重算",
                "resolution": "已核实，确认少报属实，已按实际营业额补缴",
                "resolved_at": "2025-04-15",
            }
        ]
        with open(disputes_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)


def pytest_addoption(parser):
    """添加 --runslow 命令行选项"""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="运行慢速测试（含模型加载的测试用例）",
    )


def pytest_configure(config):
    """注册 slow 标记"""
    config.addinivalue_line("markers", "slow: 标记需要模型加载的慢速测试")


def pytest_collection_modifyitems(config, items):
    """跳过标记为 slow 的测试（除非指定 --runslow）"""
    if config.getoption("--runslow"):
        return  # --runslow 已指定，不跳过
    skip_slow = pytest.mark.skip(reason="需要 --runslow 选项运行")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)