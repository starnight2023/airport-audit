# =============================================================================
# tests/test_agent.py — Phase 3: Agent 工作流单元测试
# =============================================================================
# 运行方式：
#   pytest tests/test_agent.py -v                           # 快速测试（Mock模式）
#   pytest tests/test_agent.py -v --runslow                 # 含检索模块的测试
#
# 【测试策略】
# - 所有测试使用 audit_mock 模式，不依赖 DeepSeek API Key
# - Agent 逻辑测试分为三层：工具函数、状态机节点、完整稽核流程
# - 标记为 slow 的测试需要加载 BGE-Small 模型
# =============================================================================

import os
import sys
import json
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)


# ===========================================================================
# 一、工具函数测试
# ===========================================================================

class TestTools:
    """测试工具函数"""

    def test_query_revenue_success(self):
        """验证营收查询成功"""
        from src.tools import query_revenue
        result = query_revenue("M001", "2025-01")
        assert result["status"] == "success"
        data = result["data"]
        assert data["merchant_id"] == "M001"
        assert data["month"] == "2025-01"
        assert "reported_revenue" in data
        assert "paid_amount" in data
        assert "submit_date" in data

    def test_query_revenue_nonexistent_merchant(self):
        """验证不存在的商户返回错误"""
        from src.tools import query_revenue
        result = query_revenue("M999", "2025-01")
        assert result["status"] == "error"
        assert "fallback" in result

    def test_query_revenue_nonexistent_month(self):
        """验证不存在的月份返回错误"""
        from src.tools import query_revenue
        result = query_revenue("M001", "2026-01")  # 2026年数据不存在
        assert result["status"] == "error"

    def test_query_historical_disputes_found(self):
        """验证有争议记录的商户返回正确"""
        from src.tools import query_historical_disputes
        result = query_historical_disputes("M001")
        assert result["status"] == "success"
        assert len(result["data"]) >= 1

    def test_query_historical_disputes_empty(self):
        """验证无争议记录的商户返回空列表"""
        from src.tools import query_historical_disputes
        result = query_historical_disputes("M002")
        assert result["status"] == "success"
        assert len(result["data"]) == 0

    def test_tool_registry_contains_all(self):
        """验证工具注册表包含三个工具"""
        from src.tools import TOOL_REGISTRY, TOOL_DESCRIPTIONS
        assert "query_revenue" in TOOL_REGISTRY
        assert "extract_clause" in TOOL_REGISTRY
        assert "query_historical_disputes" in TOOL_REGISTRY
        assert len(TOOL_REGISTRY) == 3
        assert len(TOOL_DESCRIPTIONS) == 3

    def test_tool_descriptions_have_required_keys(self):
        """验证工具描述包含必需的字段"""
        from src.tools import TOOL_DESCRIPTIONS
        for desc in TOOL_DESCRIPTIONS:
            assert "name" in desc
            assert "description" in desc
            assert "parameters" in desc

    def test_load_contract_found(self):
        """验证合同加载成功"""
        from src.tools import load_contract
        contract = load_contract("M001")
        assert contract is not None
        assert contract["merchant_id"] == "M001"
        assert "type" in contract
        assert "clauses" in contract

    def test_load_contract_not_found(self):
        """验证不存在的商户返回 None"""
        from src.tools import load_contract
        contract = load_contract("M999")
        assert contract is None

    @pytest.mark.slow
    def test_extract_clause_success(self):
        """验证条款检索成功（需要 Chroma 知识库）"""
        from src.tools import extract_clause
        result = extract_clause("M001", "租金计算方式")
        assert result["status"] == "success"
        assert len(result["data"]) >= 1
        assert result["data"][0]["clause_type"] == "rent_calculation"


# ===========================================================================
# 二、Agent 状态机测试（Mock 模式）
# ===========================================================================

class TestAgentGraph:
    """测试 Agent 状态机（audit_mock）"""

    def test_audit_mock_normal_merchant(self):
        """验证正常商户的稽核流程"""
        from src.agent_graph import audit_mock
        result = audit_mock("M005", "2025-03")
        assert result["mode"] == "mock"
        assert result["merchant_id"] == "M005"
        assert result["month"] == "2025-03"
        assert result["contract_type"] in ("fixed", "commission", "hybrid")
        assert "audit_result" in result

    def test_audit_mock_has_contract(self):
        """验证稽核结果包含合同信息"""
        from src.agent_graph import audit_mock
        result = audit_mock("M001", "2025-01")
        assert result["contract"] is not None
        assert result["contract"]["merchant_id"] == "M001"
        assert "type" in result["contract"]

    def test_audit_mock_has_revenue_data(self):
        """验证稽核结果包含营收数据"""
        from src.agent_graph import audit_mock
        result = audit_mock("M001", "2025-01")
        assert result["revenue_data"] is not None
        assert "reported_revenue" in result["revenue_data"]

    def test_audit_mock_has_issues_list(self):
        """验证稽核结果的 issues 字段为列表"""
        from src.agent_graph import audit_mock
        result = audit_mock("M001", "2025-01")
        assert isinstance(result["issues"], list)

    def test_audit_mock_nonexistent_merchant(self):
        """验证不存在的商户返回错误"""
        from src.agent_graph import audit_mock
        result = audit_mock("M999", "2025-01")
        assert "error" in result

    def test_audit_mock_multiple_calls_different_merchants(self):
        """验证多个商户的稽核互不影响"""
        from src.agent_graph import audit_mock
        r1 = audit_mock("M001", "2025-01")
        r2 = audit_mock("M005", "2025-03")
        r3 = audit_mock("M010", "2025-06")
        assert r1["merchant_id"] == "M001"
        assert r2["merchant_id"] == "M005"
        assert r3["merchant_id"] == "M010"
        # 每个稽核结果必须包含该商户的合同
        assert r1["contract"]["merchant_id"] == "M001"
        assert r2["contract"]["merchant_id"] == "M005"

    def test_audit_result_has_expected_fields(self):
        """验证稽核结果包含全部关键字段"""
        from src.agent_graph import audit_mock
        result = audit_mock("M001", "2025-01")
        required_fields = [
            "merchant_id", "month", "contract_type",
            "contract", "revenue_data", "audit_result",
            "issues", "summary",
        ]
        for field in required_fields:
            assert field in result, f"缺少字段: {field}"

    def test_audit_issues_have_required_fields(self):
        """验证异常项包含全部关键字段"""
        from src.agent_graph import audit_mock
        result = audit_mock("M001", "2025-03")  # 选择一个可能有异常的月份
        if result.get("issues"):
            issue = result["issues"][0]
            assert "issue_type" in issue
            assert "description" in issue
            assert "severity" in issue
            assert "clause_id" in issue


# ===========================================================================
# 三、状态机节点逻辑测试
# ===========================================================================

class TestAgentNodes:
    """测试各节点逻辑"""

    def test_planner_node_contract_loading(self):
        """验证 planner_node 能正确加载合同"""
        from src.agent_graph import planner_node
        from src.tools import load_contract

        # 构造模拟 state
        state = {
            "merchant_id": "M001",
            "month": "2025-01",
            "contract": None,
            "tool_calls_plan": None,
            "planner_reasoning": None,
            "revenue_data": None,
            "clause_data": None,
            "disputes": None,
            "tool_errors": [],
            "audit_result": None,
            "issues": [],
            "report": None,
            "trace": [],
        }

        # 因为 planner 需要调用 LLM（会失败），所以会走降级路径
        # 降级路径仍然会加载合同
        result = planner_node(state)
        assert result["contract"] is not None
        assert result["contract"]["merchant_id"] == "M001"

    def test_planner_creates_tool_plan(self):
        """验证 planner 降级时仍创建工具计划"""
        from src.agent_graph import planner_node
        state = {
            "merchant_id": "M001",
            "month": "2025-01",
            "contract": None,
            "tool_calls_plan": None,
            "planner_reasoning": None,
            "revenue_data": None,
            "clause_data": None,
            "disputes": None,
            "tool_errors": [],
            "audit_result": None,
            "issues": [],
            "report": None,
            "trace": [],
        }
        result = planner_node(state)
        assert "tool_calls_plan" in result
        # 降级时 hybrid 合同应有 2 个工具
        if result["contract"]["type"] == "hybrid":
            assert len(result["tool_calls_plan"]) >= 1

    def test_executor_node_calls_tools(self):
        """验证 executor_node 能执行工具"""
        from src.agent_graph import executor_node
        state = {
            "merchant_id": "M001",
            "month": "2025-01",
            "tool_calls_plan": [
                {"tool": "query_revenue", "args": {"merchant_id": "M001", "month": "2025-01"}},
            ],
            "revenue_data": None,
            "clause_data": None,
            "disputes": None,
            "tool_errors": [],
            "contract": None,
            "trace": [],
        }
        result = executor_node(state)
        assert result["revenue_data"] is not None
        assert result["revenue_data"]["merchant_id"] == "M001"
        assert result["revenue_data"]["month"] == "2025-01"

    def test_rule_check_node_with_data(self):
        """验证 rule_check_node 能正常执行规则引擎"""
        from src.agent_graph import rule_check_node
        from src.tools import load_contract
        contract = load_contract("M001")

        state = {
            "merchant_id": "M001",
            "month": "2025-01",
            "contract": contract,
            "revenue_data": {
                "merchant_id": "M001",
                "month": "2025-01",
                "reported_revenue": 50000.0,
                "paid_amount": 8000.0,
                "submit_date": "2025-02-03",
            },
            "clause_data": [],
            "disputes": None,
            "issues": [],
            "trace": [],
        }
        result = rule_check_node(state)
        assert "audit_result" in result
        assert result["audit_result"]["merchant_id"] == "M001"
        assert result["audit_result"]["month"] == "2025-01"

    def test_rule_check_without_revenue_data(self):
        """验证缺少营收数据时 rule_check_node 跳过"""
        from src.agent_graph import rule_check_node
        state = {
            "merchant_id": "M001",
            "month": "2025-01",
            "contract": None,
            "revenue_data": None,
            "clause_data": [],
            "disputes": None,
            "issues": [],
            "trace": [],
        }
        result = rule_check_node(state)
        assert result["audit_result"]["status"] == "error"


# ===========================================================================
# 四、完整稽核流程测试
# ===========================================================================

class TestAuditFlow:
    """测试完整的稽核流程"""

    @staticmethod
    def _find_merchant_by_type(ctype: str) -> str:
        """从 contracts.json 动态查找指定合同类型的商户ID（不硬编码）"""
        import json as _json
        _path = os.path.join(PROJECT_ROOT, "data", "contracts.json")
        with open(_path, "r", encoding="utf-8") as f:
            contracts = _json.load(f)
        for c in contracts:
            if c["type"] == ctype:
                return c["merchant_id"]
        return "M001"

    def test_audit_hybrid_contract(self):
        """hybrid 合同稽核流程"""
        from src.agent_graph import audit_mock
        mid = self._find_merchant_by_type("hybrid")
        result = audit_mock(mid, "2025-01")
        assert result["contract_type"] == "hybrid"
        assert result["audit_result"]["contract_type"] == "hybrid"

    def test_audit_fixed_contract(self):
        """fixed 合同稽核流程"""
        from src.agent_graph import audit_mock
        mid = self._find_merchant_by_type("fixed")
        result = audit_mock(mid, "2025-01")
        assert result["contract_type"] == "fixed"

    def test_audit_all_contract_types_covered(self):
        """验证三种合同类型都有覆盖"""
        from src.agent_graph import audit_mock
        types_found = set()
        for ctype in ["fixed", "commission", "hybrid"]:
            mid = self._find_merchant_by_type(ctype)
            r = audit_mock(mid, "2025-01")
            if isinstance(r, dict) and "contract_type" in r:
                types_found.add(r["contract_type"])
        # 三种合同类型都应覆盖
        assert types_found == {"fixed", "commission", "hybrid"}, f"覆盖不足: {types_found}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])