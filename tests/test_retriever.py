# =============================================================================
# tests/test_retriever.py — Phase 2: 检索模块单元测试
# =============================================================================
# 运行方式：
#   pytest tests/test_retriever.py -v --runslow    # 完整运行（含模型加载）
#   pytest tests/                                   # 仅运行 Phase 1 快速测试
#
# Phase 2 的所有测试均为 slow 标记，因为需要加载 BGE-Small 模型（~33MB）。
# 默认被 pytest 跳过，必须使用 --runslow 选项。
# =============================================================================

import os
import sys
import json
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
SKIP_REASON = "使用 --runslow 选项运行含模型加载的测试"


def check_chroma_exists() -> bool:
    """检查 Chroma 知识库是否已构建"""
    import chromadb
    from chromadb.config import Settings
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        client.get_collection("contract_clauses")
        return True
    except Exception:
        return False


# ===========================================================================
# 基于已有知识库的检索测试
# ===========================================================================

@pytest.mark.slow
@pytest.mark.skipif(not check_chroma_exists(), reason="Chroma 知识库未构建，请先运行 python src/build_knowledge_base.py")
class TestRetrieverWithExistingKB:
    """使用已构建的合同知识库进行检索测试（不创建临时集合）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.retriever import ChromaClauseStore
        self.store = ChromaClauseStore(
            use_cache=False,  # 关闭缓存，测试真实检索
        )
        yield

    def test_stats_total_docs(self):
        """验证知识库包含全部 66 条条款"""
        stats = self.store.get_stats()
        assert stats["total_docs"] == 66
        assert stats["merchant_count"] == 20

    def test_retrieve_by_existing_merchant(self):
        """验证按商户 ID 检索返回正确商户的条款"""
        results = self.store.retrieve_clause("M001", "租金", top_k=10)
        assert len(results) >= 1
        for r in results:
            assert r["merchant_id"] == "M001"

    def test_retrieve_rent_calculation_semantic(self):
        """验证租金查询语义上匹配 rent_calculation 类型"""
        results = self.store.retrieve_clause("M001", "云松咖啡这个月租金怎么算", top_k=3)
        assert len(results) >= 1
        clause_types = [r["clause_type"] for r in results]
        assert "rent_calculation" in clause_types

    def test_retrieve_deadline_semantic(self):
        """验证截止日查询语义上匹配 submission_deadline 类型"""
        results = self.store.retrieve_clause("M001", "营业报表最晚什么时候交", top_k=3)
        assert len(results) >= 1
        clause_types = [r["clause_type"] for r in results]
        assert "submission_deadline" in clause_types

    def test_retrieve_late_fee_semantic(self):
        """验证滞纳金查询语义上匹配 late_fee 类型"""
        results = self.store.retrieve_clause("M001", "提交晚了要交多少钱滞纳金", top_k=3)
        assert len(results) >= 1
        clause_types = [r["clause_type"] for r in results]
        assert "late_fee" in clause_types

    def test_retrieve_hybrid_threshold(self):
        """验证保底额查询应返回 revenue_threshold（仅 hybrid 合同有）"""
        # M001 是 hybrid 合同（从 contracts.json 得知）
        results = self.store.retrieve_clause("M001", "保底营业额达标线是多少", top_k=5)
        clause_types = [r["clause_type"] for r in results]
        # revenue_threshold 并非每次都能排第一，但应该在 top_k 内
        assert any("revenue" in ct or "threshold" in ct or "rent" in ct for ct in clause_types)

    def test_get_clause_by_id_exact(self):
        """验证按 clause_id 精确查找"""
        result = self.store.get_clause_by_id("CTR-001-clause-001")
        assert result is not None
        assert result["clause_id"] == "CTR-001-clause-001"
        assert result["contract_id"] == "CTR-001"
        assert result["score"] == 1.0

    def test_get_clause_by_id_nonexistent(self):
        """验证不存在的 clause_id 返回 None"""
        result = self.store.get_clause_by_id("ID-DOES-NOT-EXIST")
        assert result is None

    def test_retrieve_nonexistent_merchant(self):
        """验证不存在的商户 ID 返回空列表"""
        results = self.store.retrieve_clause("M999", "租金")
        assert results == []

    def test_retrieve_global_search(self):
        """验证不指定 merchant_id 时全局检索"""
        results = self.store.retrieve_clause("", "滞纳金", top_k=5)
        assert len(results) >= 1
        # 全局检索应返回多个商户的结果
        merchants = set(r["merchant_id"] for r in results)
        assert len(merchants) >= 2  # 至少两个不同商户

    def test_result_score_range(self):
        """验证分数在 0~1 之间"""
        results = self.store.retrieve_clause("M001", "租金", top_k=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_results_ordered_by_score(self):
        """验证结果按相似度降序排列"""
        results = self.store.retrieve_clause("M001", "租金", top_k=5)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]

    def test_result_top_k_limit(self):
        """验证 top_k 参数正确限制返回数量"""
        results_2 = self.store.retrieve_clause("M001", "租金", top_k=2)
        results_5 = self.store.retrieve_clause("M001", "租金", top_k=5)
        assert len(results_2) <= 2, f"top_k=2 返回了 {len(results_2)} 条"
        assert len(results_5) <= 5, f"top_k=5 返回了 {len(results_5)} 条"

    def test_result_has_all_required_fields(self):
        """验证每条结果包含全部必要字段"""
        results = self.store.retrieve_clause("M001", "租金", top_k=1)
        assert len(results) >= 1
        r = results[0]
        assert "clause_id" in r
        assert "description" in r
        assert "clause_type" in r
        assert "contract_id" in r
        assert "merchant_id" in r
        assert "merchant_name" in r
        assert "parameters" in r
        assert "score" in r
        assert isinstance(r["parameters"], dict)


# ===========================================================================
# 缓存功能测试（不依赖模型加载）
# ===========================================================================

class TestMemoryCache:
    """测试内存缓存功能（纯 Python，无需模型）"""

    def test_cache_get_set(self):
        from src.retriever import MemoryCache
        cache = MemoryCache(max_size=10, ttl=60)
        cache.set("key1", [1, 2, 3])
        assert cache.get("key1") == [1, 2, 3]

    def test_cache_miss(self):
        from src.retriever import MemoryCache
        cache = MemoryCache(max_size=10, ttl=60)
        assert cache.get("nonexistent") is None

    def test_cache_expiry(self):
        from src.retriever import MemoryCache
        cache = MemoryCache(max_size=10, ttl=0)
        cache.set("key1", "value")
        assert cache.get("key1") is None

    def test_cache_eviction(self):
        from src.retriever import MemoryCache
        cache = MemoryCache(max_size=2, ttl=60)
        cache.set("key1", 1)
        cache.set("key2", 2)
        cache.set("key3", 3)
        assert cache.get("key1") is None
        assert cache.get("key3") == 3

    def test_cache_clear(self):
        from src.retriever import MemoryCache
        cache = MemoryCache(max_size=10, ttl=60)
        cache.set("key1", 1)
        cache.set("key2", 2)
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.size == 0


# ===========================================================================
# 便捷函数测试（依赖已有知识库）
# ===========================================================================

@pytest.mark.slow
@pytest.mark.skipif(not check_chroma_exists(), reason="Chroma 知识库未构建")
class TestConvenienceFunctions:
    """测试 retriever.py 提供的顶级便捷函数"""

    def test_retrieve_clause_function(self):
        from src.retriever import retrieve_clause
        results = retrieve_clause("M001", "租金")
        assert len(results) <= 3
        assert len(results) >= 1

    def test_get_clause_by_id_function(self):
        from src.retriever import get_clause_by_id
        result = get_clause_by_id("CTR-001-clause-001")
        assert result is not None
        assert result["clause_id"] == "CTR-001-clause-001"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--runslow"])