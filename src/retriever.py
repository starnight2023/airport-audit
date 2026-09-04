# =============================================================================
# src/retriever.py — Phase 2: 合同条款语义检索模块
# =============================================================================
# 功能说明：
#   1. 封装 ChromaClauseStore 类，实现 ClauseStoreInterface
#   2. retrieve_clause(merchant_id, query_text) → 语义检索最匹配条款
#   3. get_clause_by_id(clause_id) → 精确查找
#   4. 带简易内存缓存（后续可替换为 Redis）
#
# 使用方式：
#   from src.retriever import ChromaClauseStore
#   store = ChromaClauseStore()
#   results = store.retrieve_clause("M001", "租金怎么算")
#
# 【Phase 3 拓展位】
# retrieve_clause 将被封装为 Agent 的 extract_clause 工具，
# Agent 在 rule_check_node 中通过此工具获取条款原文做异常解释。
# 【Phase 4 拓展位】
# MCP Server 将 retrieve_clause 封装为标准 MCP 工具，
# 通过 MCP 协议暴露给远程 Agent 调用。
# =============================================================================

# 网络环境适配：优先使用 HuggingFace 国内镜像（请根据实际网络环境调整）
import os as _os
_hf_endpoint = _os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
_os.environ["HF_ENDPOINT"] = _hf_endpoint

import json
import os
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
RULES_CONFIG_PATH = os.path.join(CONFIG_DIR, "rules.yaml")


# ===========================================================================
# 一、简易内存缓存（Redis 的轻量替代）
# ===========================================================================

class MemoryCache:
    """
    简易内存缓存（LRU 策略）

    【Phase 2 说明】
    在引入 Redis 之前，先用内存字典实现缓存逻辑。
    接口设计与 Redis 保持一致（get/set/clear），
    Phase 2 后期或生产环境可直接替换为 RedisCache 类。

    【设计要点】
    - max_size: 最多缓存条目数（防内存泄漏）
    - ttl: 缓存有效期（秒），过期条目自动失效
    - LRU 淘汰：超过 max_size 时淘汰最久未访问的条目
    """

    def __init__(self, max_size: int = 128, ttl: int = 300):
        self._max_size = max_size
        self._ttl = ttl
        self._cache: dict[str, tuple[float, object]] = {}  # key → (expire_time, value)

    def get(self, key: str):
        """获取缓存（None 表示未命中或已过期）"""
        if key not in self._cache:
            return None
        expire_time, value = self._cache[key]
        if time.time() >= expire_time:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value) -> None:
        """设置缓存（自动淘汰最旧条目）"""
        if len(self._cache) >= self._max_size:
            # 淘汰最早过期的条目
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[key] = (time.time() + self._ttl, value)

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def remove(self, key: str) -> None:
        """删除指定键"""
        self._cache.pop(key, None)

    @property
    def size(self) -> int:
        return len(self._cache)


# ===========================================================================
# 二、Chroma 向量检索实现
# ===========================================================================

class ChromaClauseStore:
    """
    基于 Chroma + BGE-Small 的合同条款检索器

    实现 ClauseStoreInterface 的全部方法，提供：
    - 语义检索：根据自然语言查询意图匹配最相关的条款
    - 元数据过滤：按 merchant_id 精确过滤
    - 精确查找：按 clause_id 直接读取
    - 缓存加速：短时间内的重复查询直接返回缓存结果

    使用方式:
        store = ChromaClauseStore()
        store.store_contract(contract_obj)      # 写入一条合同
        store.store_contracts_from_json(path)    # 从 JSON 文件批量写入
        results = store.retrieve_clause("M001", "租金计算方式")  # 检索
        clause = store.get_clause_by_id("CTR-001-clause-001")    # 精确查找
    """

    def __init__(
        self,
        collection_name: str = "contract_clauses",
        persist_dir: str = CHROMA_DB_DIR,
        use_cache: bool = True,
        cache_ttl: int = 300,
    ):
        """
        Args:
            collection_name: Chroma 集合名称
            persist_dir: Chroma 持久化目录
            use_cache: 是否启用内存缓存
            cache_ttl: 缓存有效期（秒）
        """
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._collection = None
        self._embed_fn = None

        # 缓存
        self._use_cache = use_cache
        self._cache = MemoryCache(max_size=128, ttl=cache_ttl) if use_cache else None

    # ---- 延迟初始化（避免 import 时加载模型） ----

    def _ensure_initialized(self):
        """确保 Chroma 集合和嵌入模型已初始化（延迟加载）"""
        if self._collection is not None:
            return

        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        # 1. 初始化嵌入模型（local_files_only 强制离线，避免网络检查卡顿）
        model_name = "BAAI/bge-small-zh-v1.5"
        try:
            model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            # 首次运行模型未缓存时回退到在线下载
            model = SentenceTransformer(model_name)
        self._embed_fn = model

        # 2. 创建 Chroma 客户端
        client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # 3. 获取集合（需要在 build_knowledge_base.py 中预先构建）
        try:
            self._collection = client.get_collection(
                name=self._collection_name,
                embedding_function=self._create_chroma_embedding_fn(),
            )
        except ValueError:
            raise RuntimeError(
                f"Chroma 集合 '{self._collection_name}' 不存在。\n"
                f"请先运行: python src/build_knowledge_base.py"
            )

    def _create_chroma_embedding_fn(self):
        """为 Chroma 创建兼容的嵌入函数（使用已加载的模型）"""
        class _ChromaEmbedFn:
            def __init__(self, model):
                self._model = model
                self._name = "bge-small-zh"
            def __call__(self, input):
                return self._model.encode(input, show_progress_bar=False).tolist()
            def name(self):
                return self._name
        return _ChromaEmbedFn(self._embed_fn)

    # ---- 核心检索方法 ----

    def retrieve_clause(
        self,
        merchant_id: str,
        query_text: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        检索与查询语义最匹配的合同条款

        支持两种检索模式：
        1. 指定 merchant_id：先元数据过滤再语义检索（精确+语义结合）
        2. merchant_id="" 或 None：全局语义检索（跨商户）

        Args:
            merchant_id: 商户ID（用于元数据过滤），为空字符串或 None 时不限商户
            query_text: 查询意图文本，如 "租金怎么算"、"保底额是多少"
            top_k: 返回条款数量上限

        Returns:
            list[dict]: [
                {
                    "clause_id": str,
                    "description": str,
                    "clause_type": str,
                    "contract_id": str,
                    "merchant_id": str,
                    "merchant_name": str,
                    "parameters": dict,       # 原始参数字典
                    "score": float,            # 余弦相似度
                }
            ]
            空列表表示无匹配结果

        【Phase 3 拓展位】
        Agent 调用此方法时，query_text 来自 LLM 生成的查询意图，
        merchant_id 来自稽核上下文的 state['merchant_id']。

        【Phase 2 设计说明】
        跳过使用 Chroma 内置的 where 过滤 + query 组合，而是：
        1. 如果指定了 merchant_id，先 get(where={merchant_id}) 获取该商户所有条款
        2. 再在内存中用嵌入向量做语义排序
        这样避免了 Chroma 复杂查询的性能问题，且对少量数据（6~8条/商户）足够快。
        """
        self._ensure_initialized()

        # ---- 检查缓存 ----
        cache_key = f"{merchant_id}|{query_text}|{top_k}"
        if self._use_cache and self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # ---- Step 1: 候选条款获取 ----
        if merchant_id:
            # 方式 A：指定商户 → 获取该商户全部条款
            result = self._collection.get(
                where={"merchant_id": merchant_id},
                include=["documents", "metadatas"],
            )
        else:
            # 方式 B：全局搜索 → 获取全部条款
            result = self._collection.get(
                include=["documents", "metadatas"],
            )

        ids = result["ids"]
        documents = result["documents"]
        metadatas = result["metadatas"]

        if not ids:
            return []

        # ---- Step 2: 语义排序 ----
        # 将查询文本转为向量
        query_embedding = self._embed_fn.encode(
            query_text, show_progress_bar=False
        )

        # 将候选条款转为向量
        doc_embeddings = self._embed_fn.encode(
            documents, show_progress_bar=False
        )

        # 计算余弦相似度
        import numpy as np
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = doc_embeddings / (
            np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10
        )
        scores = np.dot(doc_norms, query_norm)

        # ---- Step 3: 排序取 top_k ----
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            metadata = metadatas[idx]
            # 解析 parameters_json
            params = {}
            if metadata and metadata.get("parameters_json"):
                try:
                    params = json.loads(metadata["parameters_json"])
                except (json.JSONDecodeError, TypeError):
                    params = {}

            results.append({
                "clause_id": ids[idx],
                "description": metadata.get("description", ""),
                "clause_type": metadata.get("clause_type", ""),
                "contract_id": metadata.get("contract_id", ""),
                "merchant_id": metadata.get("merchant_id", ""),
                "merchant_name": metadata.get("merchant_name", ""),
                "parameters": params,
                "score": round(float(scores[idx]), 4),
            })

        # ---- 写入缓存 ----
        if self._use_cache and self._cache:
            self._cache.set(cache_key, results)

        return results

    # ---- 精确查找 ----

    def get_clause_by_id(self, clause_id: str) -> Optional[dict]:
        """
        按 clause_id 精确获取一条条款

        不走语义检索，直接按 ID 读取 Chroma 记录。
        适用于已知条款编号的精确引用场景（如规则引擎报告了某条款异常）。

        Args:
            clause_id: 条款ID (如 "CTR-001-clause-001")

        Returns:
            dict 或 None：{
                "clause_id": str,
                "description": str,
                "clause_type": str,
                "contract_id": str,
                "merchant_id": str,
                "merchant_name": str,
                "parameters": dict,
                "score": 1.0,  # 精确匹配固定为 1.0
            }
        """
        self._ensure_initialized()

        result = self._collection.get(
            ids=[clause_id],
            include=["documents", "metadatas"],
        )

        if not result["ids"]:
            return None

        metadata = result["metadatas"][0]
        params = {}
        if metadata and metadata.get("parameters_json"):
            try:
                params = json.loads(metadata["parameters_json"])
            except (json.JSONDecodeError, TypeError):
                params = {}

        return {
            "clause_id": result["ids"][0],
            "description": metadata.get("description", ""),
            "clause_type": metadata.get("clause_type", ""),
            "contract_id": metadata.get("contract_id", ""),
            "merchant_id": metadata.get("merchant_id", ""),
            "merchant_name": metadata.get("merchant_name", ""),
            "parameters": params,
            "score": 1.0,
        }

    # ---- 存储方法 ----

    def store_contract(self, contract) -> None:
        """
        将一条合同的所有条款存储到知识库

        Args:
            contract: Contract 对象（from src.models）或合同字典

        【Phase 3 拓展位】
        Agent 在合同变更或新增商户时调用此方法实时更新知识库。
        """
        self._ensure_initialized()

        # 兼容 Contract 对象和字典
        if hasattr(contract, "to_dict"):
            contract_dict = contract.to_dict()
        else:
            contract_dict = contract

        clauses_data = []
        for clause in contract_dict.get("clauses", []):
            param_str = ", ".join(
                f"{k}={v}" for k, v in clause.get("parameters", {}).items()
            )
            doc_text = (
                f"合同 {contract_dict['contract_id']} | "
                f"商户 {contract_dict['merchant_id']} | "
                f"条款类型: {clause['clause_type']}\n"
                f"条款内容: {clause['description']}\n"
                f"参数: {param_str}"
            )
            clauses_data.append({
                "id": clause["clause_id"],
                "text": doc_text,
                "metadata": {
                    "clause_id": clause["clause_id"],
                    "contract_id": contract_dict["contract_id"],
                    "merchant_id": contract_dict["merchant_id"],
                    "merchant_name": contract_dict.get("merchant_name", ""),
                    "clause_type": clause["clause_type"],
                    "description": clause["description"],
                    "parameters_json": json.dumps(
                        clause.get("parameters", {}), ensure_ascii=False
                    ),
                },
            })

        if clauses_data:
            self._collection.add(
                ids=[c["id"] for c in clauses_data],
                documents=[c["text"] for c in clauses_data],
                metadatas=[c["metadata"] for c in clauses_data],
            )

    def store_contracts_from_json(
        self, contracts_path: str = None
    ) -> int:
        """
        从 Phase 1 的 contracts.json 批量写入所有合同

        Args:
            contracts_path: 合同 JSON 文件路径（默认 data/contracts.json）

        Returns:
            int: 写入的条款数量

        与 build_knowledge_base.py 的功能等价，
        但使用面向对象方式供编程调用。
        """
        if contracts_path is None:
            contracts_path = os.path.join(
                os.path.dirname(self._persist_dir), "data", "contracts.json"
            )
            if not os.path.exists(contracts_path):
                contracts_path = os.path.join(
                    PROJECT_ROOT, "data", "contracts.json"
                )

        if not os.path.exists(contracts_path):
            raise FileNotFoundError(f"合同文件不存在: {contracts_path}")

        with open(contracts_path, "r", encoding="utf-8") as f:
            contracts_data = json.load(f)

        total = 0
        for contract in contracts_data:
            self.store_contract(contract)
            total += len(contract.get("clauses", []))

        return total

    # ---- 删除 ----

    def delete_merchant_clauses(self, merchant_id: str) -> int:
        """
        删除指定商户的所有条款

        Args:
            merchant_id: 商户ID

        Returns:
            int: 删除的记录数
        """
        self._ensure_initialized()

        result = self._collection.get(where={"merchant_id": merchant_id})
        ids_to_delete = result["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)

        # 清除相关缓存
        if self._use_cache and self._cache:
            self._cache.clear()

        return len(ids_to_delete)

    # ---- 缓存控制 ----

    def clear_cache(self):
        """清空检索缓存"""
        if self._cache:
            self._cache.clear()

    # ---- 统计 ----

    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        self._ensure_initialized()

        count = self._collection.count()
        all_metas = self._collection.get()["metadatas"]

        merchants = set()
        clause_types = {}
        for m in all_metas:
            if m:
                merchants.add(m.get("merchant_id", ""))
                ct = m.get("clause_type", "unknown")
                clause_types[ct] = clause_types.get(ct, 0) + 1

        return {
            "total_docs": count,
            "merchant_count": len(merchants),
            "merchants": sorted(merchants),
            "clause_type_distribution": clause_types,
            "cache_size": self._cache.size if self._cache else 0,
        }


# ===========================================================================
# 三、便捷函数（无需实例化即可调用）
# ===========================================================================

# 全局单例（延迟初始化）
_global_store: Optional[ChromaClauseStore] = None


def _get_store() -> ChromaClauseStore:
    """获取全局 ChromaClauseStore 单例"""
    global _global_store
    if _global_store is None:
        _global_store = ChromaClauseStore()
    return _global_store


def retrieve_clause(
    merchant_id: str,
    query_text: str,
    top_k: int = 3,
) -> list[dict]:
    """
    便捷函数：语义检索合同条款

    用法：
        from src.retriever import retrieve_clause
        results = retrieve_clause("M001", "租金计算方式")

    【Phase 3 拓展位】
    此函数将被直接封装为 Agent 的 extract_clause 工具。
    """
    return _get_store().retrieve_clause(merchant_id, query_text, top_k)


def get_clause_by_id(clause_id: str) -> Optional[dict]:
    """
    便捷函数：按条款 ID 精确查找

    用法：
        from src.retriever import get_clause_by_id
        clause = get_clause_by_id("CTR-001-clause-001")
    """
    return _get_store().get_clause_by_id(clause_id)


# ===========================================================================
# RAG 增强：多路召回（BM25 关键词 + 向量语义）
# ===========================================================================

def _build_bm25(merchant_id: str = ""):
    """
    构建 BM25 索引（对指定商户的条款文本做词频统计）

    Args:
        merchant_id: 空字符串表示全部商户

    Returns:
        (bm25_obj, clause_ids, clause_texts)
    """
    from rank_bm25 import BM25Okapi

    store = _get_store()
    store._ensure_initialized()

    if merchant_id:
        result = store._collection.get(where={"merchant_id": merchant_id},
                                       include=["documents"])
    else:
        result = store._collection.get(include=["documents"])

    ids = result["ids"]
    documents = result["documents"]

    # 中文分词：简单按字符 n-gram + 关键词切分
    def tokenize(text):
        import re
        # 提取中文词（2字以上）+ 英文数字
        tokens = re.findall(r'[一-鿿]{1,}|[a-zA-Z0-9.]+', text)
        # 中文按2字符滑窗切成词组，提升匹配率
        cjk_tokens = [t for t in tokens if re.match(r'[一-鿿]+', t)]
        other = [t for t in tokens if not re.match(r'[一-鿿]+', t)]
        bigrams = []
        for tok in cjk_tokens:
            if len(tok) <= 1:
                bigrams.append(tok)
            else:
                bigrams.append(tok)
                # 加 2-gram 提高召回
                bigrams.extend([tok[i:i+2] for i in range(len(tok)-1)])
        return bigrams + other

    tokenized_docs = [tokenize(d) for d in documents]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, ids, tokenized_docs


def retrieve_multi_path(
    merchant_id: str,
    query_text: str,
    top_k: int = 3,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
) -> list[dict]:
    """
    多路召回：向量语义 + BM25 关键词 融合检索

    召回策略：
    1. 向量路：Chroma 语义检索取 top_k*2 条
    2. BM25 路：关键词匹配取 top_k*2 条
    3. 融合：两路结果按加权分数合并，去重取 top_k

    为什么有效？
    - 向量能处理"意思相近但用词不同"（语义匹配）
    - BM25 能精确命中"含某个关键词"的条款（词汇匹配）
    - 两者互补，覆盖"语义但没用词重叠"和"用词重叠但语义无关"两个盲区

    Args:
        merchant_id: 商户ID
        query_text: 查询文本
        top_k: 返回条数
        bm25_weight: BM25 分数权重
        vector_weight: 向量分数权重

    Returns:
        list[dict]: 与 retrieve_clause 相同格式的结果
    """
    # ---- 向量路 ----
    vector_results = retrieve_clause(merchant_id, query_text, top_k=top_k * 2)

    # ---- BM25 路 ----
    try:
        bm25, ids, tokenized_docs = _build_bm25(merchant_id)
        import re
        query_tokens = re.findall(r'[一-鿿]+|[a-zA-Z0-9.]+', query_text)
        bm25_scores = bm25.get_scores(query_tokens)

        # 归一化 BM25 分数到 0~1
        max_score = max(bm25_scores) if bm25_scores.size > 0 and max(bm25_scores) > 0 else 1.0
        bm25_scores_norm = bm25_scores / max_score

        # 组装 BM25 结果
        store = _get_store()
        store._ensure_initialized()
        result = store._collection.get(ids=ids, include=["metadatas"])
        bm25_results = []
        top_indices = bm25_scores_norm.argsort()[::-1][:top_k * 2]
        for idx in top_indices:
            if bm25_scores_norm[idx] <= 0:
                continue
            meta = result["metadatas"][idx]
            params = {}
            if meta and meta.get("parameters_json"):
                try:
                    params = json.loads(meta["parameters_json"])
                except (json.JSONDecodeError, TypeError):
                    params = {}
            bm25_results.append({
                "clause_id": ids[idx],
                "description": meta.get("description", ""),
                "clause_type": meta.get("clause_type", ""),
                "contract_id": meta.get("contract_id", ""),
                "merchant_id": meta.get("merchant_id", ""),
                "merchant_name": meta.get("merchant_name", ""),
                "parameters": params,
                "bm25_score": round(float(bm25_scores_norm[idx]), 4),
            })
    except Exception:
        bm25_results = []

    # ---- 融合 ----
    # 向量结果带上 vector_score
    for r in vector_results:
        r["vector_score"] = r.get("score", 0)
    # BM25 结果带上 bm25_score，vector_score 记为 0
    for r in bm25_results:
        r["vector_score"] = 0.0
        r["score"] = 0.0  # 最终分数重新算

    # 合并：按 clause_id 去重，保留权重高的
    merged = {}
    for r in vector_results:
        merged[r["clause_id"]] = r
    for r in bm25_results:
        if r["clause_id"] in merged:
            merged[r["clause_id"]]["bm25_score"] = r["bm25_score"]
        else:
            merged[r["clause_id"]] = r

    # 计算融合分数
    for cid, r in merged.items():
        vs = r.get("vector_score", 0)
        bs = r.get("bm25_score", 0)
        r["score"] = round(vector_weight * vs + bm25_weight * bs, 4)
        r.pop("vector_score", None)
        r.pop("bm25_score", None)

    # 排序取 top_k
    sorted_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:top_k]


# ===========================================================================
# RAG 增强：Reranker 精排
# ===========================================================================

def rerank(
    results: list[dict],
    query_text: str,
    top_k: Optional[int] = None,
) -> list[dict]:
    """
    轻量级 Reranker：对召回结果做二次精排

    基于三因子评分，不需要额外模型：
    1. 原相似度分数（来自向量/多路召回）
    2. 条款类型与查询意图的相关性（关键词命中加分）
    3. 条款文本长度（太短的条款信息量低，微降权）

    相比直接取 top_k，Reranker 能纠正"分数高但类型不相关"的情况。

    Args:
        results: 召回结果列表（含 score 字段）
        query_text: 原始查询
        top_k: 精排后返回条数（默认全部）

    Returns:
        list[dict]: 重新排序后的结果，每条带 rerank_score
    """
    if not results:
        return results

    # 查询中的关键词集合
    import re
    query_terms = set(re.findall(r'[一-鿿]+|[a-zA-Z0-9]+', query_text))

    # 条款类型 → 查询意图的相关关键词
    type_keywords = {
        "rent_calculation": ["租金", "计算", "收费", "保底", "提成", "月租", "怎么收", "租金计算"],
        "submission_deadline": ["提交", "截止", "报表", "日期", "什么时候交", "逾期"],
        "late_fee": ["滞纳金", "逾期", "罚", "违约金", "罚款"],
        "revenue_threshold": ["达标线", "阈值", "保底", "营业额"],
    }

    scored = []
    for r in results:
        score = r.get("score", 0)

        # 因子2：条款类型相关度
        clause_type = r.get("clause_type", "")
        kws = type_keywords.get(clause_type, [])
        # 命中关键词数 / 关键词总数（覆盖度）
        hits = sum(1 for kw in kws if any(kw in q for q in query_terms) or kw in query_text)
        type_relevance = min(1.0, hits / max(1, len(kws)))

        # 因子3：文本长度（适中为佳）
        desc_len = len(r.get("description", ""))
        len_factor = min(1.0, desc_len / 40)  # 40字以上为佳

        # 加权总分
        rerank_score = 0.6 * score + 0.3 * type_relevance + 0.1 * len_factor
        r["rerank_score"] = round(rerank_score, 4)
        scored.append(r)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    if top_k:
        return scored[:top_k]
    return scored


# ===========================================================================
# RAG 增强：完整增强检索管线（改写→多路召回→精排）
# ===========================================================================

def enhanced_retrieve(
    merchant_id: str,
    query_text: str,
    top_k: int = 3,
    rewrite_mode: str = "rule",
    use_hyde: bool = False,
) -> list[dict]:
    """
    增强检索完整管线：
      查询改写 → 多路召回 → Reranker 精排 → top_k

    Args:
        merchant_id: 商户ID
        query_text: 用户查询
        top_k: 返回条数
        rewrite_mode: Query 改写模式（"rule"/"llm"）
        use_hyde: 是否先用 LLM 生成假设文档再检索

    Returns:
        list[dict]: 精排后的结果，每条含 rerank_score
    """
    # 1. Query 改写
    from src.query_rewriter import rewrite_query
    rewritten = rewrite_query(query_text, mode=rewrite_mode)

    # 2. 多路召回
    recalled = retrieve_multi_path(merchant_id, rewritten, top_k=top_k * 3)

    # 3. HyDE（可选）：用假设文档额外召回路并合并
    if use_hyde:
        try:
            from src.hyde import hyde_retrieve
            hyde_results = hyde_retrieve(merchant_id, query_text, top_k=top_k * 2, use_llm=True)
            # 合并去重
            seen = {r["clause_id"] for r in recalled}
            for r in hyde_results:
                if r["clause_id"] not in seen:
                    r["score"] = r.get("score", 0) * 0.8  # HyDE 路降权
                    recalled.append(r)
                    seen.add(r["clause_id"])
        except Exception:
            pass

    # 4. Reranker 精排
    return rerank(recalled, query_text, top_k=top_k)


# ===========================================================================
# 四、演示与测试
# ===========================================================================

def demo():
    """演示检索功能"""
    print("=" * 60)
    print("🔍 合同条款检索演示")
    print("=" * 60)

    # 初始化（自动加载 Chroma 知识库）
    try:
        store = ChromaClauseStore()
    except RuntimeError as e:
        print(f"❌ {e}")
        print("请先运行: python src/build_knowledge_base.py")
        return

    # 统计信息
    stats = store.get_stats()
    print(f"\n📊 知识库统计:")
    print(f"   文档数: {stats['total_docs']}")
    print(f"   商户数: {stats['merchant_count']}")
    print(f"   条款类型: {stats['clause_type_distribution']}")

    # 演示检索
    test_cases = [
        ("M001", "云松咖啡的租金怎么计算", "按商户名+意图查租金"),
        ("M001", "提交营业报表的截止日期", "查提交截止日"),
        ("M003", "保底额是多少", "查保底条款"),
        ("", "逾期提交的滞纳金怎么收", "全局搜索滞纳金规则"),
        ("M999", "不存在的商户", "查无此商户"),
    ]

    for mid, query, desc in test_cases:
        print(f"\n📝 测试: {desc}")
        print(f"   查询: merchant_id='{mid}', query='{query}'")
        results = store.retrieve_clause(mid, query, top_k=2)

        if not results:
            print(f"   ⚠️  无匹配结果")
        else:
            for r in results:
                print(f"   [{r['score']:.4f}] {r['clause_id']} | {r['description'][:60]}...")


# ===========================================================================
# 五、命令行入口
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="机场非航收入智能稽核系统 — 合同条款语义检索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python src/retriever.py                            # 运行演示模式
  python src/retriever.py --query "租金怎么算"        # 全局检索
  python src/retriever.py --merchant M001 --query "保底额"  # 按商户检索
  python src/retriever.py --clause CTR-001-clause-001       # 精确查找
        """,
    )
    parser.add_argument("--merchant", type=str, default="", help="商户ID")
    parser.add_argument("--query", type=str, help="查询文本")
    parser.add_argument("--clause", type=str, help="按 clause_id 精确查找")
    parser.add_argument("--top-k", type=int, default=3, help="返回结果数量上限")
    parser.add_argument("--stats", action="store_true", help="查看知识库统计")
    parser.add_argument("--demo", action="store_true", help="运行演示模式")

    args = parser.parse_args()

    # 演示模式
    if args.demo or not (args.query or args.clause or args.stats):
        demo()
        return

    # 统计模式
    if args.stats:
        try:
            store = ChromaClauseStore()
            stats = store.get_stats()
            print(f"📊 知识库统计:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        except RuntimeError as e:
            print(f"❌ {e}")
        return

    # 精确查找模式
    if args.clause:
        try:
            result = get_clause_by_id(args.clause)
            if result:
                print(f"✅ 找到条款:")
                for k, v in result.items():
                    print(f"  {k}: {v}")
            else:
                print(f"❌ 未找到条款: {args.clause}")
        except RuntimeError as e:
            print(f"❌ {e}")
        return

    # 检索模式
    if args.query:
        try:
            results = retrieve_clause(args.merchant, args.query, top_k=args.top_k)
            if not results:
                print(f"❌ 未匹配到结果")
            else:
                print(f"\n🔍 检索结果 (query='{args.query}', merchant='{args.merchant or '全部'}'):")
                print("=" * 60)
                for i, r in enumerate(results, 1):
                    print(f"\n--- 结果 #{i} (相似度: {r['score']:.4f}) ---")
                    print(f"  条款ID:   {r['clause_id']}")
                    print(f"  合同ID:   {r['contract_id']}")
                    print(f"  商户:     {r['merchant_name']} ({r['merchant_id']})")
                    print(f"  类型:     {r['clause_type']}")
                    print(f"  描述:     {r['description']}")
                    print(f"  参数:     {r['parameters']}")
        except RuntimeError as e:
            print(f"❌ {e}")


if __name__ == "__main__":
    main()