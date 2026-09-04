# =============================================================================
# src/build_knowledge_base.py — Phase 2: 合同知识库构建
# =============================================================================
# 功能说明：
#   1. 加载 Phase 1 生成的 contracts.json
#   2. 使用 BGE-Small 模型将每条合同条款转为向量嵌入
#   3. 存入 Chroma 向量数据库（本地持久化到 ./chroma_db）
#   4. 支持增量构建（仅新增/变更的合同）和全量重建
#
# 运行方式：
#   python src/build_knowledge_base.py                        # 增量构建（跳过已存在）
#   python src/build_knowledge_base.py --force                 # 全量重建
#   python src/build_knowledge_base.py --merchant M001         # 仅构建指定商户
#   python src/build_knowledge_base.py --verbose               # 打印每条条款详情
#
# 输出：
#   chroma_db/ — Chroma 持久化目录（包含向量索引和元数据）
#
# 【Phase 3 拓展位】
# 构建好的知识库将被 Agent 的 extract_clause 工具调用，
# 通过语义检索定位最匹配的合同条款，辅助异常解释。
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

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONTRACTS_PATH = os.path.join(DATA_DIR, "contracts.json")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
RULES_CONFIG_PATH = os.path.join(CONFIG_DIR, "rules.yaml")


# ===========================================================================
# 一、嵌入模型封装
# ===========================================================================

class BgeEmbeddingFunction:
    """
    BGE-Small 中文嵌入模型封装（Chroma 兼容格式）

    将 sentence-transformers 的 BGE-Small 模型封装为 Chroma 可调用的
    EmbeddingFunction。BGE-Small 是专为中文优化的小型嵌入模型，
    输出 512 维向量，适合本地单机部署。

    首次运行会自动从 HuggingFace 下载模型（约 33MB），
    下载后缓存在 ~/.cache/huggingface/ 目录。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        Args:
            model_name: HuggingFace 模型名称（可从 rules.yaml 的 rag.embedding_model 读取）
        """
        from sentence_transformers import SentenceTransformer
        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            # 首次运行模型未缓存时回退到在线下载
            self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_embedding_dimension()
        self._model_name = model_name

    def __call__(self, input):
        """
        将文本列表转为向量列表

        Args:
            input: list[str] 文本列表

        Returns:
            list[list[float]] 向量列表
        """
        # BGE 模型推荐在编码时对 query 添加前缀，提升检索效果
        # 但 Chroma 内部会自动处理，此处保持与 retriever 中的编码一致
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()

    def name(self) -> str:
        """返回嵌入函数名称（Chroma 标识用）"""
        return f"bge-{self._model_name.split('/')[-1]}"

    @property
    def dimension(self) -> int:
        """返回嵌入向量的维度"""
        return self._dimension


# ===========================================================================
# 二、Chroma 集合配置
# ===========================================================================

def get_chroma_collection(
    collection_name: str = "contract_clauses",
    embedding_function: Optional[callable] = None,
    persist_dir: str = CHROMA_DB_DIR,
):
    """
    获取或创建 Chroma 集合

    Args:
        collection_name: 集合名称
        embedding_function: 嵌入函数（BgeEmbeddingFunction 实例）
        persist_dir: 持久化目录

    Returns:
        chromadb.Collection 对象

    【Phase 6 拓展位】
    persist_dir 应从 ConfigProvider 读取，支持环境变量覆盖。
    """
    import chromadb
    from chromadb.config import Settings

    # 创建 Chroma 客户端（持久化模式）
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    # 获取或创建集合
    # Chroma 的 get_or_create_collection 在集合存在时返回已有集合
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
    )

    return collection


# ===========================================================================
# 三、知识库构建核心逻辑
# ===========================================================================

def build_knowledge_base(
    contracts_path: str = CONTRACTS_PATH,
    collection_name: str = "contract_clauses",
    persist_dir: str = CHROMA_DB_DIR,
    force: bool = False,
    merchant_id: Optional[str] = None,
    verbose: bool = False,
    model_name: str = "BAAI/bge-small-zh-v1.5",
) -> dict:
    """
    构建合同条款向量知识库

    流程：
    1. 加载 contracts.json
    2. 过滤合同（可指定商户）
    3. 初始化 BGE-Small 嵌入模型
    4. 获取 Chroma 集合
    5. 逐条将条款写入向量库

    Args:
        contracts_path: 合同 JSON 文件路径
        collection_name: Chroma 集合名称
        persist_dir: Chroma 持久化目录
        force: True=清空重建，False=增量构建（跳过已存在的 clause_id）
        merchant_id: 仅构建指定商户
        verbose: 打印每条条款详情
        model_name: 嵌入模型名称

    Returns:
        dict: {"total_clauses": int, "added": int, "skipped": int, "collection": str}

    【Phase 2 补偿说明】
    本函数可以独立运行，也可以在 Agent 启动时自动调用（确保知识库已构建）。
    Phase 2 使用 Chroma 持久化，构建一次后后续启动直接加载。
    """
    # ---- Step 1: 加载合同数据 ----
    if not os.path.exists(contracts_path):
        raise FileNotFoundError(f"合同文件不存在: {contracts_path}\n请先运行 python src/generate_data.py")

    with open(contracts_path, "r", encoding="utf-8") as f:
        contracts_data = json.load(f)
    print(f"📄 加载合同: {len(contracts_data)} 份")

    # ---- Step 2: 过滤 ----
    if merchant_id:
        contracts_data = [c for c in contracts_data if c["merchant_id"] == merchant_id]
        print(f"🔍 过滤商户: {merchant_id} → {len(contracts_data)} 份合同")

    # ---- Step 3: 展平为条款列表 ----
    clauses = []
    for contract in contracts_data:
        for clause in contract.get("clauses", []):
            # 构建结构化文本（与 models.Clause.to_document_text 格式一致）
            param_str = ", ".join(
                f"{k}={v}" for k, v in clause.get("parameters", {}).items()
            )
            doc_text = (
                f"合同 {contract['contract_id']} | 商户 {contract['merchant_id']} | "
                f"条款类型: {clause['clause_type']}\n"
                f"条款内容: {clause['description']}\n"
                f"参数: {param_str}"
            )
            clauses.append({
                "clause_id": clause["clause_id"],
                "doc_text": doc_text,
                "merchant_id": contract["merchant_id"],
                "contract_id": contract["contract_id"],
                "merchant_name": contract["merchant_name"],
                "clause_type": clause["clause_type"],
                "description": clause["description"],
                "parameters_json": json.dumps(clause.get("parameters", {}), ensure_ascii=False),
            })
    print(f"📑 待写入条款: {len(clauses)} 条")

    if len(clauses) == 0:
        return {"total_clauses": 0, "added": 0, "skipped": 0, "collection": collection_name}

    # ---- Step 4: 初始化嵌入模型 ----
    print("🧠 初始化 BGE-Small 嵌入模型...")
    embed_fn = BgeEmbeddingFunction(model_name=model_name)
    print(f"   模型维度: {embed_fn.dimension}")

    # ---- Step 5: 获取集合 ----
    collection = get_chroma_collection(
        collection_name=collection_name,
        embedding_function=embed_fn,
        persist_dir=persist_dir,
    )
    existing_count = collection.count()
    print(f"📚 Chroma 集合 '{collection_name}': 已有 {existing_count} 条记录")

    # ---- Step 6: 写入数据 ----
    if force and existing_count > 0:
        print("🗑️  强制重建: 删除已有集合并重建...")
        # 先按ID逐批删除已有数据
        all_ids = collection.get()["ids"]
        for i in range(0, len(all_ids), 100):
            batch_ids = all_ids[i:i + 100]
            collection.delete(ids=batch_ids)
        # 重建集合（删除旧集合并创建新的空集合）
        import chromadb
        from chromadb.config import Settings
        _tmp_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        _tmp_client.delete_collection(collection_name)
        collection = get_chroma_collection(
            collection_name=collection_name,
            embedding_function=embed_fn,
            persist_dir=persist_dir,
        )
        existing_count = 0  # 重建后集合为空

    # 获取已有 ID 列表（增量构建时跳过）
    if not force and existing_count > 0:
        existing_ids = set(collection.get()["ids"])
    else:
        existing_ids = set()

    added = 0
    skipped = 0
    batch = []

    for clause in clauses:
        cid = clause["clause_id"]

        if cid in existing_ids:
            if verbose:
                print(f"  ⏭️  跳过 (已存在): {cid}")
            skipped += 1
            continue

        batch.append({
            "id": cid,
            "text": clause["doc_text"],
            "metadata": {
                "clause_id": cid,
                "contract_id": clause["contract_id"],
                "merchant_id": clause["merchant_id"],
                "merchant_name": clause["merchant_name"],
                "clause_type": clause["clause_type"],
                "description": clause["description"],
                "parameters_json": clause["parameters_json"],
            },
        })
        added += 1

        if verbose:
            print(f"  ✅ 添加: {cid} | {clause['clause_type']} | {clause['description'][:50]}...")

    # 批量写入 Chroma（每批 32 条，避免单次写入过大）
    batch_size = 32
    for i in range(0, len(batch), batch_size):
        chunk = batch[i:i + batch_size]
        collection.add(
            ids=[item["id"] for item in chunk],
            documents=[item["text"] for item in chunk],
            metadatas=[item["metadata"] for item in chunk],
        )

    print(f"\n✅ 知识库构建完成:")
    print(f"   新增: {added} 条 | 跳过: {skipped} 条 | 集合总计: {collection.count()} 条")

    return {
        "total_clauses": len(clauses),
        "added": added,
        "skipped": skipped,
        "collection": collection_name,
        "collection_size": collection.count(),
    }


# ===========================================================================
# 四、辅助功能
# ===========================================================================

def get_collection_stats(persist_dir: str = CHROMA_DB_DIR) -> dict:
    """
    查看 Chroma 知识库的统计信息

    Returns:
        dict: {"collection": str, "total_docs": int, "merchants": list}
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        collection = client.get_collection("contract_clauses")
    except ValueError:
        return {"error": "集合 'contract_clauses' 不存在，请先运行 build_knowledge_base.py"}

    count = collection.count()

    # 获取所有 metadatas 中的 merchant_id 做去重统计
    all_metadatas = collection.get()["metadatas"]
    if all_metadatas:
        merchants = sorted(set(m["merchant_id"] for m in all_metadatas if m))
    else:
        merchants = []

    clause_types = {}
    for m in all_metadatas:
        if m:
            ct = m.get("clause_type", "unknown")
            clause_types[ct] = clause_types.get(ct, 0) + 1

    return {
        "collection": "contract_clauses",
        "total_docs": count,
        "merchant_count": len(merchants),
        "merchants": merchants,
        "clause_type_distribution": clause_types,
    }


def delete_merchant_clauses(
    merchant_id: str,
    persist_dir: str = CHROMA_DB_DIR,
) -> int:
    """
    删除指定商户的全部条款记录（合同更新时使用）

    Args:
        merchant_id: 商户ID
        persist_dir: Chroma 持久化目录

    Returns:
        int: 删除的记录数

    【Phase 2 生产场景】
    当商户续约或合同变更时，先删除旧条款再重新写入。
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        collection = client.get_collection("contract_clauses")
    except ValueError:
        return 0

    result = collection.get(where={"merchant_id": merchant_id})
    ids_to_delete = result["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


# ===========================================================================
# 五、命令行入口
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="机场非航收入智能稽核系统 — 合同知识库构建",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python src/build_knowledge_base.py                    # 增量构建
  python src/build_knowledge_base.py --force             # 全量重建
  python src/build_knowledge_base.py --merchant M001     # 仅构建指定商户
  python src/build_knowledge_base.py --stats             # 查看知识库统计
  python src/build_knowledge_base.py --delete M001       # 删除商户记录
        """,
    )
    parser.add_argument("--force", action="store_true", help="强制重建（清空已有数据）")
    parser.add_argument("--merchant", type=str, help="仅构建指定商户ID")
    parser.add_argument("--verbose", action="store_true", help="打印每条条款的详情")
    parser.add_argument("--stats", action="store_true", help="查看知识库统计信息（不构建）")
    parser.add_argument("--delete", type=str, help="删除指定商户的所有条款")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-zh-v1.5", help="嵌入模型名称")
    parser.add_argument("--persist-dir", type=str, default=CHROMA_DB_DIR, help="Chroma 持久化目录")

    args = parser.parse_args()

    # ---- 统计/删除模式 ----
    if args.stats:
        stats = get_collection_stats(args.persist_dir)
        if "error" in stats:
            print(stats["error"])
        else:
            print("\n📊 Chroma 知识库统计")
            print("=" * 40)
            print(f"  集合名称:    {stats['collection']}")
            print(f"  文档总数:    {stats['total_docs']}")
            print(f"  商户数量:    {stats['merchant_count']}")
            print(f"  条款类型分布: {stats['clause_type_distribution']}")
            print(f"  商户列表:    {stats['merchants']}")
        return

    if args.delete:
        deleted = delete_merchant_clauses(args.delete, args.persist_dir)
        print(f"🗑️  已删除商户 {args.delete} 的 {deleted} 条条款")
        return

    # ---- 构建模式 ----
    start_time = time.time()
    print("🚀 开始构建合同知识库...")
    print("=" * 50)

    try:
        result = build_knowledge_base(
            contracts_path=CONTRACTS_PATH,
            collection_name="contract_clauses",
            persist_dir=args.persist_dir,
            force=args.force,
            merchant_id=args.merchant,
            verbose=args.verbose,
            model_name=args.model,
        )
        elapsed = time.time() - start_time
        print(f"\n⏱️  耗时: {elapsed:.2f}s")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()