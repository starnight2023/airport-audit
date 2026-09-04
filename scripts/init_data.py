#!/usr/bin/env python3
# =============================================================================
# scripts/init_data.py — 容器首次启动时初始化数据
# =============================================================================
# 功能：
#   1.  检查 data/ 目录下是否有 contracts.json，没有则生成模拟数据
#   2.  检查 chroma_db/ 目录下是否有向量库，没有则构建知识库
#
# 在 docker-compose 的 api 和 streamlit 服务启动前运行。
# 幂等设计：重复运行不会重复生成已有数据。
# =============================================================================

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONTRACTS_PATH = os.path.join(DATA_DIR, "contracts.json")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")


def check_data_exists() -> bool:
    """检查合同数据是否已生成"""
    if not os.path.exists(CONTRACTS_PATH):
        print("⚠️  合同数据文件不存在")
        return False
    # 检查是否有 bills CSV 文件
    bills_dir = os.path.join(DATA_DIR, "bills")
    if not os.path.exists(bills_dir) or not any(f.endswith(".csv") for f in os.listdir(bills_dir)):
        print("⚠️  账单 CSV 文件不存在")
        return False
    print("✅ 合同和账单数据已存在")
    return True


def check_kb_exists() -> bool:
    """检查 Chroma 知识库是否已构建"""
    # Chroma 持久化目录的标志文件
    if not os.path.exists(CHROMA_DB_DIR):
        print("⚠️  Chroma 知识库目录不存在")
        return False
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection("contract_clauses")
        count = collection.count()
        if count == 0:
            print("⚠️  Chroma 知识库为空")
            return False
        print(f"✅ 知识库已存在（{count} 条条款）")
        return True
    except Exception:
        print("⚠️  Chroma 知识库未就绪")
        return False


def main():
    """初始化数据"""
    print("=" * 50)
    print("🚀 数据初始化检查")
    print("=" * 50)

    # Step 1: 生成模拟数据
    if not check_data_exists():
        print("\n📦 生成模拟数据...")
        from src.generate_data import main as gen_data
        gen_data()
        print("✅ 模拟数据生成完成")
    else:
        print("\n📦 模拟数据已存在，跳过")

    # Step 2: 构建知识库
    if not check_kb_exists():
        print("\n🔨 构建合同知识库...")
        from src.build_knowledge_base import build_knowledge_base
        build_knowledge_base(force=True)
        print("✅ 知识库构建完成")
    else:
        print("\n🔨 知识库已存在，跳过")

    print("\n✅ 数据初始化完成")


if __name__ == "__main__":
    main()