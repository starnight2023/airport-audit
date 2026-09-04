# =============================================================================
# src/interfaces.py — 可拓展接口定义
# =============================================================================
# 作用：为后续5个阶段定义"插槽接口"，当前阶段仅提供基类/桩代码，
#       确保 Phase 1 代码在不引入新依赖的情况下定义好拓展契约。
#
# 使用方式：
#   - 后续阶段继承这些 ABC 实现具体类
#   - Phase 1 的代码通过类型标注引用这些接口，而非具体实现
#
# 设计原则：
#   - 接口与实现分离：Phase 1 的 rule_engine 实现 RuleEngineInterface
#   - 依赖倒置：高层模块（Agent）依赖接口，不依赖具体实现
#   - 开闭原则：新增阶段通过实现接口扩展，不修改已有代码
# =============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 当前导入提示：
#   Phase 1 不使用外部 LLM/向量库，所以不导入 chromadb/sentence-transformers 等。
#   以下 import 标注了各阶段的依赖引入时机。
# ---------------------------------------------------------------------------


# ===========================================================================
# Phase 2 接口：知识库与检索
# ===========================================================================
# 引入时机：Phase 2 开发时启用
# 依赖：chromadb, sentence-transformers, redis
# ===========================================================================

class ClauseStoreInterface(ABC):
    """
    合同条款存储与检索接口

    【设计目的】
    屏蔽底层向量库（Chroma）或传统数据库的实现细节，
    使 Agent 和规则引擎可以通过统一接口获取合同条款。

    【实现类】
    - ChromaClauseStore (Phase 2): 基于 Chroma + BGE-Small 的向量检索
    - RedisClauseCache (Phase 2): 基于 Redis 的缓存层（装饰器模式）
    - DictClauseStore (Phase 3 测试): 基于内存字典的 Mock 实现
    """

    @abstractmethod
    def store_contract(self, contract: Any) -> None:
        """
        存储合同的所有条款到知识库

        Args:
            contract: Contract 对象（from src.models）
        """
        ...

    @abstractmethod
    def retrieve_clause(
        self,
        merchant_id: str,
        query_text: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        语义检索最匹配的合同条款

        Args:
            merchant_id: 商户ID（用于元数据过滤）
            query_text: 查询意图文本（如"租金计算方式是什么"）
            top_k: 返回条款数量上限

        Returns:
            list of dict: [
                {
                    "clause_id": str,
                    "description": str,
                    "clause_type": str,
                    "parameters": dict,
                    "score": float,  # 相似度分数
                }
            ]
        """
        ...

    @abstractmethod
    def get_clause_by_id(self, clause_id: str) -> Optional[dict]:
        """
        按 clause_id 精确获取一条条款（精确匹配，不走向量检索）

        Args:
            clause_id: 条款ID (如 "CTR-001-clause-001")

        Returns:
            条款字典，不存在返回 None
        """
        ...

    @abstractmethod
    def delete_merchant_clauses(self, merchant_id: str) -> None:
        """
        删除指定商户的所有条款（用于合同数据更新）

        Args:
            merchant_id: 商户ID
        """
        ...


# ===========================================================================
# Phase 3 接口：工具注册与执行
# ===========================================================================
# 引入时机：Phase 3 开发时启用（引入 langgraph、langchain-core 后）
# ===========================================================================

class ToolRegistryInterface(ABC):
    """
    工具注册与执行接口

    【设计目的】
    为 LangGraph Agent 提供统一工具管理，支持：
    - 动态注册/注销工具
    - 获取工具描述供 LLM 决策
    - 带降级兜底的执行

    【实现类】
    - LangChainToolRegistry (Phase 3): 基于 LangChain Tool 规范的注册器
    - MCPServerRegistry (Phase 4): 基于 MCP 协议的远程工具注册器
    """

    @abstractmethod
    def register_tool(
        self,
        name: str,
        func: callable,
        description: str,
        parameters_schema: dict,
    ) -> None:
        """
        注册一个新工具

        Args:
            name: 工具名称（Agent 调度时使用）
            func: 工具函数
            description: 工具描述（供 LLM 理解用途）
            parameters_schema: JSON Schema 格式的参数描述
        """
        ...

    @abstractmethod
    def execute_tool(self, name: str, **kwargs) -> Any:
        """
        执行已注册的工具，含降级兜底

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果；若失败返回降级后结果
        """
        ...

    @abstractmethod
    def get_tool_descriptions(self) -> list[dict]:
        """
        获取所有已注册工具的描述列表（供 Agent Planner 使用）

        Returns:
            list of dict: [{"name": str, "description": str, "parameters": dict}]
        """
        ...

    @abstractmethod
    def unregister_tool(self, name: str) -> None:
        """注销一个工具"""
        ...


# ===========================================================================
# Phase 4 接口：稽核服务与 MCP
# ===========================================================================
# 引入时机：Phase 4 开发时启用（引入 fastapi、mcp 后）
# ===========================================================================

class AuditServiceInterface(ABC):
    """
    稽核服务接口

    【设计目的】
    将稽核业务逻辑从 API 层和 Agent 层中解耦，
    使同一稽核逻辑可被 FastAPI、MCP、Streamlit 等不同前端调用。

    【实现类】
    - RuleEngineAuditService (Phase 1 预览): 仅用规则引擎的简易实现
    - AgentAuditService (Phase 3): 基于 LangGraph Agent 的完整实现
    """

    @abstractmethod
    def audit(self, merchant_id: str, month: str) -> Any:
        """
        对指定商户的指定月份执行稽核

        Args:
            merchant_id: 商户ID
            month: 月份 (格式: "2024-01")

        Returns:
            AuditResult 对象的字典表示
        """
        ...

    @abstractmethod
    def audit_batch(
        self, requests: list[dict[str, str]]
    ) -> list[Any]:
        """
        批量稽核

        Args:
            requests: [{"merchant_id": str, "month": str}, ...]

        Returns:
            稽核结果列表
        """
        ...


class MCPToolInterface(ABC):
    """
    MCP 协议工具封装接口

    【设计目的】
    定义 MCP Server 和 MCP Client 的抽象契约，
    使同一工具可同时在本地函数调用和远程 MCP 调用间切换。

    【实现类】
    - LocalMCPServer (Phase 4): 基于 MCP SDK 封装的 Server
    - LocalMCPClient (Phase 4): 本地 MCP Client 调用封装
    """

    @abstractmethod
    def get_tool_schema(self) -> dict:
        """
        获取 MCP 格式的工具 Schema

        Returns:
            {
                "name": str,
                "description": str,
                "inputSchema": {
                    "type": "object",
                    "properties": {...},
                }
            }
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具并返回结果"""
        ...


# ===========================================================================
# Phase 5 接口：报告格式化
# ===========================================================================
# 引入时机：Phase 5 开发时启用（引入 streamlit 后）
# ===========================================================================

class ReportFormatterInterface(ABC):
    """
    稽核报告格式化接口

    【设计目的】
    使同一份 AuditResult 可以输出为多种格式：
    - JSON：供 API 调用者使用
    - HTML/Markdown：供前端展示
    - PDF：供存档打印

    【实现类】
    - StreamlitReportFormatter (Phase 5): 供 Streamlit 渲染的格式化器
    - MarkdownReportFormatter (Phase 5): 导出 Markdown 报告
    """

    @abstractmethod
    def format(self, result: Any) -> Any:
        """
        将 AuditResult 格式化为特定输出

        Args:
            result: AuditResult 对象

        Returns:
            格式化后的输出（类型视具体实现而定）
        """
        ...

    @abstractmethod
    def format_batch(self, results: list[Any]) -> list[Any]:
        """批量格式化"""
        ...


# ===========================================================================
# Phase 6 接口：容器化配置
# ===========================================================================
# 引入时机：Phase 6 开发时启用
# ===========================================================================

class ConfigProviderInterface(ABC):
    """
    配置提供接口

    【设计目的】
    支持从多种来源加载配置：
    - YAML 文件（Phase 1）
    - 环境变量（Phase 4 Docker）
    - 配置中心（生产环境）

    【实现类】
    - YamlConfigProvider (Phase 1): 基于 YAML 的配置加载
    - EnvConfigProvider (Phase 6): 基于环境变量的配置加载
    - CompositeConfigProvider (Phase 6): 分层合并（环境变量优先于 YAML）
    """

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        ...

    @abstractmethod
    def get_section(self, section: str) -> dict:
        """获取配置段落"""
        ...


# ===========================================================================
# Phase 1 预留：简单的桩代码（仅用于验证接口兼容性）
# ===========================================================================

class StubClauseStore(ClauseStoreInterface):
    """
    桩实现 —— 仅用于 Phase 1 测试接口兼容性
    Phase 2 将替换为真正的 Chroma 向量存储
    """

    def __init__(self):
        self._store: dict[str, list[dict]] = {}

    def store_contract(self, contract: Any) -> None:
        for clause in contract.clauses:
            mid = contract.merchant_id
            if mid not in self._store:
                self._store[mid] = []
            self._store[mid].append({
                "clause_id": clause.clause_id,
                "description": clause.description,
                "clause_type": clause.clause_type,
                "parameters": clause.parameters,
                "score": 1.0,
            })

    def retrieve_clause(
        self, merchant_id: str, query_text: str, top_k: int = 3
    ) -> list[dict]:
        results = self._store.get(merchant_id, [])
        return results[:top_k]

    def get_clause_by_id(self, clause_id: str) -> Optional[dict]:
        for merchant_clauses in self._store.values():
            for clause in merchant_clauses:
                if clause["clause_id"] == clause_id:
                    return clause
        return None

    def delete_merchant_clauses(self, merchant_id: str) -> None:
        self._store.pop(merchant_id, None)
