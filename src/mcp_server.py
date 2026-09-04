# =============================================================================
# src/mcp_server.py — Phase 4: MCP Server 封装营收查询工具
# =============================================================================
# 功能说明：
#   1. 将 query_revenue 工具函数封装为 MCP 标准协议的工具
#   2. 通过 stdio 通信（本地进程间通信）
#   3. 实现 list_tools + call_tool 标准 MCP 接口
#
# 运行方式：
#   python src/mcp_server.py                              # 独立启动 Server
#   python -c "from src.mcp_client import demo; demo()"   # Client 连接测试
#
# 【MCP 协议说明】
# MCP (Model Context Protocol) 是 Anthropic 推出的 AI 工具调用标准协议。
# 本 Server 将 query_revenue 封装为 MCP 工具，任何支持 MCP 的 Client
# 都能通过协议发现和调用此工具。
#
# 【Phase 5 拓展位】
# extract_clause 也可以封装为 MCP 工具，注册到同一 Server。
# 【Phase 6 拓展位】
# stdio 通信可替换为 http/SSE 通信，实现远程 MCP 服务。
# =============================================================================

import json
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# MCP SDK 导入
# ---------------------------------------------------------------------------
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# ===========================================================================
# 一、MCP Server 定义
# ===========================================================================

# 创建 MCP Server 实例
# name 是 Server 的唯一标识，会在 client 连接时发送
server = Server("airport-audit-revenue")


# ===========================================================================
# 二、工具描述（MCP Tool Schema 格式）
# ===========================================================================

REVENUE_TOOL = Tool(
    name="revenue_query",
    description="查询商户指定月份的营收数据，返回申报营业额、实缴金额、报表提交日期。",
    inputSchema={
        "type": "object",
        "properties": {
            "merchant_id": {
                "type": "string",
                "description": "商户ID，如 M001",
            },
            "month": {
                "type": "string",
                "description": "月份，如 2024-01",
            },
        },
        "required": ["merchant_id", "month"],
    },
)


# ===========================================================================
# 三、工具执行函数
# ===========================================================================

def _execute_revenue_query(merchant_id: str, month: str) -> dict:
    """
    执行营收查询（同步函数）

    从 tools 模块导入 query_revenue 并调用。
    MCP Server 的 call_tool 是异步接口，但实际业务逻辑是同步的，
    用 asyncio.to_thread 或 loop.run_in_executor 包装。
    """
    from src.tools import query_revenue
    return query_revenue(merchant_id, month)


# ===========================================================================
# 四、MCP 协议处理器
# ===========================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    MCP list_tools 处理器

    Client 连接时调用此方法发现可用工具列表。
    返回的 Tool 列表包含名称、描述和参数 Schema，
    Client 据此知道可以调用哪些工具及参数格式。
    """
    return [REVENUE_TOOL]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    MCP call_tool 处理器

    Client 调用工具时触发此方法。
    name 是工具名称（revenue_query），
    arguments 是参数字典（merchant_id、month）。

    工具不存在时返回错误信息（符合 MCP 协议规范）。
    工具执行失败时返回 error 状态（含 fallback 降级信息）。

    Returns:
        list[TextContent]: MCP 通信内容块列表。
        TextContent 的 text 字段包含 JSON 序列化的结果。
    """
    if name != "revenue_query":
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"未知工具: {name}，可用工具: revenue_query",
            }),
        )]

    # 提取参数
    merchant_id = arguments.get("merchant_id", "")
    month = arguments.get("month", "")

    if not merchant_id or not month:
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": "缺少必需参数: merchant_id 和 month",
            }),
        )]

    # 执行查询
    import asyncio
    result = await asyncio.to_thread(_execute_revenue_query, merchant_id, month)
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


# ===========================================================================
# 五、Server 启动入口
# ===========================================================================

async def main():
    """
    使用 stdio 通信启动 MCP Server

    stdio_server 将当前进程的 stdin/stdout 作为 MCP 通信通道。
    Client 通过启动此 Server 为子进程并读写其 stdin/stdout 来实现调用。
    """
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    print("🚀 MCP Revenue Query Server 启动 (stdio 模式)", file=sys.stderr)
    asyncio.run(main())