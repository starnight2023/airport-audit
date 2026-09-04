# =============================================================================
# src/mcp_client.py — Phase 4: MCP Client 封装
# =============================================================================
# 功能说明：
#   1. 启动 MCP Server 子进程（通过 MCP SDK 的 stdio_client）
#   2. 自动发现 Server 上的可用工具
#   3. 提供 call_revenue_query() 函数供 Agent 调用
#   4. 支持上下文管理器自动管理生命周期
#
# 使用方式：
#   async with MCPClient() as client:
#       result = await client.call_revenue_query("M001", "2025-01")
#
# 【Phase 6 拓展位】
# stdio 通信可扩展为 SSE/WebSocket 实现远程 MCP 调用。
# =============================================================================

import asyncio
import json
import os
import sys
from typing import Optional

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
# MCP SDK
# ---------------------------------------------------------------------------
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import types as mcp_types


# ===========================================================================
# 一、MCP Client
# ===========================================================================

MCP_SERVER_SCRIPT = os.path.join(SRC_DIR, "mcp_server.py")


class MCPClient:
    """
    MCP Client — 通过 MCP SDK 连接本地 MCP Server

    功能：
    - 自动发现 Server 上注册的工具
    - 支持通过工具名称和参数调用
    - 提供 revenue_query 快捷方法
    - 上下文管理器自动管理生命周期

    使用示例：
        async with MCPClient() as client:
            result = await client.call_revenue_query("M001", "2025-01")
    """

    def __init__(self, server_script: str = MCP_SERVER_SCRIPT):
        self._server_script = server_script
        self._session = None
        self._read_stream = None
        self._write_stream = None
        self._stdio_cm = None
        self._connected = False
        self._available_tools: list[str] = []

    async def connect(self):
        """
        启动 MCP Server 子进程并建立会话

        流程：
        1. 通过 StdioServerParameters 配置子进程启动参数
        2. stdio_client 创建通信流
        3. 通过 session 发送 initialize 建立协议连接
        4. 发送 tools/list 发现可用工具
        """
        if self._connected:
            return

        # 配置 Server 进程参数
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self._server_script],
            encoding="utf-8",
        )

        # 建立 stdio 通信
        # stdio_client 是异步上下文管理器，手动管理其生命周期
        from mcp.client.session import ClientSession

        self._stdio_cm = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()
        self._read_stream = read_stream
        self._write_stream = write_stream

        # 手动管理 session 生命周期
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        await session.initialize()
        self._session = session

        # 发现工具
        tools_result = await self._session.list_tools()
        self._available_tools = [tool.name for tool in tools_result.tools]

        self._connected = True

    async def close(self):
        """关闭会话和通信流"""
        self._connected = False
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._stdio_cm:
            await self._stdio_cm.__aexit__(None, None, None)
            self._stdio_cm = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ---- 工具调用 ----

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """
        调用 MCP 工具

        Args:
            name: 工具名称（从 list_tools 获取）
            arguments: 工具参数字典

        Returns:
            dict: 工具执行结果
        """
        if not self._connected:
            await self.connect()

        if name not in self._available_tools:
            raise RuntimeError(
                f"MCP 工具 '{name}' 不可用，可用工具: {self._available_tools}"
            )

        result = await self._session.call_tool(name, arguments)

        # 解析 TextContent
        for content in result.content:
            if content.type == "text":
                return json.loads(content.text)

        return {"status": "error", "message": "未找到 TextContent 响应"}

    async def call_revenue_query(self, merchant_id: str, month: str) -> dict:
        """
        快捷方法：查询商户营收数据（通过 MCP 协议）

        Args:
            merchant_id: 商户ID
            month: 月份

        Returns:
            dict: 与 query_revenue 返回格式一致的营收数据
        """
        return await self.call_tool("revenue_query", {
            "merchant_id": merchant_id,
            "month": month,
        })

    async def list_tools(self):
        """列出 Server 上所有可用工具"""
        if not self._connected:
            await self.connect()
        return await self._session.list_tools()


# ===========================================================================
# 二、演示
# ===========================================================================

async def demo():
    """演示 MCP Client → Server 通信"""
    print("=" * 50)
    print("🔌 MCP Client → Server 通信演示")
    print("=" * 50)

    try:
        async with MCPClient() as client:
            # 列出工具
            tools = await client.list_tools()
            print(f"\n📋 发现工具 ({len(tools.tools)} 个):")
            for t in tools.tools:
                print(f"   - {t.name}: {t.description[:50]}...")

            # 查询营收
            print(f"\n📊 测试 call_revenue_query('M001', '2025-01'):")
            result = await client.call_revenue_query("M001", "2025-01")

            if result["status"] == "success":
                d = result["data"]
                print(f"   申报营业额: ¥{d['reported_revenue']:.2f}")
                print(f"   实缴金额:   ¥{d['paid_amount']:.2f}")
                print(f"   提交日期:   {d['submit_date']}")
            else:
                print(f"   ❌ {result.get('message', '')}")

            # 测试异常场景
            print(f"\n⚠️  测试不存在的商户:")
            result = await client.call_revenue_query("M999", "2025-01")
            print(f"   状态: {result['status']}, fallback: {result.get('fallback', 'N/A')}")

        print(f"\n✅ MCP 通信演示完成")

    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """命令行入口"""
    asyncio.run(demo())


if __name__ == "__main__":
    main()