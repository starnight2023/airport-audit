# =============================================================================
# src/api.py — Phase 4: FastAPI 稽核服务接口
# =============================================================================
# 功能说明：
#   1. POST /audit —— 执行完整稽核（调用 agent_graph.audit）
#   2. POST /audit/mock —— 执行 Mock 稽核（无需 API Key）
#   3. POST /audit/batch —— 批量稽核（逐商户/逐月执行）
#   4. GET /health —— 健康检查
#
# 运行方式：
#   uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
#
# 测试方式：
#   curl -X POST http://localhost:8000/audit -H "Content-Type: application/json" \
#     -d '{"merchant_id": "M001", "month": "2024-01"}'
#
#   curl -X POST http://localhost:8000/audit/mock -H "Content-Type: application/json" \
#     -d '{"merchant_id": "M005", "month": "2024-03"}'
#
#   curl http://localhost:8000/health
#
# 【Phase 5 拓展位】
# Streamlit 前端通过 requests 调用此 API。
# 【Phase 6 拓展位】
# Docker 容器化后，uvicorn 通过 docker-compose 管理。
# 端口和主机地址通过 config/rules.yaml 或环境变量配置。
# =============================================================================

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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# FastAPI 导入
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field


# ===========================================================================
# 一、请求/响应模型
# ===========================================================================

class AuditRequest(BaseModel):
    """
    稽核请求体

    【Phase 5 拓展位】
    Streamlit 前端发送此 JSON 到 /audit 接口。
    """
    merchant_id: str = Field(..., description="商户ID，如 M001")
    month: str = Field(..., description="月份，如 2024-01")


class AuditBatchRequest(BaseModel):
    """
    批量稽核请求体

    支持同时稽核多个商户或多个月份。
    """
    requests: list[AuditRequest] = Field(
        ..., description="稽核请求列表",
        min_length=1,
        max_length=50,
    )


# ===========================================================================
# 二、FastAPI 应用初始化
# ===========================================================================

app = FastAPI(
    title="机场非航收入智能稽核系统 API",
    description="""
    基于确定性规则引擎 + Agentic RAG + LangGraph 工作流的
    机场非航收入自动化稽核系统 API。

    功能：
    - /audit: 全量 AI 稽核（需 DeepSeek API Key）
    - /audit/mock: 快速规则引擎稽核（无需 API Key）
    - /audit/batch: 批量稽核
    - /health: 服务健康检查
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 跨域支持（允许 Streamlit 前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段允许所有来源，生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# 三、请求日志中间件
# ===========================================================================

@app.middleware("http")
async def log_requests(request, call_next):
    """
    请求日志中间件

    记录每次稽核请求的方法、路径、耗时。
    Phase 5 的评测脚本可以通过此日志做性能分析。
    """
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start

    # 只记录 /audit 相关请求
    if "/audit" in request.url.path:
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed:.2f}s)"
        )

    return response


# ===========================================================================
# 四、健康检查
# ===========================================================================

@app.get("/health")
async def health_check():
    """
    健康检查端点

    Phase 6 的 Docker healthcheck 配置将定期调用此接口
    判断服务是否正常运行。

    Returns:
        {"status": "ok", "service": "airport-audit-api", "version": "1.0.0"}
    """
    return {
        "status": "ok",
        "service": "airport-audit-api",
        "version": "1.0.0",
    }


# ===========================================================================
# 五、稽核端点
# ===========================================================================

@app.post("/audit")
async def run_audit(request: AuditRequest):
    """
    执行完整稽核（需 DeepSeek API Key）

    调用 agent_graph.audit() 执行完整的四节点 LangGraph 状态机：
    Planner(LLM) → Executor(工具) → RuleCheck(规则引擎) → Report(LLM)

    返回包含稽核状态、异常明细、合同条款引用和 LLM 生成的报告。

    Args:
        request: {"merchant_id": "M001", "month": "2024-01"}

    Returns:
        dict: 稽核结果（含 issues、report、trace 等字段）
    """
    try:
        from src.agent_graph import audit as agent_audit
        result = agent_audit(request.merchant_id, request.month)

        # 标准化响应
        return _format_audit_response(result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"稽核执行异常: {str(e)}")


@app.post("/audit/mock")
async def run_audit_mock(request: AuditRequest):
    """
    执行 Mock 稽核（无需 API Key）

    跳过 LLM 调用，直接使用规则引擎和默认工具计划。
    适用于开发调试、演示环境。

    Args:
        request: {"merchant_id": "M001", "month": "2024-01"}

    Returns:
        dict: 稽核结果（不含 LLM 报告，含 mode=mock 标记）
    """
    try:
        from src.agent_graph import audit_mock
        result = audit_mock(request.merchant_id, request.month)

        return _format_mock_response(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock 稽核异常: {str(e)}")


@app.post("/audit/batch")
async def run_audit_batch(request: AuditBatchRequest):
    """
    批量稽核

    对列表中的每对 (merchant_id, month) 执行稽核。
    全部使用 mock 模式（无需 API Key）。

    批量长度限制 1~50 条。

    Args:
        request: {"requests": [{"merchant_id": "M001", "month": "2024-01"}, ...]}

    Returns:
        dict: {"total": int, "results": list[dict]}
    """
    try:
        from src.agent_graph import audit_mock

        results = []
        errors = []

        for req in request.requests:
            try:
                result = audit_mock(req.merchant_id, req.month)
                results.append(_format_mock_response(result))
            except Exception as e:
                errors.append({
                    "merchant_id": req.merchant_id,
                    "month": req.month,
                    "error": str(e),
                })

        return {
            "total": len(request.requests),
            "success_count": len(results),
            "error_count": len(errors),
            "results": results,
            "errors": errors if errors else None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量稽核异常: {str(e)}")


# ===========================================================================
# 六、响应格式化
# ===========================================================================

def _format_audit_response(result: dict) -> dict:
    """
    格式化 audit() 返回结果为 API 响应

    【Phase 5 拓展位】
    此格式与 Streamlit 前端的数据渲染结构对齐。
    """
    contract = result.get("contract", {})
    report = result.get("report", {})
    issues = result.get("issues", [])
    audit_result = result.get("audit_result", {})

    return {
        "status": "success",
        "merchant_id": result.get("merchant_id", ""),
        "merchant_name": contract.get("merchant_name", ""),
        "month": result.get("month", ""),
        "contract_type": contract.get("type", ""),
        "audit_status": audit_result.get("status", "unknown"),
        "issue_count": len(issues),
        "issues": issues,
        "report": {
            "summary": report.get("summary", ""),
            "status": report.get("status", "unknown"),
            "findings": report.get("findings", []),
            "suggestion": report.get("suggestion", ""),
            "contract_refs": report.get("contract_refs", []),
        },
        "trace": result.get("trace", []),
    }


def _format_mock_response(result: dict) -> dict:
    """
    格式化 audit_mock() 返回结果为 API 响应

    mock 模式不含 LLM 报告，使用规则引擎原始摘要。
    """
    contract = result.get("contract", {})
    issues = result.get("issues", [])
    audit_result = result.get("audit_result", {})

    return {
        "status": "success",
        "mode": "mock",
        "merchant_id": result.get("merchant_id", ""),
        "merchant_name": contract.get("merchant_name", ""),
        "month": result.get("month", ""),
        "contract_type": contract.get("type", ""),
        "audit_status": audit_result.get("status", "unknown"),
        "issue_count": len(issues),
        "issues": issues,
        "summary": audit_result.get("summary", ""),
    }


# ===========================================================================
# 七、MCP 客户端集成（可选）
# ===========================================================================

# 是否启动时连接 MCP Server（通过环境变量 MCP_ENABLED=true 开启）
# 默认关闭，因为 MCP 主要用于演示协议能力，非核心功能
_MCP_ENABLED = os.environ.get("MCP_ENABLED", "false").lower() == "true"


@app.on_event("startup")
async def startup():
    """服务启动时初始化"""
    # 连接 MCP Server（如启用）
    if _MCP_ENABLED:
        try:
            from src.mcp_client import MCPClient
            mcp_client = MCPClient()
            await mcp_client.connect()
            # 存入 app.state 供后续使用
            app.state.mcp_client = mcp_client
            print("🔌 MCP Client 已连接")
        except Exception as e:
            print(f"⚠️  MCP Client 连接失败: {e}（不影响稽核功能）")
    else:
        print("ℹ️  MCP 客户端未启用（设置 MCP_ENABLED=true 可开启）")

    print(f"🚀 机场非航收入智能稽核 API 已启动")
    print(f"   📖 API 文档: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown():
    """服务关闭时清理"""
    if _MCP_ENABLED and hasattr(app.state, "mcp_client"):
        try:
            await app.state.mcp_client.close()
            print("🔌 MCP Client 已断开")
        except Exception:
            pass


# ===========================================================================
# 八、直接运行入口
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )