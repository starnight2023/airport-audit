# =============================================================================
# Dockerfile — 机场非航收入智能稽核系统
# =============================================================================
# 用途：构建 api 和 streamlit 两个服务的共用基础镜像。
# 通过 docker-compose.yml 中不同的 command 控制启动方式。
#
# 构建镜像：
#   docker compose build
#
# 启动服务：
#   docker compose up -d
#
# 单服务构建（调试用）：
#   docker build -t airport-audit-api --target api .
#   docker build -t airport-audit-streamlit --target streamlit .
# =============================================================================

# ---- 基础镜像 ----
FROM python:3.11-slim AS base

# 设置环境变量（避免 Python 输出缓冲）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_ENDPOINT=https://hf-mirror.com

# 安装系统依赖（chromadb 和 sentence-transformers 需要）
# 使用国内镜像加速 apt 与 pip 下载
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null; \
    sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# ---- Python 依赖层（利用 Docker 缓存） ----
COPY requirements.txt .
# 使用清华 pip 镜像加速依赖安装
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# ---- 源代码层 ----
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# ---- API 服务镜像 ----
FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Streamlit 服务镜像 ----
FROM base AS streamlit
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]