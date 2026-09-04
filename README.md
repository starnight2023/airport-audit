# 机场非航收入稽核 Agent 系统


## 项目简介

针对机场非航收入稽核面临的合同条款多、计费规则复杂、人工比对易出错的痛点，本系统面向机场财务与稽核人员，实现商户月度账单的金额复算、异常识别与稽核报告的全流程自动化。

核心：**确定性规则引擎保证金额计算 100% 可追溯（零 LLM 幻觉），AI Agent 编排稽核流程，混合检索保障合同条款召回质量**。

## 在线演示

系统已部署在云服务器，[**点击在线稽核**](http://119.29.191.104/airport/)
- 标准稽核：纯 Python 规则引擎复算，不调用 LLM、无需 API Key
- 深度稽核：调用 LLM 规划稽核流程并生成分析报告，返回异常明细、合同条款引用与自然语言报告

## 核心能力

**稽核**

- 支持固定租金、营业额提成、保底+提成（取高值）三种计费模式
- 自动判定租金提交截止日与宽限期，识别逾期提交
- 金额复算完全由确定性规则引擎完成，LLM 不参与计算


**系统特性**

- 混合检索：BM25 关键词 + 向量语义双路召回 + Reranker 精排
- 历史争议参考：检索相似争议记录辅助判断
- REST API 与 MCP 标准协议双通道服务化
- Streamlit 可视化面板，支持单商户稽核与稽核历史查询
- Docker Compose 一键部署，数据与向量库持久化
## 系统架构

稽核工作流基于 LangGraph 构建 Agent 状态机，全流程仅 2 次 LLM 调用（Planner 决策 + Report 生成），中间环节全部由确定性代码执行，从架构上根除金额幻觉风险。

工作流分为四个阶段：

- **Planner（规划层）**：根据合同类型自动规划稽核路径，决定需执行的校验项；
- **Executor（执行层）**：依次调用营收查询、条款检索等工具。其中条款检索采用 Agentic RAG 管线，经 Query 改写 → 向量 + BM25 多路召回 → 三因子精排后，返回 Top-K 条款供后续引用；
- **RuleCheck（校验层）**：调用确定性规则引擎，完成金额复算与逾期校验。金额计算走 `clause_id` 精确取条款的确定性路径，完全不依赖检索排序，也完全不经过 LLM；
- **Report（报告层）**：基于规则引擎的复算结果与检索到的条款上下文，生成结构化稽核报告。


## 技术栈

| 分类 | 技术 | 版本（最低） | 用途 |
|------|------|------|------|
| 运行时 | Python | 3.9+（镜像 3.11） | 服务端语言 |
| 规则引擎 | pandas / PyYAML | ≥2.0 / ≥6.0 | 数据处理 / 规则配置 |
| 向量检索 | ChromaDB | ≥1.5.0 | 合同条款向量库 |
| 文本向量化 | sentence-transformers（BGE-Small） | ≥2.2.0 | 中文语义向量化 |
| 混合检索 | rank-bm25 | ≥0.2.2 | 关键词多路召回 |
| Agent 编排 | LangGraph | ≥1.0.0 | 稽核工作流状态机 |
| LLM | DeepSeek（OpenAI SDK） | ≥1.0.0 | 规划与报告生成 |
| 服务化 | FastAPI / uvicorn | ≥0.100 / ≥0.23 | REST API |
| MCP | MCP SDK | ≥1.0.0 | AI 工具调用标准协议 |
| 前端 | Streamlit | ≥1.28.0 | 可视化面板 |
| 部署 | Docker / Docker Compose | 24+ | 容器化编排 |
| 测试 | pytest | ≥7.0 | 单元测试 |

## 项目结构

```
audit/
├── src/                              # 核心源码
│   ├── generate_data.py              # 模拟数据生成
│   ├── rule_engine.py                # 确定性规则引擎
│   ├── build_knowledge_base.py       # 合同条款向量库构建
│   ├── retriever.py                  # 混合检索（BM25+向量双路召回 + 精排）
│   ├── hyde.py / query_rewriter.py   # 检索增强 / Query 改写
│   ├── tools.py                      # Agent 工具函数
│   ├── agent_graph.py                # LangGraph 稽核状态机
│   ├── api.py / mcp_server.py / mcp_client.py  # 服务化
│   ├── app.py                        # Streamlit 前端
│   └── models.py                     # 数据模型
├── scripts/                          # 评测与运维脚本
│   ├── evaluate.py                   # 自动化评测（vs RAG 基线）
│   ├── benchmark_rag.py              # RAG 检索方案对比
│   ├── quick_eval.py                 # 规则引擎快速评测
│   └── init_data.py                  # 数据初始化（容器首启）
├── evaluation/                       # 生成式评测框架（指标 / 抽取器 / 评测报告）
├── tests/                            # pytest 单元测试
├── config/rules.yaml                 # 计费与稽核规则配置
├── data/                             # 生成数据（gitignore，init_data 重建）
├── chroma_db/                        # 向量库持久化（gitignore）
├── Dockerfile                        # api / streamlit 双目标镜像
├── docker-compose.yml                # 容器编排
├── requirements.txt                  # Python 依赖
├── .env.example                      # 环境变量模板
└── LICENSE                           # MIT 许可证
```

## 快速开始

### 环境要求

- Python 3.9+（Docker 镜像基于 3.11-slim）
- pip
- Docker 24+ 与 Docker Compose（仅容器部署需要）

### 安装

```bash
git clone https://github.com/starnight2023/airport-audit-public.git
cd airport-audit-public
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 如需使用深度稽核，在 .env 中配置 DEEPSEEK_API_KEY
```


### 生成数据

`data/` 已被 gitignore，首次使用需生成模拟数据：

```bash
python src/generate_data.py
```

生成 50 家商户的合同（固定租金 / 营业额提成 / 保底+提成各 20 / 15 / 15 份）、每商户 12 个月账单 CSV，以及异常标注（默认 10% 异常注入）。运行 `python scripts/init_data.py` 可额外构建 Chroma 向量库。

### 运行规则引擎

```bash
python src/rule_engine.py                        # 全量稽核
python src/rule_engine.py --merchant M001        # 指定商户
python src/rule_engine.py --month 2025-01        # 指定月份
python src/rule_engine.py --export report.json   # 导出稽核报告
```

### 启动 API 服务

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

交互式文档：http://localhost:8000/docs

### 启动 Web 面板

```bash
streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501
```

访问 http://localhost:8501，支持单商户稽核、稽核历史与结果展示。面板提供两种稽核模式：

- 标准稽核：纯 Python 规则引擎，不调用 LLM，无需 API Key
- 深度稽核：调用 DeepSeek LLM 生成分析报告，需在 `.env` 配置 DEEPSEEK_API_KEY

### 运行测试

```bash
pytest tests/ -v             # 全部单元测试
pytest tests/ --runslow      # 含模型加载的慢速测试
```

> 首次运行时若 `data/` 缺失，测试会自动生成模拟数据，clone 后可直接跑通。

## API 接口

| 方法 | 路径 | 用途 | 来源 |
|------|------|------|------|
| POST | /audit | 深度稽核：完整 AI 稽核（需 DeepSeek API Key） | [src/api.py:166](src/api.py#L166) |
| POST | /audit/mock | 标准稽核：纯 Python 规则引擎（无需 API Key） | [src/api.py:195](src/api.py#L195) |
| POST | /audit/batch | 批量标准稽核：一次提交 1–50 对（商户, 月份） | [src/api.py:219](src/api.py#L219) |
| GET | /health | 服务健康检查 | [src/api.py:144](src/api.py#L144) |

请求体示例（POST /audit）：

```json
{"merchant_id": "M001", "month": "2025-01"}
```

## MCP 服务

`src/mcp_server.py` 将营收查询封装为 MCP 标准协议工具，任何支持 MCP 的客户端可通过协议发现与调用：

- 工具 `revenue_query(merchant_id, month)`：返回申报营业额、实缴金额、报表提交日期

```bash
python src/mcp_server.py
```

API 服务设置 `MCP_ENABLED=true` 后，启动时会自动连接 MCP Client。

## 自动化评测

### 评测脚本

| 脚本 | 用途 |
|------|------|
| scripts/quick_eval.py | 规则引擎快速评测（精确率、召回率、F1、金额复算准确率） |
| scripts/evaluate.py | 完整评测：全量账单指标 + 与纯 RAG 基线对比，输出 Markdown 报告 |
| scripts/benchmark_rag.py | 检索方案对比（Recall@K、MRR、Top-1 准确率） |

```bash
python scripts/quick_eval.py [--regenerate] [--export eval_report.json]
python scripts/evaluate.py [-o eval_report.md] [--verbose] [--full]
python scripts/benchmark_rag.py
```

### 评测方法（口径说明）

评测分四套互补口径：

- **全量 600 条账单**（50 商户 × 12 月）：评估规则引擎整体稽核能力，含金额复算、异常识别、结果准确率
- **边界补充集 200 条**（25 家合成商户）：覆盖主集缺失的阈值附近维度（金额容差 1% 精确边界、提交宽限期、零营业额、缺失提交日期、多缴、达标线、双异常），专测边界判定
- **抽样 9 条**（5 异常 + 4 正常）：评估 Agent 全链路（含条款引用、无依据回答控制）
- **纯 RAG 基线对比**（同一抽样集）：对比规则引擎方案与纯 LLM 推理方案

### 系统评价指标（模拟数据全量账单评测，600 条）

| 指标 | 值 |
|------|-----|
| 应缴金额核对准确率 | 100% |
| 结果准确率 | 98.33% |
| 召回率 | 81.82% |
| 精确率 | 100% |
| F1 | 90.00% |
| 无依据回答控制率 | 100% |

### 边界补充集评测（200 条 / 25 家合成商户，规则引擎判定 vs 领域标签）

| 指标 | 值 |
|------|-----|
| 应缴金额核对准确率 | 100%（200/200）|
| 异常识别 F1 | 100% |

#### 生成式层（200 条真实 LLM 报告）

| 指标 | 值 |
|------|-----|
| 报告事实一致性（Report Consistency） | 100%（0 错 / 1000 事实）|
| Claim 准确率 / 召回率 | 99.6%（996/1000）|
| Claim 精确率 | 100% |
| 引用真实性（Citation Truthfulness） | 100%（424/424）|
| 引用完整性（Citation Completeness） | 100%（397/397）|
| 引用精确率（Citation Precision） | 93.63%（397/424）|
| 幻觉率（Hallucination Rate） | 0%（0/1954 断言）|

> 仅剩 2 类已知测量口径损失：抽取器锚定致 Claim 99.6%、滞纳金条款过度引用致 Citation Precision 93.63%。

### 与纯 RAG 基线对比（抽样评测，9 条用例：5 异常 + 4 正常）

| 指标 | 纯 RAG 基线 | 规则引擎方案 |
|------|------------|------------|
| 异常识别召回率 | 60.00% | 80.00% |
| 精确率 | 100.00% | 100.00% |
| F1 | 75.00% | 88.89% |
| 综合准确率 | 77.78% | 88.89% |

纯 RAG 基线由 LLM 直接复算金额并判断异常，9 条中漏检 2 条；规则引擎方案仅漏检 1 条固定合同少报营业额（金额角度无法发现），F1 领先 13.89%，且金额计算由确定性规则完成、零幻觉。

## Docker 部署

```bash
cp .env.example .env   # 填入 DEEPSEEK_API_KEY
docker compose up -d   # 构建并启动 API + Streamlit 两个服务
```

| 服务 | 访问地址 |
|------|----------|
| Streamlit 前端 | http://localhost:8502 |
| FastAPI 文档 | http://localhost:8001/docs |
| 健康检查 | http://localhost:8001/health |

首次启动执行 `init_data.py` 自动生成数据并构建向量库（需下载 BGE-Small 模型，约 3–8 分钟）；数据、向量库与模型缓存持久化在 Docker 卷中。查看日志 `docker compose logs -f`，停止 `docker compose down`。

## 安全与隐私

- API Key 通过环境变量注入，`.env` 已被 gitignore，Docker 镜像不包含密钥。
- 系统处理商户营收、应缴金额等财务数据，演示、测试与截图请优先使用脱敏数据。

## License

[MIT](LICENSE) © 2026 starnight2023
