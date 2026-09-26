# Multi-Agent 电商推荐与营销系统（Python 版）

基于 **Python + FastAPI + LangGraph** 构建的多 Agent 电商推荐项目。系统将用户画像、商品推荐、库存决策和营销文案生成拆分为独立 Agent，通过 Supervisor 编排、异步并行和降级机制完成推荐链路。

## 核心能力

- **Multi-Agent 编排**：Supervisor + 4 个专业 Agent。
- **异步并行执行**：画像与召回、重排与库存校验分阶段并行。
- **稳定性治理**：统一超时、指数退避重试和 fallback 降级。
- **推荐重排**：基于用户画像和候选商品执行 LLM 排序。
- **营销文案生成**：按用户分群切换 Prompt 模板并执行敏感词过滤。
- **A/B 实验**：一致性哈希分桶与 Thompson Sampling 动态分配。
- **运行指标**：提供 Agent 调用次数、成功率和平均延迟查询。

## 系统架构

```mermaid
flowchart LR
    C[Client] --> F[FastAPI]
    F --> S[Supervisor]

    subgraph P1[Phase 1]
        U[User Profile Agent]
        R[Product Rec Agent]
    end

    subgraph P2[Phase 2]
        K[Rerank]
        I[Inventory Agent]
    end

    S --> U
    S --> R
    U --> K
    R --> K
    R --> I
    K --> A[Aggregate]
    I --> A
    A --> M[Marketing Copy Agent]
    M --> O[Recommendation Response]
```

LangGraph 版本使用状态图实现相同流程，入口为 `/api/v1/recommend/graph`。

## Agent 职责

| Agent | 职责 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| UserProfileAgent | RFM、分群与偏好分析 | 用户行为数据 | 用户画像 |
| ProductRecAgent | 候选筛选与 LLM 重排 | 用户画像、商品列表 | 推荐商品 |
| InventoryAgent | 可用库存校验、预警和限购 | 商品列表 | 可用商品及限购规则 |
| MarketingCopyAgent | 分群文案生成与合规过滤 | 用户画像、商品列表 | 营销文案 |

## 技术栈

- Python 3.11+
- FastAPI
- LangGraph、LangChain
- asyncio
- Pydantic
- Tenacity
- MiniMax / OpenAI-Compatible LLM API
- Redis FeatureStore（可选组件）
- structlog
- Docker Compose

## 快速启动

### 本地运行

```bash
cd python
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
python main.py
```

服务地址：

- API：`http://127.0.0.1:8000`
- Swagger：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

### Docker Compose

```bash
docker compose up --build
```

Compose 仅启动 Python API 和 Redis。Redis 用于可选的 FeatureStore 组件，默认推荐链路不依赖 Redis。

## 环境变量

| 变量 | 说明 |
| --- | --- |
| `ECOM_LLM_API_KEY` | LLM API Key |
| `ECOM_LLM_BASE_URL` | OpenAI-Compatible API 地址 |
| `ECOM_LLM_MODEL` | 模型名称，默认 `MiniMax-M1` |
| `ECOM_REDIS_URL` | 可选 Redis 地址 |
| `ECOM_AB_TEST_ENABLED` | 是否启用 A/B 实验 |

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/recommend` | Supervisor 推荐流程 |
| `POST` | `/api/v1/recommend/graph` | LangGraph 推荐流程 |
| `GET` | `/api/v1/experiments` | 查询实验状态 |
| `GET` | `/api/v1/metrics` | 查询 Agent 与业务指标 |
| `POST` | `/api/v1/experiments/{experiment_id}/outcome` | 记录实验结果 |

推荐请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u001","scene":"homepage","num_items":5}'
```

## 核心实现指标

以下数据可由代码和配置直接验证：

| 项目 | 配置 |
| --- | ---: |
| 专业 Agent 数量 | 4 |
| 并行执行阶段 | 2 |
| 单个 Agent 最大重试次数 | 2 |
| 用户画像 Agent 超时 | 5s |
| 商品推荐 Agent 超时 | 8s |
| 营销文案 Agent 超时 | 10s |
| 库存 Agent 超时 | 5s |
| A/B 分桶数量 | 100 |
| 营销 Prompt 模板 | 5 |
| 广告法敏感词规则 | 10 |
| REST API 数量 | 6 |
| A/B 单元测试 | 5 |

## 项目结构

```text
.
├── docker-compose.yml
├── python
│   ├── agents/                  # 4 个业务 Agent 与基础容错逻辑
│   ├── config/                  # 应用配置
│   ├── models/                  # Pydantic 数据结构
│   ├── orchestrator/            # Supervisor 与 LangGraph 编排
│   ├── services/                # A/B、FeatureStore 与指标组件
│   ├── tests/                   # 单元测试
│   ├── .env.example
│   ├── Dockerfile
│   ├── main.py                  # FastAPI 入口
│   └── requirements.txt
└── README.md
```

## 当前限制

- 商品推荐当前基于 `MOCK_PRODUCTS` 示例数据，尚未接入真实商品库。
- Redis `FeatureStore` 已实现为可注入组件，但默认推荐链路未连接 Redis。
- 库存判断基于请求中的商品库存字段，尚未接入 WMS 或数据库。
- A/B 实验数据与指标当前保存在内存中，服务重启后会重置。
- 当前没有 Milvus 向量检索、MCP、WMS、RAG 或 ReAct 的真实运行链路。
- 仓库未提供 P99、QPS、CTR、CVR、GMV 等压测或线上实验数据，请勿将预期目标写成实测结果。

## License

MIT
