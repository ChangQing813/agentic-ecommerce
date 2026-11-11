# 多 Agent 电商推荐系统

个人工程项目：使用 FastAPI、LangGraph、PostgreSQL、Redis 和 OpenAI-compatible 模型服务，完成行为驱动的商品推荐及后台文案生成。商品为示例数据，无真实交易系统；不宣称生产容量或业务提升。

## 启动

安装并启动 Docker Desktop（Linux 引擎），在项目根目录执行：

```cmd
copy .env.example .env
```

编辑根目录 `.env`，需要真实模型时填写密钥并开启开关。已有 `.env` 时不要覆盖。然后：

```cmd
docker compose up -d --build
```

- 演示页面：http://localhost:8000/
- 接口测试：http://localhost:8000/docs
- 存活检查：`GET /health`
- 数据库就绪检查：`GET /ready`
- Agent 指标：`GET /api/v1/metrics`

Compose 自动等待数据库就绪，执行 `manage.py --seed`，再启动 API。重复启动不覆盖已有商品。停止使用 `docker compose stop`；不要删除数据卷以免丢失记录。

## 配置

Compose 使用根目录 `.env`；本机运行 Python 时使用 `python/.env`。示例文件不保存真实密钥。

| 配置 | 用途 |
|---|---|
| ECOM_LLM_ENABLED | 是否启用真实模型，默认 false |
| ECOM_LLM_API_KEY | 供应商模型密钥，仅后端使用 |
| ECOM_LLM_BASE_URL / ECOM_LLM_MODEL | 供应商地址和模型 |
| ECOM_API_KEY | 可选服务调用密钥，客户端传 X-API-Key；不是模型密钥 |
| ECOM_MAX_ACTIVE_REQUESTS | 每进程推荐请求上限，默认 32 |
| PIP_INDEX_URL | 构建依赖下载源，Compose 默认清华镜像 |

模型开关关闭也能运行，精排使用规则，文案使用模板，响应明确标记降级。演示页服务密钥只保存在页面内存，不写入本地存储。

## 推荐与行为

```json
{"user_id":"demo-user","scene":"homepage","num_items":5}
```

向 `POST /api/v1/recommend` 提交上述请求。演示页可直接获取推荐、记录点击并展示 Agent 状态；不支持预算或自然语言需求。

`POST /api/v1/behaviors` 支持 view、click、cart、purchase。新事件使用唯一 event_id；重试复用完整原始事件（包括 occurred_at）。首次写入 recorded、重复 duplicate、不同内容复用 ID 返回 409，未知商品返回 404。购买行为是测试输入，不是可信订单转化。

## 唯一运行入口与代码结构

- `python/main.py`：API、静态演示页及资源生命周期。
- `python/services/multi_agent.py`：当前 LangGraph 编排及四个 Agent。
- `repository.py`：商品、行为、初始迁移和连接池。
- `copy_cache.py`：后台队列、文案校验、Redis 缓存。
- `governance.py`：熔断、容量错误和聚合指标。
- `recommendation.py`：独立规则基线，用于离线评测。
- `python/agents/`、`python/orchestrator/`：历史参考实现，不是 API 入口。
- `ab_test.py`、`metrics.py`：历史实验模块，未接入线上推荐。

## 验证与评测

本机 Python 3.11+，在根目录执行：

```cmd
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r python/requirements.txt
.venv\Scripts\python -m unittest discover -s python/tests -t python
.venv\Scripts\python python/tests/test_ab_test.py
.venv\Scripts\python python/scripts/evaluate.py
```

评测写入 `reports/offline-evaluation.json`：30 条合成规则约束用例，不调用模型、不使用生产数据、不代表 CTR/CVR 或推荐相关性提升。

HTTP 压测是显式操作，会产生真实模型费用。小样本仅验证脚本，不代表生产容量：

```cmd
.venv\Scripts\python python/scripts/benchmark.py --allow-model --requests 10 --concurrency 2
```

服务开启鉴权时先设置环境变量 ECOM_API_KEY。输出 `reports/http-benchmark.json`，包含成功率、P50/P95/P99、拒绝及请求明细。不要将拒绝吞吐视为成功吞吐。真实规则/LLM 质量比较需固定候选、盲评标准及独立测试集，尚未完成。

自动检查定义在 `.github/workflows/checks.yml`；推送到 GitHub 后执行测试、规则评测、Compose 校验和镜像构建。本地通过不代表远程 CI 已运行。

## 故障与运行边界

请求预算 12 秒；节点使用更小的自身预算。模型并发每进程 8，最多两次传输尝试，仅暂时性错误重试。连续 5 次逻辑调用失败触发 30 秒熔断，半开只允许一个探测。推荐容量超限返回 429。

库存阶段重读数据库，失败时清空候选；候选白名单和严格 JSON schema 校验阻止模型新增商品或篡改价格。库存是数据库快照，交易仍需订单系统复查。

新用户无类目偏好时跳过模型画像。文案读取缓存最多等待 50ms；未命中立即返回模板，后台队列容量 32、单 worker、单次尝试，后台等待最多 12 秒。成功缓存一小时，失败本地冷却 30 秒。任务不持久化、去重不跨实例。缓存只含公共商品信息。

PostgreSQL 连接/语句超时 3 秒、锁等待/连接池等待 1 秒，池容量 8、不额外扩容。取消协程不能强制停止同步线程，数据库超时约束的是单条 SQL。每次全量读商品和近期事件仍需优化。

尚无用户级授权、分布式限流、WMS、完整迁移升级、持久化监控告警或灰度系统。基础词表不能替代文案事实审核。confidence 是状态占位值，不是校准概率；experiment_group 的 control 是兼容字段，不代表真实在线实验。

详细见 [架构](docs/architecture.md)、[验收记录](docs/acceptance.md)、[面试说明](docs/interview.md)。

## 排错

- 运行旧镜像：`docker compose up -d --build --force-recreate api`。
- 日志：`docker compose logs --tail=100 api`。
- 下载 SSL 中断：检查 Docker Desktop 代理。CMD 可用 `set PIP_INDEX_URL=https://pypi.org/simple` 切换源；不要关闭 TLS 校验。
- 模板文案：检查 marketing_copy.data.decision，排队不是模型完成；cache_hit 表示缓存命中。
