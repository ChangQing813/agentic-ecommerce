"""
Multi-Agent E-Commerce Recommendation System — FastAPI Entry Point

Endpoints:
  POST /api/v1/recommend          - LangGraph recommendation pipeline
  POST /api/v1/behaviors          - record user behavior for FeatureStore
  GET  /api/v1/experiments        - 查看A/B实验状态
  GET  /api/v1/metrics            - 查看系统监控指标
  GET  /health                    - 健康检查
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from config import get_settings
from models.schemas import (
    BehaviorEventRequest,
    BehaviorEventResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from orchestrator.graph import (
    ab_engine,
    build_recommendation_graph,
    configure_feature_store,
)
from services.feature_store import FeatureStore
from services.metrics import MetricsCollector

logger = structlog.get_logger()
settings = get_settings()


metrics_collector = MetricsCollector()
rec_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rec_graph

    redis_client = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )

    try:
        await redis_client.ping()
        feature_store = FeatureStore(
            redis_client=redis_client,
            ttl=settings.feature_ttl_seconds,
        )
        configure_feature_store(feature_store)
        app.state.redis_client = redis_client
        app.state.feature_store = feature_store
        logger.info("feature_store.enabled")
    except Exception as exc:
        logger.warning(
            "feature_store.unavailable",
            error=str(exc),
        )
        configure_feature_store(None)
        await redis_client.aclose()
        redis_client = None

    rec_graph = build_recommendation_graph()
    logger.info("app.startup", model=settings.llm_model)
    yield

    configure_feature_store(None)
    if redis_client is not None:
        await redis_client.aclose()

    logger.info("app.shutdown")


app = FastAPI(
    title="Multi-Agent E-Commerce Recommendation System",
    description="用户画像Agent + 商品推荐Agent + 营销文案Agent + 库存决策Agent，并行+聚合模式",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "healthy", "model": settings.llm_model}


@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Run the LangGraph recommendation pipeline."""
    result = await _run_graph(request)
    response = _build_response(request, result)
    _collect_metrics(response.agent_results)
    return response


@app.post("/api/v1/behaviors", response_model=BehaviorEventResponse)
async def record_behavior(
    behavior: BehaviorEventRequest,
    request: Request,
):
    """Record a user behavior event into the Redis FeatureStore."""
    feature_store = getattr(request.app.state, "feature_store", None)
    if feature_store is None:
        raise HTTPException(
            status_code=503,
            detail="FeatureStore is unavailable",
        )

    await feature_store.record_behavior(
        user_id=behavior.user_id,
        behavior_type=behavior.behavior_type,
        item_id=behavior.item_id,
        metadata=behavior.metadata,
    )
    return BehaviorEventResponse(
        user_id=behavior.user_id,
        behavior_type=behavior.behavior_type,
        item_id=behavior.item_id,
    )


@app.get("/api/v1/experiments")
async def get_experiments():
    """查看所有A/B实验状态"""
    experiments = {}
    for exp_id, exp in ab_engine.experiments.items():
        experiments[exp_id] = {
            "name": exp.name,
            "enabled": exp.enabled,
            "groups": [
                {
                    "name": g.name,
                    "weight": g.weight,
                    "config": g.config,
                    "successes": g.successes,
                    "failures": g.failures,
                }
                for g in exp.groups
            ],
            "stats": ab_engine.get_stats(exp_id),
        }
    return experiments


@app.get("/api/v1/metrics")
async def get_metrics():
    """查看系统监控指标"""
    return {
        "agents": metrics_collector.get_agent_stats(),
        "business": metrics_collector.get_business_stats(),
    }


@app.post("/api/v1/experiments/{experiment_id}/outcome")
async def record_outcome(experiment_id: str, group: str, success: bool):
    """记录A/B测试结果,更新Thompson Sampling"""
    ab_engine.record_outcome(experiment_id, group, success)
    return {"status": "recorded"}


async def _run_graph(request: RecommendationRequest):
    if not rec_graph:
        raise HTTPException(status_code=503, detail="Graph not initialized")
    state = {
        "user_id": request.user_id,
        "scene": request.scene,
        "num_items": request.num_items,
        "context": request.context,
    }
    return await rec_graph.ainvoke(state)


def _build_response(request: RecommendationRequest, result: dict[str, Any]):
    return RecommendationResponse(
        request_id=result.get("request_id", ""),
        user_id=result.get("user_id", request.user_id),
        products=result.get("final_products", []),
        marketing_copies=result.get("marketing_copies", []),
        experiment_group=result.get("experiment_group", "control"),
        agent_results=result.get("agent_results", {}),
        total_latency_ms=round(result.get("total_latency_ms", 0), 1),
    )


def _collect_metrics(agent_results: dict[str, Any]):
    for name, result in agent_results.items():
        metrics_collector.record_agent_call(
            agent_name=name,
            success=result.success,
            latency_ms=result.latency_ms,
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
