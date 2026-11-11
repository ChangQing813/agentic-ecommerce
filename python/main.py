from contextlib import asynccontextmanager
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from redis.asyncio import Redis
import secrets
import structlog
from config import get_settings
from models.schemas import BehaviorEventRequest, BehaviorEventResponse, RecommendationRequest, RecommendationResponse
from services.repository import Repository
from services.governance import Overloaded
from services.multi_agent import MultiAgentRecommendationService
from services.feature_store import FeatureStore
from services.copy_cache import CopyCache

settings = get_settings()
logger = structlog.get_logger()

async def authorize(x_api_key: str = Header(default='')):
    if settings.api_key and not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(401, 'Invalid API key')

@asynccontextmanager
async def lifespan(app):
    repo = Repository(settings.database_url)
    await run_in_threadpool(repo.ready)
    app.state.repository = repo
    app.state.recommendations = MultiAgentRecommendationService(repo, settings)
    redis = Redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
    app.state.feature_store = FeatureStore(redis)
    copies = CopyCache(redis, app.state.recommendations)
    app.state.recommendations.copy_cache = copies
    copies.start()
    try:
        yield
    finally:
        await copies.close()
        await app.state.recommendations.gateway.close()
        await redis.aclose()
        repo.engine.dispose()

app = FastAPI(title='E-commerce recommendation service', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=['GET','POST'], allow_headers=['Content-Type','X-API-Key'])

@app.get('/health')
async def health():
    return {'status': 'alive'}

@app.get('/ready')
async def ready():
    try:
        await run_in_threadpool(app.state.repository.ready)
    except Exception:
        raise HTTPException(503, 'Database unavailable or schema not initialized')
    return {'status': 'ready'}

@app.post('/api/v1/recommend', response_model=RecommendationResponse, dependencies=[Depends(authorize)])
async def recommend(request: RecommendationRequest):
    if request.scene != 'homepage':
        raise HTTPException(422, 'Only homepage is supported in the baseline')
    try:
        return await app.state.recommendations.recommend(request)
    except Overloaded:
        raise HTTPException(429, "Request capacity exceeded", headers={"Retry-After":"1"})
    except TimeoutError:
        raise HTTPException(503, "Recommendation time budget exhausted")
    except Exception:
        logger.exception("recommendation.failed")
        raise HTTPException(503, "Recommendation dependency unavailable")

@app.post('/api/v1/behaviors', response_model=BehaviorEventResponse, dependencies=[Depends(authorize)])
async def record_behavior(event: BehaviorEventRequest):
    try:
        created = await run_in_threadpool(app.state.repository.record, event)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    if created:
        try:
            await app.state.feature_store.record_behavior(event.user_id, event.behavior_type, event.item_id, event.metadata)
        except Exception:
            logger.warning('feature_cache.update_failed', event_id=event.event_id)
    return BehaviorEventResponse(event_id=event.event_id, status='recorded' if created else 'duplicate', user_id=event.user_id, behavior_type=event.behavior_type, item_id=event.item_id)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000)

@app.get("/api/v1/metrics", dependencies=[Depends(authorize)])
async def metrics():
    return app.state.recommendations.metrics.snapshot()

@app.get("/", include_in_schema=False)
async def demo():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
