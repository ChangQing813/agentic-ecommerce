"""Bounded, best-effort copy generation. Jobs contain public catalog facts only."""
import asyncio
import hashlib
import json
import time
import structlog

logger = structlog.get_logger()

class CopyCache:
    def __init__(self, redis, service, capacity=32):
        self.redis, self.service = redis, service
        self.queue = asyncio.Queue(maxsize=capacity)
        self.pending = set()
        self.worker = None
        self.cooldown = {}

    def key(self, products):
        payload = {'version':'copy-v2','model':self.service.settings.llm_model,
                   'products':[{'id':p.product_id,'name':p.name,'category':p.category} for p in products]}
        return 'copy:' + hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=True).encode()).hexdigest()

    def start(self):
        self.worker = asyncio.create_task(self.run())

    async def close(self):
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker,return_exceptions=True)
        self.pending.clear()

    async def lookup(self, products):
        key = self.key(products)
        try:
            async with asyncio.timeout(.05):
                raw = await self.redis.get(key)
            if raw:
                from services.multi_agent import CopyOutput
                copies = [c.model_dump(by_alias=True) for c in CopyOutput.model_validate_json(raw).copies]
                if len(copies)==len(products) and {c['product_id'] for c in copies}=={p.product_id for p in products}:
                    return copies, 'cache_hit'
        except Exception:
            return None, 'cache_unavailable'
        if not self.service.settings.llm_enabled or not self.service.settings.llm_api_key:
            return None, 'model_disabled_or_missing_key'
        if key in self.pending:
            return None, 'generation_pending'
        if self.cooldown.get(key,0)>time.monotonic():
            return None, 'generation_cooldown'
        try:
            self.queue.put_nowait((key,list(products)))
            self.pending.add(key)
            return None, 'generation_queued'
        except asyncio.QueueFull:
            return None, 'generation_queue_full'

    async def run(self):
        from services.multi_agent import call_trace
        while True:
            key, products = await self.queue.get()
            token = call_trace.set({'attempts':0,'max_attempts':1,'transport_timeout':10})
            started = time.monotonic()
            try:
                state = {'products':products}
                async with asyncio.timeout(12):
                    await self.service.marketing.execute(state,self.service.gateway)
                    await self.redis.set(key,json.dumps({'copies':state['copies']},ensure_ascii=True),ex=3600)
                self.service.metrics.record('marketing_background','success',None,(time.monotonic()-started)*1000)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Bound failure metadata; queue is best-effort, not a durable task system.
                if len(self.cooldown)>=256:
                    self.cooldown.clear()
                self.cooldown[key] = time.monotonic()+30
                self.service.metrics.record('marketing_background','degraded',type(exc).__name__,(time.monotonic()-started)*1000)
                logger.warning('copy.background_failed',error=type(exc).__name__)
            finally:
                call_trace.reset(token)
                self.pending.discard(key)
                self.queue.task_done()
