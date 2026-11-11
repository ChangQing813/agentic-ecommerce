import asyncio
import unittest
from types import SimpleNamespace
from models.schemas import Product
from services.copy_cache import CopyCache

class CopyCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_dedup_cache_and_shutdown(self):
        class Redis:
            data = {}
            async def get(self,k): return self.data.get(k)
            async def set(self,k,v,ex): self.data[k]=v
        class Marketing:
            async def execute(self,state,gateway):
                state['copies']=[{'product_id':p.product_id,'copy':'test'} for p in state['products']]
        class Metrics:
            def record(self,*args): pass
        service=SimpleNamespace(settings=SimpleNamespace(llm_model='test',llm_enabled=True,llm_api_key='test'),marketing=Marketing(),gateway=None,metrics=Metrics())
        cache=CopyCache(Redis(),service,capacity=1)
        products=[Product(product_id='p',name='test',category='c',price=1,stock=1)]
        self.assertEqual((await cache.lookup(products))[1],'generation_queued')
        self.assertEqual((await cache.lookup(products))[1],'generation_pending')
        other=[products[0].model_copy(update={'name':'changed'})]
        self.assertEqual((await cache.lookup(other))[1],'generation_queue_full')
        cache.start()
        try:
            await asyncio.wait_for(cache.queue.join(),1)
            copies,reason=await cache.lookup(products)
            self.assertEqual(reason,'cache_hit')
            self.assertEqual(copies[0]['product_id'],'p')
        finally:
            await cache.close()
        self.assertTrue(cache.worker.done())
