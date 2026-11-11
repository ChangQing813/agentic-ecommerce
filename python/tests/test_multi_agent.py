import asyncio
import unittest
from config.settings import Settings
from models.schemas import Product, RecommendationRequest
from services.multi_agent import MultiAgentRecommendationService

class Repo:
    def catalog(self):
        return [Product(product_id='a',name='A',category='手机',price=10,stock=1),Product(product_id='b',name='B',category='耳机',price=20,stock=2)]
    def recent_events(self):
        return []

class Gateway:
    def __init__(self,settings,mode='ok'):
        self.settings, self.mode = settings, mode
        self.cancelled = False
    async def ask(self,role,payload):
        if self.mode=='hang':
            try:
                await asyncio.sleep(10)
            finally:
                self.cancelled = True
        if 'category preferences' in role:
            return {'categories':[]}
        if 'Rank candidate' in role:
            return {'ids':['invented'] if self.mode=='invalid' else ['b','a']}
        return {'copies':[{'product_id':p['id'],'copy':'商品推荐'} for p in payload['products']]}

class MultiAgentTests(unittest.IsolatedAsyncioTestCase):
    def service(self, mode='ok', repo=None, **kwargs):
        settings = Settings(llm_enabled=True, recommendation_timeout=1, agent_timeout_user_profile=.03,agent_timeout_product_rec=.03,agent_timeout_marketing_copy=.03,**kwargs)
        gateway = Gateway(settings,mode)
        return MultiAgentRecommendationService(repo or Repo(),settings,gateway)

    async def test_model_success_and_request_isolation(self):
        service = self.service()
        a,b = await asyncio.gather(service.recommend(RecommendationRequest(user_id='a',num_items=1)),service.recommend(RecommendationRequest(user_id='b',num_items=2)))
        self.assertEqual(a.products[0].product_id,'b')
        self.assertEqual(len(b.products),2)
        self.assertNotEqual(a.request_id,b.request_id)
        self.assertFalse(a.degraded)
        self.assertEqual(set(a.agent_results),{'user_profile','inventory','product_rec','marketing_copy'})

    async def test_hallucinated_ids_are_rejected(self):
        result = await self.service('invalid').recommend(RecommendationRequest(user_id='u'))
        self.assertTrue(result.degraded)
        self.assertEqual(result.agent_results['product_rec'].status,'degraded')
        self.assertEqual({p.product_id for p in result.products},{'a','b'})

    async def test_timeout_fallback_and_cancelled_model_work(self):
        service = self.service('hang')
        result = await asyncio.wait_for(service.recommend(RecommendationRequest(user_id='u')), .5)
        self.assertTrue(result.degraded)
        self.assertTrue(service.gateway.cancelled)
        self.assertEqual(len(result.products),2)
        self.assertEqual(len(result.marketing_copies),2)

    async def test_inventory_failure_fails_closed(self):
        class BrokenInventory(Repo):
            calls = 0
            def catalog(self):
                self.calls += 1
                if self.calls>1:
                    raise OSError('inventory unavailable')
                return super().catalog()
        result = await self.service(repo=BrokenInventory()).recommend(RecommendationRequest(user_id='u'))
        self.assertEqual(result.products,[])
        self.assertEqual(result.agent_results['inventory'].status,'degraded')

    async def test_request_cancellation_propagates(self):
        service = self.service('hang')
        task = asyncio.create_task(service.recommend(RecommendationRequest(user_id='u')))
        await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_database_failure_is_not_fake_success(self):
        class Broken(Repo):
            def catalog(self):
                raise OSError('database unavailable')
        with self.assertRaises(OSError):
            await self.service(repo=Broken()).recommend(RecommendationRequest(user_id='u'))

class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_only_transient_and_validate_json(self):
        import httpx
        from services.multi_agent import ModelGateway
        settings = Settings(llm_enabled=True,llm_api_key='test')
        gateway = ModelGateway(settings)
        await gateway.client.aclose()
        calls = []
        async def handler(request):
            calls.append(request)
            if len(calls)==1:
                return httpx.Response(429)
            return httpx.Response(200,json={'choices':[{'message':{'content':'{"ids":["a"]}'}}]})
        gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            self.assertEqual(await gateway.ask('rank',{}),{'ids':['a']})
            self.assertEqual(len(calls),2)
        finally:
            await gateway.close()

    async def test_auth_error_is_not_retried(self):
        import httpx
        from services.multi_agent import ModelGateway
        gateway = ModelGateway(Settings(llm_enabled=True,llm_api_key='test'))
        await gateway.client.aclose()
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(401)
        gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with self.assertRaises(httpx.HTTPStatusError):
                await gateway.ask('rank',{})
            self.assertEqual(len(calls),1)
        finally:
            await gateway.close()

class GovernanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_overload_and_cancellation_release_capacity(self):
        from services.governance import Overloaded
        settings = Settings(llm_enabled=True,max_active_requests=1,agent_timeout_user_profile=5)
        service = MultiAgentRecommendationService(Repo(),settings,Gateway(settings,'hang'))
        task = asyncio.create_task(service.recommend(RecommendationRequest(user_id='a')))
        await asyncio.sleep(.03)
        with self.assertRaises(Overloaded):
            await service.recommend(RecommendationRequest(user_id='b'))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(service.active,0)

    async def test_breaker_single_probe_and_recovery(self):
        from services.governance import CircuitBreaker, CircuitOpen
        breaker = CircuitBreaker(threshold=1,cooldown=0)
        breaker.failure()
        breaker.enter()
        with self.assertRaises(CircuitOpen):
            breaker.enter()
        breaker.success()
        breaker.enter()

    async def test_profile_must_preserve_categories(self):
        from services.multi_agent import ProfileAgent
        settings = Settings(llm_enabled=True)
        state = {'catalog':Repo().catalog(),'events':[{'user_id':'u','item_id':'a','behavior_type':'click'}],'request':RecommendationRequest(user_id='u')}
        with self.assertRaises(ValueError):
            await ProfileAgent().execute(state,Gateway(settings))

class LatencyPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_cold_start_skips_profile_and_inventory_has_no_model(self):
        settings = Settings(llm_enabled=True)
        class Counting(Gateway):
            roles = []
            async def ask(self,role,payload):
                self.roles.append(role)
                return await super().ask(role,payload)
        gateway = Counting(settings)
        result = await MultiAgentRecommendationService(Repo(),settings,gateway).recommend(RecommendationRequest(user_id='new'))
        self.assertFalse(any('category preferences' in r for r in gateway.roles))
        self.assertEqual(result.agent_results['user_profile'].data['decision'],'cold_start_default_profile')
        self.assertIsNone(result.agent_results['inventory'].data['model'])

    async def test_copy_policy_does_not_retry_transport_timeout(self):
        import httpx
        from services.multi_agent import ModelGateway, call_trace
        gateway = ModelGateway(Settings(llm_enabled=True,llm_api_key='test'))
        await gateway.client.aclose()
        calls = []
        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout('test',request=request)
        gateway.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        token = call_trace.set({'attempts':0,'max_attempts':1})
        try:
            with self.assertRaises(httpx.ReadTimeout):
                await gateway.ask('copy',{})
            self.assertEqual(len(calls),1)
        finally:
            call_trace.reset(token)
            await gateway.close()
