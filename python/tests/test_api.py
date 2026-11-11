import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from fastapi.testclient import TestClient
from config import get_settings
from services.repository import Repository
from models.schemas import Product

class APITests(unittest.TestCase):
    def test_api_without_model_and_with_unavailable_redis(self):
        import main
        with tempfile.TemporaryDirectory() as folder:
            previous = main.settings
            main.settings = previous.model_copy(update={'database_url': 'sqlite:///' + str(Path(folder)/'api.db'), 'redis_url': 'redis://localhost:1/0', 'api_key': 'test-secret', 'llm_enabled': False})
            repo = Repository(main.settings.database_url)
            repo.migrate()
            repo.seed([Product(product_id='p', name='商品', category='耳机', price=99, stock=3)])
            repo.engine.dispose()
            try:
                with TestClient(main.app) as client:
                    self.assertEqual(client.get('/ready').status_code, 200)
                    page = client.get('/')
                    self.assertEqual(page.status_code, 200)
                    self.assertIn('/api/v1/recommend', page.text)
                    self.assertIn('/api/v1/behaviors', page.text)
                    self.assertEqual(client.post('/api/v1/recommend', json={'user_id':'u'}).status_code, 401)
                    headers = {'X-API-Key':'test-secret'}
                    event = {'event_id':'api-event', 'user_id':'u', 'item_id':'p', 'behavior_type':'click', 'occurred_at':datetime.now(timezone.utc).isoformat()}
                    first = client.post('/api/v1/behaviors', json=event, headers=headers)
                    self.assertEqual(first.status_code, 200, first.text)
                    self.assertEqual(client.post('/api/v1/behaviors', json=event, headers=headers).json()['status'], 'duplicate')
                    response = client.post('/api/v1/recommend', json={'user_id':'u'}, headers=headers)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()['products'][0]['product_id'], 'p')
                    self.assertEqual(client.post('/api/v1/recommend', json={'user_id':'u','num_items':0}, headers=headers).status_code, 422)
            finally:
                main.settings = previous
