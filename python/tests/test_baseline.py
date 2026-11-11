import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
from sqlalchemy import update
from models.schemas import Product, BehaviorEventRequest, RecommendationRequest
from services.repository import Repository, products
from services.recommendation import RecommendationService

class BaselineTests(unittest.TestCase):
    def test_persistence_dedup_preferences_and_availability(self):
        with tempfile.TemporaryDirectory() as folder:
            url = 'sqlite:///' + str(Path(folder) / 'test.db')
            repo = Repository(url)
            repo.migrate()
            repo.seed([Product(product_id='a', name='A', category='手机', price=1, stock=5), Product(product_id='b', name='B', category='耳机', price=2, stock=5)])
            event = BehaviorEventRequest(event_id='e1', user_id='u', item_id='b', behavior_type='click', occurred_at=datetime.now(timezone.utc))
            self.assertTrue(repo.record(event))
            self.assertFalse(repo.record(event))
            with self.assertRaises(ValueError):
                repo.record(event.model_copy(update={'user_id':'other'}))
            # Another user's purchases make A popular, while U still prefers B.
            for i in range(3):
                repo.record(BehaviorEventRequest(event_id=f'other-{i}', user_id='other', item_id='a', behavior_type='purchase'))
            repo.engine.dispose()
            repo = Repository(url)
            try:
                service = RecommendationService(repo)
                request = RecommendationRequest(user_id='u', num_items=1)
                self.assertEqual(service.recommend(request).products[0].product_id, 'b')
                self.assertEqual(service.recommend(RecommendationRequest(user_id='new', num_items=1)).products[0].product_id, 'a')
                with repo.engine.begin() as c:
                    c.execute(update(products).where(products.c.product_id=='b').values(is_active=False))
                self.assertEqual(service.recommend(request).products[0].product_id, 'a')
                with repo.engine.begin() as c:
                    c.execute(update(products).values(stock=0))
                self.assertEqual(service.recommend(request).products, [])
                self.assertEqual(len(repo.recent_events()), 4)
            finally:
                repo.engine.dispose()

if __name__ == '__main__':
    unittest.main()
