"""Database-backed deterministic recommendation baseline."""
from collections import Counter
from time import perf_counter
from uuid import uuid4
from models.schemas import RecommendationResponse

class RecommendationService:
    def __init__(self, repository):
        self.repository = repository

    def recommend(self, request):
        started = perf_counter()
        catalog = self.repository.catalog()
        by_id = {p.product_id: p for p in catalog}
        popularity, preferences = Counter(), Counter()
        weights = {'view': 1, 'click': 2, 'cart': 3, 'purchase': 5}
        for e in self.repository.recent_events():
            weight = weights[e['behavior_type']]
            popularity[e['item_id']] += weight
            if e['user_id'] == request.user_id and e['item_id'] in by_id:
                preferences[by_id[e['item_id']].category] += weight
        sellable = [p for p in catalog if p.is_active and p.stock > 0]
        popular = sorted(sellable, key=lambda p: (-popularity[p.product_id], p.product_id))
        preferred = sorted((p for p in sellable if preferences[p.category]), key=lambda p: (-preferences[p.category], -popularity[p.product_id], p.product_id))
        # Both channels reserve candidate capacity before merging.
        limit = request.num_items * 3
        pool = {p.product_id: p for p in preferred[:limit] + popular[:limit]}
        ordered = sorted(pool.values(), key=lambda p: (-preferences[p.category], -popularity[p.product_id], p.product_id))
        selected = []
        while ordered and len(selected) < request.num_items:
            index = next((i for i,p in enumerate(ordered) if not selected or p.category != selected[-1].category), 0)
            selected.append(ordered.pop(index))
        return RecommendationResponse(request_id=str(uuid4()), user_id=request.user_id, products=selected,
            marketing_copies=[{'product_id': p.product_id, 'copy': f'{p.name}，了解商品详情。'} for p in selected],
            total_latency_ms=round((perf_counter()-started)*1000, 1))
