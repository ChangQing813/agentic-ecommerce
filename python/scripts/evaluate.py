"""Offline synthetic rule evaluation, independent of production data and credentials."""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.schemas import Product, BehaviorEventRequest, RecommendationRequest
from services.repository import Repository
from services.recommendation import RecommendationService

def main():
    cases=[]
    with tempfile.TemporaryDirectory() as folder:
        repo=Repository('sqlite:///'+str(Path(folder)/'evaluation.db'))
        try:
            repo.migrate()
            repo.seed([Product(product_id='phone',name='phone',category='phone',price=100,stock=3),Product(product_id='tablet',name='tablet',category='tablet',price=200,stock=3),Product(product_id='sold-out',name='sold-out',category='tablet',price=1,stock=0),Product(product_id='inactive',name='inactive',category='tablet',price=1,stock=3,is_active=False)])
            service=RecommendationService(repo)
            for i in range(30):
                category='phone' if i%2==0 else 'tablet'
                repo.record(BehaviorEventRequest(event_id=f'eval-{i}',user_id=f'eval-{i}',item_id=category,behavior_type='click'))
                response=service.recommend(RecommendationRequest(user_id=f'eval-{i}',num_items=1))
                cases.append({'case':i,'expected_category':category,'actual_category':response.products[0].category,'passed':response.products[0].category==category and all(p.stock>0 and p.is_active for p in response.products)})
        finally: repo.engine.dispose()
    report={'mode':'synthetic-offline-rule','cases':cases,'passed':sum(c['passed'] for c in cases),'total':len(cases),'limitations':'Evaluates deterministic baseline constraints only. No live LLM, human relevance rating, CTR or CVR comparison.'}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/offline-evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"Synthetic rule evaluation: {report['passed']}/{report['total']}")
    if report['passed']!=report['total']: raise SystemExit(1)
if __name__=='__main__': main()
