"""Bounded HTTP benchmark; explicit --allow-model required for recommendation calls."""
import argparse, asyncio, json, math, os, time
from pathlib import Path
import httpx

async def run(args):
    if not args.allow_model:
        raise SystemExit('Recommendation requests may incur model costs. Pass --allow-model explicitly.')
    queue=asyncio.Queue()
    for i in range(args.requests): queue.put_nowait(i)
    results=[]
    async with httpx.AsyncClient(timeout=65,headers={'X-API-Key':os.getenv('ECOM_API_KEY','')}) as client:
        async def worker():
            while not queue.empty():
                i=queue.get_nowait(); start=time.perf_counter()
                try:
                    r=await client.post(args.url.rstrip('/')+'/api/v1/recommend',json={'user_id':'benchmark-user','num_items':5,'scene':'homepage'})
                    body=r.json()
                    results.append({'status':r.status_code,'latency_ms':(time.perf_counter()-start)*1000,'degraded':body.get('degraded'),'products':len(body.get('products',[]))})
                except Exception as e:
                    results.append({'status':0,'latency_ms':(time.perf_counter()-start)*1000,'error':type(e).__name__})
                finally: queue.task_done()
        start=time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed=time.perf_counter()-start
    lat=sorted(r['latency_ms'] for r in results)
    percentile=lambda p:lat[max(0,math.ceil(len(lat)*p)-1)]
    report={'mode':'live-http','url':args.url,'requests':len(results),'concurrency':args.concurrency,'elapsed_seconds':elapsed,'throughput_including_rejections':len(results)/elapsed,'success_rate':sum(r['status']==200 for r in results)/len(results),'p50_ms':percentile(.5),'p95_ms':percentile(.95),'p99_ms':percentile(.99),'results':results,'limitations':'Small samples are smoke tests, not capacity evidence. 429 responses are counted separately from successes. No CTR/CVR claims.'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8000');p.add_argument('--requests',type=int,default=1);p.add_argument('--concurrency',type=int,default=1);p.add_argument('--allow-model',action='store_true');p.add_argument('--output',default='reports/http-benchmark.json');a=p.parse_args()
    if not 1<=a.requests<=1000 or not 1<=a.concurrency<=64: p.error('requests: 1..1000; concurrency: 1..64')
    asyncio.run(run(a))
