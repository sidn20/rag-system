import asyncio
import aiohttp
import time
import statistics
from datetime import datetime

QUESTIONS = [
    "what is machine learning",
    "who created linux",
    "what are python libraries",
    "what is reinforcement learning",
    "what is artificial intelligence",
    "tell me about operating systems",
    "what is supervised learning",
    "what are neural networks",
]

async def send_request(session: aiohttp.ClientSession, question: str, request_num: int) -> dict:
    start = time.perf_counter()
    try:
        async with session.post(
            "http://localhost:8000/predict",
            json={"question": question, "top_k": 2}
        ) as response:
            data = await response.json()
            latency = (time.perf_counter() - start) * 1000
            return {
                "request_num": request_num,
                "question": question,
                "status": response.status,
                "latency_ms": round(latency, 2),
                "sources": data.get("sources", [])
            }
    except Exception as e:
        return {
            "request_num": request_num,
            "question": question,
            "status": 500,
            "latency_ms": -1,
            "error": str(e)
        }

async def run_load_test(n_requests: int = 20, concurrency: int = 5):
    print(f"\nLoad Test: {n_requests} requests, {concurrency} concurrent\n")

    questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(n_requests)]
    results = []

    async with aiohttp.ClientSession() as session:
        # Send in batches of 'concurrency'
        for batch_start in range(0, n_requests, concurrency):
            batch = questions[batch_start:batch_start + concurrency]
            tasks = [
                send_request(session, q, batch_start + i)
                for i, q in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            success = sum(1 for r in batch_results if r["status"] == 200)
            print(f"  Batch {batch_start//concurrency + 1}: {success}/{len(batch)} ok — "
                  f"avg {statistics.mean(r['latency_ms'] for r in batch_results if r['latency_ms'] > 0):.1f}ms")

    # Summary
    successful = [r for r in results if r["status"] == 200]
    latencies = [r["latency_ms"] for r in successful]

    print(f"\n{'='*50}")
    print(f"  LOAD TEST SUMMARY")
    print(f"{'='*50}")
    print(f"  Total requests   : {n_requests}")
    print(f"  Successful       : {len(successful)}")
    print(f"  Failed           : {n_requests - len(successful)}")
    print(f"  Min latency      : {min(latencies):.1f}ms")
    print(f"  Max latency      : {max(latencies):.1f}ms")
    print(f"  Mean latency     : {statistics.mean(latencies):.1f}ms")
    print(f"  Stdev            : {statistics.stdev(latencies):.1f}ms")
    print(f"  p95 latency      : {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    asyncio.run(run_load_test(n_requests=20, concurrency=5))
