"""Process-local admission, circuit breaker and bounded aggregate metrics."""
import time
from collections import Counter

class Overloaded(Exception):
    pass

class CircuitOpen(Exception):
    pass

class CircuitBreaker:
    def __init__(self, threshold=5, cooldown=30):
        self.threshold, self.cooldown = threshold, cooldown
        self.failures = 0
        self.until = 0
        self.probing = False
    def enter(self):
        if self.until:
            if time.monotonic() < self.until or self.probing:
                raise CircuitOpen('model_circuit_open')
            self.probing = True
    def success(self):
        self.failures, self.until, self.probing = 0, 0, False
    def failure(self):
        self.failures += 1
        if self.probing or self.failures >= self.threshold:
            self.until = time.monotonic() + self.cooldown
        self.probing = False
    def cancel(self):
        self.probing = False

class GovernanceMetrics:
    def __init__(self):
        self.counts = Counter()
        self.latency = Counter()
        self.buckets = Counter()
    def record(self, name, status, reason, elapsed):
        self.counts[(name,status,reason or 'none')] += 1
        self.latency[name] += elapsed
        for upper in (10,50,100,500,1000,3000,10000,float("inf")):
            if elapsed <= upper:
                self.buckets[(name,str(upper))] += 1
    def snapshot(self):
        return {'nodes':[{'agent':n,'status':s,'reason':r,'count':v} for (n,s,r),v in self.counts.items()], 'total_node_latency_ms':dict(self.latency), 'latency_buckets':[{'agent':n,'le_ms':b,'count':v} for (n,b),v in self.buckets.items()]}
