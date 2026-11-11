"""Transactional catalog and event repository. Redis is a disposable cache."""
import json
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Text, Boolean, Float, select, insert
from sqlalchemy.exc import IntegrityError

metadata = MetaData()
products = Table('products', metadata,
    Column('product_id', String(128), primary_key=True),
    Column('payload', Text, nullable=False),
    Column('is_active', Boolean, nullable=False),
    Column('stock', Integer, nullable=False))
events = Table('behavior_events', metadata,
    Column('event_id', String(128), primary_key=True),
    Column('user_id', String(128), nullable=False, index=True),
    Column('item_id', String(128), nullable=False),
    Column('behavior_type', String(16), nullable=False),
    Column('occurred_at', Float, nullable=False),
    Column('received_at', Float, nullable=False),
    Column('payload', Text, nullable=False))
versions = Table('schema_versions', metadata, Column('version', Integer, primary_key=True))

class Repository:
    def __init__(self, url):
        options = {}
        if url.startswith("postgresql"):
            options = {"connect_args": {"connect_timeout": 3, "options": "-c statement_timeout=3000 -c lock_timeout=1000"}, "pool_size": 8, "max_overflow": 0, "pool_timeout": 1}
        self.engine = create_engine(url, future=True, pool_pre_ping=True, **options)

    def migrate(self):
        # Version 1 is additive. Run once before starting API workers.
        metadata.create_all(self.engine)
        with self.engine.begin() as c:
            if c.execute(select(versions.c.version)).first() is None:
                c.execute(insert(versions).values(version=1))

    def catalog(self):
        from models.schemas import Product
        with self.engine.connect() as c:
            rows = c.execute(select(products.c.payload, products.c.is_active, products.c.stock)).all()
        return [Product(**{**json.loads(r.payload), 'is_active': r.is_active, 'stock': r.stock}) for r in rows]

    def seed(self, items):
        with self.engine.begin() as c:
            existing = set(c.execute(select(products.c.product_id)).scalars())
            for p in items:
                if p.product_id not in existing:
                    c.execute(insert(products).values(product_id=p.product_id, payload=p.model_dump_json(), is_active=p.is_active, stock=p.stock))

    def record(self, event):
        if event.occurred_at.tzinfo is None:
            raise ValueError('occurred_at must include timezone')
        if event.occurred_at.timestamp() > datetime.now(timezone.utc).timestamp() + 300:
            raise ValueError('occurred_at cannot be more than 5 minutes in the future')
        payload = event.model_dump_json()
        try:
            with self.engine.begin() as c:
                if c.execute(select(products.c.product_id).where(products.c.product_id == event.item_id)).first() is None:
                    raise LookupError('Unknown product')
                c.execute(insert(events).values(event_id=event.event_id, user_id=event.user_id, item_id=event.item_id, behavior_type=event.behavior_type, occurred_at=event.occurred_at.timestamp(), received_at=datetime.now(timezone.utc).timestamp(), payload=payload))
            return True
        except IntegrityError:
            with self.engine.connect() as c:
                old = c.execute(select(events.c.payload).where(events.c.event_id == event.event_id)).scalar_one_or_none()
            if old is None:
                raise
            # Server-generated receive time is excluded from the identity.
            if json.loads(old) != json.loads(payload):
                raise ValueError('event_id already belongs to a different event')
            return False

    def recent_events(self):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).timestamp()
        with self.engine.connect() as c:
            return [dict(r) for r in c.execute(select(events).where(events.c.occurred_at >= cutoff)).mappings()]

    def ready(self):
        with self.engine.connect() as c:
            if c.execute(select(versions.c.version)).scalar_one() != 1:
                raise RuntimeError('Unsupported schema version')
