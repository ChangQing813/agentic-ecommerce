from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Multi-Agent E-Commerce System"
    debug: bool = False
    database_url: str = "postgresql+psycopg://ecom:ecom@localhost:5432/ecom"
    api_key: str = ""
    cors_origins: list[str] = ["http://localhost:3000"]

    max_active_requests: int = Field(default=32, ge=1, le=1024)
    recommendation_timeout: float = Field(default=12.0, gt=0, le=60)
    llm_enabled: bool = False
    llm_max_concurrency: int = Field(default=8, ge=1, le=128)
    marketing_wait_budget: float = Field(default=2.0, gt=0, le=10)
    llm_call_timeout: float = 3.0

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "qwen-max"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048

    # Optional Redis FeatureStore configuration
    redis_url: str = "redis://localhost:6379/0"
    feature_ttl_seconds: int = 86400

    # A/B Testing
    ab_test_enabled: bool = True
    ab_test_default_bucket_count: int = 100

    # Agent timeouts (seconds)
    agent_timeout_user_profile: float = 5.0
    agent_timeout_product_rec: float = 8.0
    agent_timeout_marketing_copy: float = 10.0
    agent_timeout_inventory: float = 5.0

    model_config = {"env_file": ".env", "env_prefix": "ECOM_"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()

