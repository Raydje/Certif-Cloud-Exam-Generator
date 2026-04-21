from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings
from configs.models_pricing import models_pricing


class Settings(BaseSettings):
    # LLM
    gemini_api_key: str
    gemini_model: str 
    gemini_embedding_model: str 
    embedding_dimensions: int

    # Pricing dollar per 1M tokens (Standard Google AI Studio Pay-as-you-go)
    # Populated at runtime by set_model_pricing based on the active gemini_model
    llm_input_cost_per_1m: float = 0.0
    llm_output_cost_per_1m: float = 0.0

    @model_validator(mode="after")
    def set_model_pricing(self) -> "Settings":
        fallback = models_pricing.get("gemini-2.5-flash", [0.0, 0.0])
        pricing = models_pricing.get(self.gemini_model, fallback)
        self.llm_input_cost_per_1m = pricing[0]
        self.llm_output_cost_per_1m = pricing[1]
        return self

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_timeout: int = 15  # seconds

    # MongoDB storage + checkpoints storage for LangGraph
    mongodb_uri: str
    mongodb_database: str
    mongodb_collection: str
    mongodb_timeout: int = 15  # seconds
    mongodb_checkpoints_collection: str = "checkpoints"

    # CORS
    cors_allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8501", "http://localhost:8080"]
    
    # LLM timeout
    #TODO

    # Input validation
    #TODO

    # Graph control
    #TODO

    # Node temperatures
    # TODO

    # Auth (JWT)
    # TODO

    # Redis (rate limiting)
    # TODO

    # Per-user rate limiting & budget enforcement
    # Applies to all non-admin users (admin:all scope bypasses both checks)
    # TODO

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
