import contextlib
import re

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import MongoClient
from configs.setting import get_settings
from app.core.utils import logger

_INDEX_FIELDS = [("rule_type", 1), ("section", 1), ("rule_number", 1)]
_ID_RE = re.compile(r"^MISRA_(RULE|DIR)_(\d+)\.(\d+)$")


# Sync pymongo client — MongoDBSaver (langgraph-checkpoint-mongodb) requires pymongo, not Motor.
class MongoDBCheckpointService:
    """
    Thin wrapper around a synchronous MongoDB client for LangGraph checkpoint storage.
    This service is used by the langgraph-checkpoint-mongodb package to persist graph checkpoints.
        Responsibilities:
            - client init
            - insert checkpoint
            - retrieve checkpoint by ID
    Note: This service is separate from the main MongoDBService which uses Motor for async operations related to MISRA rules storage and retrieval.
    The separation ensures that the synchronous operations required by LangGraph do not interfere with the asynchronous operations
    """
    def __init__(self) -> None:
        settings = get_settings()
        self.client: MongoClient = MongoClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_database]

    def close(self) -> None:
        self.client.close()


# MongoDB service for cloud documentation storage and retrieval
class MongoDBService:
    def __init__(self) -> None:
        settings = get_settings()
        timeout_ms = settings.mongodb_timeout * 1000
        # Driver-level timeouts ensure the Motor client respects connection and
        # operation deadlines natively, rather than relying on asyncio.wait_for
        # which cancels the coroutine mid-flight and can leave the connection
        # pool in an inconsistent state.
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            socketTimeoutMS=timeout_ms,
        )
        self.db = self.client[settings.mongodb_database]
        self.collection: AsyncIOMotorCollection = self.db[settings.mongodb_collection]

    def close(self) -> None:
        self.client.close()
