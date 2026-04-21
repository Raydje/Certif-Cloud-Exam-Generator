import asyncio
import re
from typing import AsyncGenerator, List, Optional
import structlog
from urllib.parse import urlparse
import requests.exceptions

from langchain_community.document_loaders import SitemapLoader
from langchain_core.documents import Document
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Internal imports based on your architecture
from app.core.cloud_provider import CloudProvider
from app.ingestion.parser import parse_html_document

logger = structlog.get_logger(__name__)


class ProviderSitemapLoader:
    """
    A resilient, asynchronous, provider-aware sitemap loader.

    This class wraps LangChain's SitemapLoader to fetch documentation
    pages based on a CloudProvider's specific sitemap URLs and filters.
    It implements defensive programming practices including asynchronous
    concurrency limits, automatic retries with exponential backoff, and
    custom parsing delegation.
    """

    def __init__(
        self,
        provider: CloudProvider,
        max_concurrency: int = 5,
        requests_per_second: int = 2,
    ):
        """
        Args:
            provider: The cloud provider adapter (e.g., GCP PCA adapter).
            max_concurrency: Maximum number of concurrent sitemap index fetches.
            requests_per_second: Rate limit to avoid IP bans (passed to underlying loader).
        """
        self.provider = provider
        # Fixed: sitemap_urls is a property on CloudProvider, not a method
        self.sitemap_urls = provider.sitemap_urls
        self.url_filter_func = provider.url_filter()

        self.max_concurrency = max_concurrency
        self.requests_per_second = requests_per_second

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
        ),
        reraise=True,
    )
    async def _fetch_single_sitemap(self, sitemap_url: str) -> List[Document]:
        """
        Fetches documents from a single sitemap URL asynchronously with built-in retry logic.
        """
        log = logger.bind(sitemap_url=sitemap_url)
        log.info("initializing_sitemap_fetch")

        try:
            # We use LangChain's SitemapLoader but strictly control its behavior
            loader = SitemapLoader(
                web_path=sitemap_url,
                filter_urls=self._build_regex_filters(),
                parsing_function=parse_html_document,  # Delegates to app/ingestion/parser.py
                continue_on_failure=True,  # Prevent one bad URL from crashing the batch
            )

            # Rate limiting configuration to respect target server
            loader.requests_per_second = self.requests_per_second
            loader.requests_kwargs = {
                "timeout": 15,
                "headers": {
                    "User-Agent": "Certif-Exam-Generator-Bot/1.0 (Integration/Portfolio)"
                },
            }

            # Offload blocking HTTP request cycle to a thread pool
            docs = await asyncio.to_thread(loader.load)

            log.info("sitemap_fetch_complete", document_count=len(docs))
            return docs

        except Exception as e:
            log.error("sitemap_fetch_failed", error=str(e))
            raise

    def _build_regex_filters(self) -> Optional[List[str]]:
        """
        Converts the provider's URL roots into regex patterns for the SitemapLoader.
        This ensures we don't download the entirety of cloud.google.com.
        """
        # If the provider has specific roots (e.g., https://cloud.google.com/architecture),
        # we convert them to regex to filter the sitemap XML natively before fetching HTML.
        roots = getattr(self.provider, "url_roots", None)
        if not roots:
            return None

        regex_filters = []
        for root in roots:
            parsed = urlparse(root)
            # Example: ^https://cloud\.google\.com/architecture.*$
            # re.escape handles dots, hyphens, plus signs, etc safely
            escaped = re.escape(parsed.netloc + parsed.path)
            regex_filters.append(rf"^{parsed.scheme}://{escaped}.*$")

        return regex_filters

    async def aload_all(self) -> AsyncGenerator[Document, None]:
        """
        Main entry point for asynchronous ingestion.
        Yields parsed documents ready for chunking.
        Sitemap indices are processed concurrently respecting max_concurrency.
        """
        logger.info(
            "starting_provider_ingestion",
            provider=self.provider.__class__.__name__,
            sitemap_count=len(self.sitemap_urls),
            concurrency=self.max_concurrency,
        )

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _fetch_with_semaphore(sitemap: str) -> List[Document]:
            async with semaphore:
                return await self._fetch_single_sitemap(sitemap)

        # Filter sitemaps and prepare tasks
        tasks = []
        for sitemap in self.sitemap_urls:
            if not self.url_filter_func(sitemap):
                logger.debug("skipping_sitemap_by_filter", sitemap_url=sitemap)
                continue
            tasks.append(_fetch_with_semaphore(sitemap))

        # Wait for all sitemap indices to load concurrently
        # return_exceptions=True prevents one completely failed sitemap from bringing down the others
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("sitemap_task_failed", error=str(result))
                continue

            for doc in result:
                # Ensure metadata integrity required for the hashing phase later
                doc.metadata["cloud"] = self.provider.name
                # cert_tags are injected so Pinecone metadata filtering works efficiently
                doc.metadata["cert_tags"] = getattr(self.provider, "cert_tags", [])

                yield doc
