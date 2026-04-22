import asyncio
import re
from typing import AsyncGenerator, List, Optional
import structlog
from urllib.parse import urlparse
import requests

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
from app.ingestion.parser import parse_gcp_doc
from app.core.utils import logger


from bs4 import BeautifulSoup


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
    async def _fetch_single_sitemap(
        self, sitemap_url: str, max_pages: Optional[int] = None
    ) -> List[Document]:
        """
        Fetches documents from a single sitemap URL asynchronously with built-in retry logic.

        Args:
            sitemap_url: The URL of the sitemap XML.
            max_pages: Optional limit on the number of pages to fetch from this sitemap.
        """
        logger.info(
            "initializing_sitemap_fetch", sitemap_url=sitemap_url, max_pages=max_pages
        )

        try:
            # When testing, if we only want max_pages, we don't instantiate SitemapLoader
            # with the sitemap_url, because it forces XML fetching. Instead, we use a custom
            # loader or instantiate it differently. Let's just create it with the sitemap_url,
            # but manually bypass the .load() method if max_pages is set.
            loader = SitemapLoader(
                web_path=sitemap_url,
                filter_urls=self._build_regex_filters(),
                parsing_function=parse_gcp_doc,
                continue_on_failure=True,
            )

            loader.requests_per_second = self.requests_per_second
            loader.requests_kwargs = {
                "timeout": 15,
                "headers": {
                    "User-Agent": "Certif-Exam-Generator-Bot/1.0 (Integration/Portfolio)"
                },
            }

            if max_pages:  # utilized during testing to limit scope and speed
                # We bypass LangChain's sitemap discovery logic.
                response = await asyncio.to_thread(
                    requests.get, sitemap_url, timeout=15
                )
                soup = BeautifulSoup(response.content, "xml")
                urls = [loc.text for loc in soup.find_all("loc")]

                filters = self._build_regex_filters()
                if filters:
                    filtered_urls = [
                        u for u in urls if any(re.match(f, u) for f in filters)
                    ]
                    urls = filtered_urls

                target_urls = list(urls)[:max_pages]
                # Langchain's SitemapLoader is heavily tied to fetching the sitemap XML itself.
                # To bypass it, we call its internal scrape_all generator with our urls.
                # It yields a sequence of tuples: (soup, url)

                def _fetch_manually():
                    docs = []
                    # scrape_all yields page_soup sequentially
                    # We have to fetch and scrape manually if we want metadata matching target_urls.
                    # Or we can just use scrape_all and zip it with target_urls.
                    soups = list(loader.scrape_all(target_urls))
                    for page_soup, page_url in zip(soups, target_urls):
                        if not page_soup:
                            continue

                        text = (
                            loader.parsing_function(page_soup)
                            if loader.parsing_function
                            else page_soup.get_text()
                        )
                        if text:
                            metadata = {"source": page_url, "loc": page_url}
                            if loader.meta_function:
                                try:
                                    # Try old signature (meta_function(soup))
                                    metadata.update(loader.meta_function(page_soup))
                                except TypeError:
                                    # Fallback for newer LangChain version (meta_function(meta, _content))
                                    metadata.update(
                                        loader.meta_function(metadata, page_soup)
                                    )
                            docs.append(Document(page_content=text, metadata=metadata))
                    return docs

                docs = await asyncio.to_thread(_fetch_manually)

            else:
                docs = await asyncio.to_thread(loader.load)

            logger.info("sitemap_fetch_complete", document_count=len(docs))
            return docs

        except Exception as e:
            logger.error("sitemap_fetch_failed", error=str(e))
            raise

    def _build_regex_filters(self) -> Optional[List[str]]:
        """
        Converts the provider's URL roots into regex patterns for the SitemapLoader.
        This ensures we don't download the entirety of cloud.google.com.
        """
        # If the provider has specific roots (e.g., https://cloud.google.com/architecture),
        # we convert them to regex to filter the sitemap XML natively before fetching HTML.
        roots = self.provider.url_roots
        if not roots:
            return None

        regex_filters = []
        for root in roots:
            escaped_root = re.escape(root)
            # Ensure we match either the exact root, or a subpath/query (prevents prefix bugs)
            # e.g., matches /architecture and /architecture/foo, but NOT /architecture-center
            # Also reject non-English URLs (those with ?hl=... where ... is not en)
            regex_filters.append(
                rf"^(?!.*[?&]hl=(?!en(?:[&#]|$))){escaped_root}(?:[/?#].*)?$"
            )

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

        # Stream results as soon as each sitemap task completes
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                for doc in result:
                    # Ensure metadata integrity required for the hashing phase later
                    doc.metadata["cloud"] = self.provider.name
                    # cert_tags are injected so Pinecone metadata filtering works efficiently
                    doc.metadata["cert_tags"] = self.provider.cert_tags

                    yield doc
            except Exception as e:
                logger.error("sitemap_task_failed", error=str(e))
