import asyncio
import pytest
from unittest.mock import AsyncMock
from typing import List, Callable
from langchain_core.documents import Document

from app.ingestion.sitemap_loader import ProviderSitemapLoader
from app.core.cloud_provider import CertBlueprint


class MockProvider:
    name = "mock-cloud"
    url_roots = ["https://mock.cloud.com/docs"]
    cert_tags = ["mock-cert"]

    @property
    def sitemap_urls(self) -> List[str]:
        return [
            "https://mock.cloud.com/sitemap1.xml",
            "https://mock.cloud.com/sitemap2.xml",
            "https://mock.cloud.com/sitemap3.xml",
            "https://mock.cloud.com/sitemap4.xml",
        ]

    def url_filter(self) -> Callable[[str], bool]:
        def _filter(url: str) -> bool:
            # Filter out sitemap3 to test skipping
            return "sitemap3" not in url

        return _filter

    def cert_blueprints(self, tag: str) -> List[CertBlueprint]:
        return []


@pytest.mark.asyncio
async def test_aload_all_streams_and_handles_errors():
    provider = MockProvider()
    loader = ProviderSitemapLoader(provider=provider, max_concurrency=5)

    events = []

    async def mock_fetch_single_sitemap(
        sitemap_url: str, max_pages=None
    ) -> List[Document]:
        if "sitemap1" in sitemap_url:
            await asyncio.sleep(0.01)
            events.append("sitemap1_done")
            return [Document(page_content="doc1")]
        elif "sitemap4" in sitemap_url:
            await asyncio.sleep(0.05)
            events.append("sitemap4_error")
            raise ValueError("Simulated network error")
        elif "sitemap2" in sitemap_url:
            await asyncio.sleep(0.1)
            events.append("sitemap2_done")
            return [Document(page_content="doc2")]
        return []

    loader._fetch_single_sitemap = AsyncMock(side_effect=mock_fetch_single_sitemap)

    results = []
    async for doc in loader.aload_all():
        results.append(doc)
        events.append(f"yielded_{doc.page_content}")

    # sitemap3 is filtered out, so it should be called exactly 3 times
    assert loader._fetch_single_sitemap.call_count == 3

    # Ensure it yielded doc1 before doc2 finished, proving it streamed instead of waited
    print("Events sequence:", events)
    idx_s1 = events.index("sitemap1_done")
    idx_y1 = events.index("yielded_doc1")
    idx_e4 = events.index("sitemap4_error")
    idx_s2 = events.index("sitemap2_done")
    idx_y2 = events.index("yielded_doc2")

    assert idx_s1 < idx_y1 < idx_e4 < idx_s2 < idx_y2

    assert len(results) == 2
    for doc in results:
        assert doc.metadata["cloud"] == "mock-cloud"
        assert doc.metadata["cert_tags"] == ["mock-cert"]
