import asyncio
import re
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
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
        return ["https://mock.cloud.com/sitemap_index.xml"]

    def url_filter(self) -> Callable[[str], bool]:
        def _filter(url: str) -> bool:
            return "skip" not in url

        return _filter

    def cert_blueprints(self, tag: str) -> List[CertBlueprint]:
        return []


@pytest.fixture
def provider():
    return MockProvider()


def test_build_regex_filters_none(provider):
    """Test that URL filtering returns None if no roots are specified."""
    provider.url_roots = []
    loader = ProviderSitemapLoader(provider)
    assert loader._build_regex_filters() is None


def test_build_regex_filters(provider):
    """Test that URL prefixes build strictly matching regex filters."""
    loader = ProviderSitemapLoader(provider)
    filters = loader._build_regex_filters()
    assert len(filters) == 1

    # Should match standard urls and nested paths
    assert re.match(filters[0], "https://mock.cloud.com/docs")
    assert re.match(filters[0], "https://mock.cloud.com/docs/page1")
    # Should skip non-english language params
    assert not re.match(filters[0], "https://mock.cloud.com/docs/page1?hl=fr")


@pytest.mark.asyncio
@patch("app.ingestion.sitemap_loader.requests.get")
async def test_get_sub_sitemaps(mock_get, provider):
    """Test parsing sitemap indexes to extract sub-sitemaps."""
    mock_response = MagicMock()
    mock_response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <sitemapindex>
        <sitemap><loc>https://mock.cloud.com/sitemap1.xml</loc></sitemap>
        <sitemap><loc>https://mock.cloud.com/sitemap2.xml</loc></sitemap>
    </sitemapindex>
    """
    mock_get.return_value = mock_response

    loader = ProviderSitemapLoader(provider)
    sub_sitemaps = await loader.get_sub_sitemaps()

    assert sub_sitemaps == [
        "https://mock.cloud.com/sitemap1.xml",
        "https://mock.cloud.com/sitemap2.xml",
    ]


@pytest.mark.asyncio
@patch("app.ingestion.sitemap_loader.requests.get")
@patch("app.ingestion.sitemap_loader.SitemapLoader")
async def test_fetch_single_sitemap_with_max_pages(
    mock_sitemap_loader_cls, mock_get, provider
):
    """Test manual fetch fallback when max_pages is explicitly set."""
    mock_resp = MagicMock()
    mock_resp.content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset>
        <url><loc>https://mock.cloud.com/docs/1</loc></url>
        <url><loc>https://mock.cloud.com/docs/2</loc></url>
        <url><loc>https://mock.cloud.com/docs/3</loc></url>
    </urlset>
    """
    mock_get.return_value = mock_resp

    mock_loader_instance = mock_sitemap_loader_cls.return_value
    mock_loader_instance.parsing_function = None
    mock_loader_instance.meta_function = None

    # Simulate Langchain's scrape_all returning BeautifulSoup objects
    mock_soup_1 = MagicMock()
    mock_soup_1.get_text.return_value = "Content 1"
    mock_soup_2 = MagicMock()
    mock_soup_2.get_text.return_value = "Content 2"
    mock_loader_instance.scrape_all.return_value = iter([mock_soup_1, mock_soup_2])

    loader = ProviderSitemapLoader(provider)
    docs = await loader._fetch_single_sitemap(
        "https://mock.cloud.com/sitemap1.xml", max_pages=2
    )

    # Assert truncated to 2 pages
    assert len(docs) == 2
    assert docs[0].page_content == "Content 1"
    assert docs[0].metadata["source"] == "https://mock.cloud.com/docs/1"


@pytest.mark.asyncio
async def test_aload_all_streams_and_handles_errors(provider):
    """Test concurrent streams and ingestion resiliency using mocked sub_sitemaps."""
    loader = ProviderSitemapLoader(provider=provider, max_concurrency=5)
    events = []

    # Mock index fetcher
    loader.get_sub_sitemaps = AsyncMock(
        return_value=[
            "https://mock.cloud.com/sitemap1.xml",
            "https://mock.cloud.com/sitemap_skip.xml",  # should be skipped by filter
            "https://mock.cloud.com/sitemap2.xml",
            "https://mock.cloud.com/sitemap3_error.xml",
        ]
    )

    async def mock_fetch_single_sitemap(
        sitemap_url: str, max_pages=None
    ) -> List[Document]:
        if "sitemap1" in sitemap_url:
            await asyncio.sleep(0.01)
            events.append("sitemap1_done")
            return [Document(page_content="doc1", metadata={})]
        elif "sitemap3_error" in sitemap_url:
            await asyncio.sleep(0.05)
            events.append("sitemap3_error")
            raise ValueError("Simulated network error")
        elif "sitemap2" in sitemap_url:
            await asyncio.sleep(0.1)
            events.append("sitemap2_done")
            return [Document(page_content="doc2", metadata={})]
        return []

    loader._fetch_single_sitemap = AsyncMock(side_effect=mock_fetch_single_sitemap)

    results = []
    async for doc in loader.aload_all():
        results.append(doc)
        events.append(f"yielded_{doc.page_content}")

    # Ensure filter successfully skipped "sitemap_skip"
    assert loader._fetch_single_sitemap.call_count == 3

    # Ensure stream execution matches async completion times instead of batch blocks
    idx_s1 = events.index("sitemap1_done")
    idx_y1 = events.index("yielded_doc1")
    idx_e3 = events.index("sitemap3_error")
    idx_s2 = events.index("sitemap2_done")
    idx_y2 = events.index("yielded_doc2")

    assert idx_s1 < idx_y1 < idx_e3 < idx_s2 < idx_y2

    assert len(results) == 2
    for doc in results:
        assert doc.metadata["cloud"] == "mock-cloud"
        assert doc.metadata["cert_tags"] == ["mock-cert"]
