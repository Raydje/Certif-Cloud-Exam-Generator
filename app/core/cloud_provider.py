"""
CloudProvider protocol and GCP v1 implementation.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CertBlueprint:
    """One exam domain from an official certification blueprint.

    Used by the Planner node to build QuestionPlan objects proportional
    to each domain's exam weight.
    """

    cert_id: str  # e.g. "gcp-pca"
    domain_id: str  # e.g. "1" — stable key, survives label renames
    domain_name: str  # human-readable label from Google's exam guide
    weight_pct: int  # official exam weight percentage (sum ≈ 100)
    topics: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CloudProvider(Protocol):
    """Structural interface every cloud provider adapter must satisfy.

    Implementing classes do NOT need to inherit from this; duck typing
    is enough. Use `isinstance(obj, CloudProvider)` for runtime checks.
    """

    name: str
    """Short identifier used as the Pinecone namespace (e.g. "gcp")."""

    url_roots: list[str]
    """URL prefixes that scope ingestion; converted to regex by ProviderSitemapLoader."""

    cert_tags: list[str]
    """Pinecone metadata tags injected on every chunk (e.g. ["gcp", "gcp-pca"])."""
    
    @property
    def sitemap_urls(self) -> list[str]:
        """Return the sitemap index URL(s) to crawl."""
        ...

    def url_filter(self) -> Callable[[str], bool]:
        """Return a predicate that accepts a sitemap root URL.

        Applied by ProviderSitemapLoader before fetching each sitemap entry.
        Use it to exclude CDN mirrors, translated copies, etc.
        """
        ...

    def cert_blueprints(self) -> list[CertBlueprint]:
        """Return the ordered list of exam domains for this provider's cert(s)."""
        ...


# ---------------------------------------------------------------------------
# GCP implementation (v1 — Professional Cloud Architect only)
# ---------------------------------------------------------------------------


class GCPProvider:
    """Cloud provider adapter for Google Cloud Platform.

    Scopes ingestion to the documentation sections actually tested on the
    Professional Cloud Architect (PCA) exam; everything else in the 180-shard
    sitemap index is intentionally skipped.
    """

    name = "gcp"

    # URL prefixes fed to _build_regex_filters() in ProviderSitemapLoader.
    # Covers every product category weighted in the GCP PCA exam guide.
    url_roots: list[str] = [
        # Architecture & Governance (PCA Core)
        "https://cloud.google.com/architecture",
        "https://cloud.google.com/resource-manager/docs",
        "https://cloud.google.com/billing/docs",
        "https://cloud.google.com/migrate/docs",
        # Compute & Containers (PCA / PMLE)
        "https://cloud.google.com/compute/docs",
        "https://cloud.google.com/kubernetes-engine/docs",
        "https://cloud.google.com/run/docs",
        "https://cloud.google.com/functions/docs",
        "https://cloud.google.com/app-engine/docs",
        # Storage & Databases (PCA / PMLE)
        "https://cloud.google.com/storage/docs",
        "https://cloud.google.com/filestore/docs",
        "https://cloud.google.com/sql/docs",
        "https://cloud.google.com/spanner/docs",
        "https://cloud.google.com/bigtable/docs",
        "https://cloud.google.com/firestore/docs",
        # Data, Analytics & ML (PMLE Core)
        "https://cloud.google.com/bigquery/docs",
        "https://cloud.google.com/pubsub/docs",
        "https://cloud.google.com/dataflow/docs",
        "https://cloud.google.com/dataproc/docs",
        "https://cloud.google.com/dataprep/docs",
        "https://cloud.google.com/vertex-ai/docs",
        "https://cloud.google.com/composer/docs",
        # Networking (PCA Core)
        "https://cloud.google.com/vpc/docs",
        "https://cloud.google.com/load-balancing/docs",
        "https://cloud.google.com/cdn/docs",
        "https://cloud.google.com/dns/docs",
        "https://cloud.google.com/network-connectivity/docs/interconnect",
        "https://cloud.google.com/network-connectivity/docs/vpn",
        "https://cloud.google.com/armor/docs",
        # Security & Identity (PCA / PMLE)
        "https://cloud.google.com/iam/docs",
        "https://cloud.google.com/identity/docs",
        "https://cloud.google.com/kms/docs",
        "https://cloud.google.com/security-command-center/docs",
        "https://cloud.google.com/certificate-authority-service/docs",
        # Operations & Observability (PCA / PMLE)
        "https://cloud.google.com/monitoring/docs",
        "https://cloud.google.com/logging/docs",
        "https://cloud.google.com/error-reporting/docs",
        "https://cloud.google.com/trace/docs",
        "https://cloud.google.com/profiler/docs",
        "https://cloud.google.com/deploy/docs",
    ]

    cert_tags: list[str] = ["gcp", "gcp-pca", "gcp-pmle"]

    @property
    def sitemap_urls(self) -> list[str]:
        """Return the GCP sitemap index; shards are resolved by LangChain."""
        return ["https://cloud.google.com/sitemap.xml"]

    def url_filter(self) -> Callable[[str], bool]:
        """Accept any sitemap URL that originates from cloud.google.com."""

        def _filter(url: str) -> bool:
            return "cloud.google.com" in url

        return _filter

    def cert_blueprints(self, tag : str) -> list[CertBlueprint]:
        """GCP Professional Cloud Architect exam domains.

        Weights and domains sourced from the official Google exam guide (2026 revision).
        https://cloud.google.com/learn/certification/guides/professional-cloud-architect
        """
        
        if tag not in self.cert_tags:
            raise ValueError(f"Invalid cert tag '{tag}' for GCPProvider. Valid tags: {self.cert_tags}")
        
        yaml_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "configs",
                "certs",
                "gcp",
                f"{tag}.yaml",
            )
        )

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return [CertBlueprint(**item) for item in data]
