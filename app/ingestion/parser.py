"""
HTML parser for cloud.google.com DevSite documentation pages.

The public entry point is `parse_gcp_doc`, which matches the signature expected
by LangChain's SitemapLoader.parsing_function: (BeautifulSoup) -> str.

It returns **clean HTML** (not plain text) so the downstream
HTMLHeaderTextSplitter can split on <h1>–<h4> tags.
"""

from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Noise elements to strip from the DevSite page before returning content.
# These are chrome — navigation, feedback widgets, copy-code buttons, etc.
# ---------------------------------------------------------------------------

_NOISE_SELECTORS: list[str] = [
    # Structural chrome
    "header",
    "footer",
    "nav",
    "aside",
    # Scripts / styles / media
    "script",
    "style",
    "noscript",
    "img",  # Images carry no text signal for RAG
    # DevSite-specific widgets
    ".devsite-toc",               # Right-side table of contents
    ".devsite-nav",               # Left navigation tree
    ".devsite-breadcrumb",        # Breadcrumb trail at top
    ".devsite-feedback",          # "Was this helpful?" widget
    ".devsite-rating",            # Star rating widget
    ".devsite-page-nav",          # Prev / Next page footer nav
    ".devsite-header",            # Top bar
    ".devsite-article-heading",   # Duplicates <h1> inside metadata bar
    ".devsite-banner",            # Announcement banners
    # Interactive elements with no content value
    "button",
    "[role='navigation']",
    "[role='banner']",
    "[role='complementary']",     # Sidebars
    # Copy-code overlay buttons injected by DevSite JS
    ".devsite-code-copy-button",
    ".devsite-copy-code-button",
    "devsite-copy-code-button",   # Custom element variant
    # Cookie / consent banners
    "#cookie-banner",
    ".cookie-notice",
]


def parse_gcp_doc(soup: BeautifulSoup) -> str:
    """Parse a cloud.google.com DevSite page and return clean HTML.

    Signature matches ``SitemapLoader(parsing_function=parse_gcp_doc)``.

    Strategy
    --------
    1. Locate the main article container using DevSite's known selectors,
       falling back progressively to ``<main>`` then ``<body>``.
    2. Decompose every noise element in-place (nav, TOC, feedback, etc.).
    3. Return the cleaned inner HTML so ``HTMLHeaderTextSplitter`` can
       split on preserved ``<h1>``–``<h4>`` tags downstream.

    Returns an empty string when no content container is found (e.g. error
    pages, redirects) so callers can skip the document silently.
    """
    # --- 1. Find the main content container -----------------------------------
    # DevSite renders documentation inside <article class="devsite-article">.
    # The actual prose lives in a child div.devsite-article-body.
    # We prefer the narrowest container that still has all the content.
    content: Tag | None = (
        soup.find("div", class_="devsite-article-body")
        or soup.find("article", class_=lambda c: c and "devsite-article" in c)
        or soup.find("main")
        or soup.body
    )

    if content is None:
        return ""

    # Work on a copy so we don't mutate the caller's tree
    content = content.__copy__()

    # --- 2. Strip noise -------------------------------------------------------
    for selector in _NOISE_SELECTORS:
        for el in content.select(selector):
            el.decompose()

    # --- 3. Return clean HTML -------------------------------------------------
    return str(content)
