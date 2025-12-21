#!/usr/bin/env python3
"""
Simple Fetch Server - httpx 0.28+ compatible replacement for mcp-server-fetch

Fetches web pages and converts HTML to markdown.
Handles Semantic Scholar URLs by converting them to API calls.
"""

import re
import json
import httpx
from fastmcp import FastMCP
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from timing import ToolTimer, measure_io_async

mcp = FastMCP("fetch")


def _extract_s2_paper_id(url: str) -> str | None:
    """Extract Semantic Scholar paper ID from URL.

    Supports:
    - https://www.semanticscholar.org/paper/TITLE/PAPER_ID
    - https://www.semanticscholar.org/paper/PAPER_ID
    """
    # Pattern: /paper/optional-title/40-char-hex-id or just /paper/40-char-hex-id
    match = re.search(r'/paper/(?:[^/]+/)?([a-f0-9]{40})/?$', url, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


async def _fetch_s2_paper_via_api(paper_id: str, max_length: int) -> str:
    """Fetch Semantic Scholar paper details via their REST API.

    API docs: https://api.semanticscholar.org/api-docs/
    Includes retry with exponential backoff for rate limiting (429).
    """
    import asyncio

    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    fields = "title,abstract,year,authors,venue,citationCount,influentialCitationCount,tldr"

    max_retries = 3
    retry_delay = 1.0  # Start with 1 second

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(max_retries + 1):
            response = await client.get(
                api_url,
                params={"fields": fields},
                headers={"Accept": "application/json"}
            )

            if response.status_code == 429:
                # Rate limited - retry with backoff
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise httpx.HTTPStatusError(
                        "Rate limited (429). Max retries exceeded.",
                        request=response.request,
                        response=response
                    )

            response.raise_for_status()
            break

        data = response.json()

    # Format as readable markdown
    lines = []
    lines.append(f"# {data.get('title', 'Unknown Title')}")
    lines.append("")

    if data.get('year'):
        lines.append(f"**Year:** {data['year']}")

    if data.get('venue'):
        lines.append(f"**Venue:** {data['venue']}")

    if data.get('authors'):
        author_names = [a.get('name', '') for a in data['authors'][:10]]
        lines.append(f"**Authors:** {', '.join(author_names)}")
        if len(data['authors']) > 10:
            lines.append(f"  _(and {len(data['authors']) - 10} more)_")

    if data.get('citationCount'):
        lines.append(f"**Citations:** {data['citationCount']}")

    lines.append("")

    # TL;DR summary if available
    if data.get('tldr') and data['tldr'].get('text'):
        lines.append("## TL;DR")
        lines.append(data['tldr']['text'])
        lines.append("")

    # Abstract
    if data.get('abstract'):
        lines.append("## Abstract")
        lines.append(data['abstract'])

    result = "\n".join(lines)
    return result[:max_length]


@mcp.tool()
async def fetch(url: str, max_length: int = 50000) -> str:
    """
    Fetch a URL and return its content as markdown.

    Args:
        url: The URL to fetch
        max_length: Maximum content length to return (default 50000)

    Returns:
        The page content converted to markdown
    """
    timer = ToolTimer("fetch")
    try:
        # Check if this is a Semantic Scholar paper URL - use API instead
        s2_paper_id = _extract_s2_paper_id(url)
        if s2_paper_id:
            result = await _fetch_s2_paper_via_api(s2_paper_id, max_length)
            timer.finish()
            return result

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            }
        ) as client:
            # Network I/O - measure it
            response = await measure_io_async(lambda: client.get(url))

            # Handle HTTP 202 (Accepted but not ready) - common with Semantic Scholar
            if response.status_code == 202:
                timer.finish()
                return f"Error: HTTP 202 Accepted - Content not ready. This URL ({url}) requires server-side processing. Try a different URL or wait and retry later."

            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if "text/html" in content_type:
                # Parse HTML and convert to markdown (compute, not I/O)
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script and style elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()

                # Get main content if available
                main = soup.find("main") or soup.find("article") or soup.find("body")
                if main:
                    html_content = str(main)
                else:
                    html_content = str(soup)

                # Convert to markdown
                markdown = md(html_content, heading_style="ATX", strip=["a"])

                # Truncate if too long
                if len(markdown) > max_length:
                    markdown = markdown[:max_length] + "\n\n[Content truncated...]"

                timer.finish()
                return markdown

            elif "application/json" in content_type:
                timer.finish()
                return response.text[:max_length]

            else:
                # Plain text or other
                timer.finish()
                return response.text[:max_length]

    except httpx.HTTPStatusError as e:
        timer.finish()
        return f"HTTP Error {e.response.status_code}: {e.response.reason_phrase}"
    except httpx.RequestError as e:
        timer.finish()
        return f"Request Error: {str(e)}"
    except Exception as e:
        timer.finish()
        return f"Error fetching URL: {str(e)}"


if __name__ == "__main__":
    import os

    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8080"))
        mcp.run(transport="http", host=host, port=port, path="/mcp")
    else:
        mcp.run()
