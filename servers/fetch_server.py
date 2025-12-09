#!/usr/bin/env python3
"""
Simple Fetch Server - httpx 0.28+ compatible replacement for mcp-server-fetch

Fetches web pages and converts HTML to markdown.
"""

import httpx
from mcp.server.fastmcp import FastMCP
from markdownify import markdownify as md
from bs4 import BeautifulSoup

mcp = FastMCP("fetch")


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
    try:
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
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if "text/html" in content_type:
                # Parse HTML and convert to markdown
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

                return markdown

            elif "application/json" in content_type:
                return response.text[:max_length]

            else:
                # Plain text or other
                return response.text[:max_length]

    except httpx.HTTPStatusError as e:
        return f"HTTP Error {e.response.status_code}: {e.response.reason_phrase}"
    except httpx.RequestError as e:
        return f"Request Error: {str(e)}"
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


if __name__ == "__main__":
    mcp.run()
