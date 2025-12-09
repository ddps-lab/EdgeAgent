#!/usr/bin/env python3
"""
Summarize MCP Server

Provides text summarization using OpenAI (default) or Upstage API.

Environment Variables:
    SUMMARIZE_PROVIDER: "openai" (default) or "upstage"
    OPENAI_API_KEY: Required for OpenAI provider
    UPSTAGE_API_KEY: Required for Upstage provider

Usage:
    SUMMARIZE_PROVIDER=openai python servers/summarize_server.py
    SUMMARIZE_PROVIDER=upstage python servers/summarize_server.py
"""

import os
from pathlib import Path
from typing import Literal
from fastmcp import FastMCP

# Load .env file from project root
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed

mcp = FastMCP("summarize")

# Provider selection
PROVIDER = os.getenv("SUMMARIZE_PROVIDER", "openai").lower()


def _summarize_openai(text: str, max_length: int, style: str) -> str:
    """Summarize using OpenAI API"""
    import openai

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    style_prompts = {
        "concise": "Provide a brief, concise summary.",
        "detailed": "Provide a detailed summary with key points.",
        "bullet": "Summarize as bullet points.",
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"You are a text summarization assistant. {style_prompts.get(style, style_prompts['concise'])} Keep the summary under {max_length} words.",
            },
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"},
        ],
        max_tokens=max_length * 2,  # Approximate token count
    )
    return response.choices[0].message.content


def _summarize_upstage(text: str, max_length: int, style: str) -> str:
    """Summarize using Upstage Solar API"""
    import httpx

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY environment variable is required")

    style_prompts = {
        "concise": "Provide a brief, concise summary.",
        "detailed": "Provide a detailed summary with key points.",
        "bullet": "Summarize as bullet points.",
    }

    # Use Upstage Solar API (chat completions compatible)
    response = httpx.post(
        "https://api.upstage.ai/v1/solar/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "solar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a text summarization assistant. {style_prompts.get(style, style_prompts['concise'])} Keep the summary under {max_length} words.",
                },
                {
                    "role": "user",
                    "content": f"Summarize the following text:\n\n{text}",
                },
            ],
            "max_tokens": max_length * 2,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


@mcp.tool()
def summarize_text(
    text: str,
    max_length: int = 150,
    style: Literal["concise", "detailed", "bullet"] = "concise",
) -> str:
    """
    Summarize the given text.

    Args:
        text: The text to summarize
        max_length: Maximum length of summary in words (default: 150)
        style: Summary style - "concise", "detailed", or "bullet"

    Returns:
        Summarized text
    """
    if not text or not text.strip():
        return "Error: Empty text provided"

    if len(text) < 100:
        return text  # Text too short to summarize

    try:
        if PROVIDER == "upstage":
            return _summarize_upstage(text, max_length, style)
        else:
            return _summarize_openai(text, max_length, style)
    except Exception as e:
        return f"Error summarizing text: {str(e)}"


@mcp.tool()
def summarize_documents(
    documents: list[str],
    max_length_per_doc: int = 100,
    style: Literal["concise", "detailed", "bullet"] = "concise",
) -> list[str]:
    """
    Summarize multiple documents.

    Args:
        documents: List of documents to summarize
        max_length_per_doc: Maximum length per summary in words
        style: Summary style for all documents

    Returns:
        List of summarized documents
    """
    return [summarize_text(doc, max_length_per_doc, style) for doc in documents]


@mcp.tool()
def get_provider_info() -> dict:
    """
    Get information about the current summarization provider.

    Returns:
        Dictionary with provider information
    """
    return {
        "provider": PROVIDER,
        "available_styles": ["concise", "detailed", "bullet"],
        "default_max_length": 150,
    }


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport == "http":
        # Streamable HTTP for serverless/remote deployment
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8003"))
        mcp.run(transport="http", host=host, port=port, path="/mcp")
    else:
        # stdio for local development / Claude Desktop
        mcp.run()
