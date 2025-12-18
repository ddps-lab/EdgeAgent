"""LLM-based classifier for MCP servers."""

import json
import re
from typing import List, Optional, Dict, Any

from .base import BaseClassifier
from ..models import MCPServer, ClassificationResult
from ..taxonomy import get_taxonomy_description, CATEGORY_TAXONOMY
from ..config import LLM_CONFIG, BATCH_SIZE


class LLMClassifier(BaseClassifier):
    """Classifier that uses LLM to categorize MCP servers."""

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.config = LLM_CONFIG.get(provider, LLM_CONFIG["openai"])
        self.client = self._init_client()
        self._taxonomy_desc = get_taxonomy_description()

    @property
    def name(self) -> str:
        return f"llm_{self.provider}"

    def _init_client(self):
        """Initialize the LLM client based on provider."""
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI()
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic()
        elif self.provider == "google":
            import google.generativeai as genai
            from ..config import GOOGLE_API_KEY
            genai.configure(api_key=GOOGLE_API_KEY)
            return genai
        elif self.provider == "upstage":
            from openai import OpenAI
            from ..config import UPSTAGE_API_KEY
            return OpenAI(
                api_key=UPSTAGE_API_KEY,
                base_url="https://api.upstage.ai/v1/solar"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _build_prompt(self, server: MCPServer) -> str:
        """Build classification prompt for a single server."""
        tools_summary = server.get_tools_summary(max_tools=15)

        prompt = f"""You are an expert at categorizing MCP (Model Context Protocol) servers.

Given the following MCP server information, classify it into the most appropriate category.

## Server Information
- Server ID: {server.server_id}
- Server Name: {server.server_name}
- Number of Tools: {server.tools_count}

## Tools:
{tools_summary}

## {self._taxonomy_desc}

## Instructions
1. Analyze the server name, ID, and tool descriptions carefully
2. Select the most appropriate major category from the 14 options
3. Select the most specific subcategory within that major category
4. You MUST choose from the categories listed above - do NOT create new categories
5. Provide a confidence score (0.0-1.0)
6. Explain your reasoning briefly (1-2 sentences)

## Output Format (JSON only, no markdown)
{{"category_major": "...", "category_minor": "...", "confidence": 0.XX, "reasoning": "..."}}
"""
        return prompt

    def _build_batch_prompt(self, servers: List[MCPServer]) -> str:
        """Build classification prompt for multiple servers."""
        servers_info = []
        for i, server in enumerate(servers):
            tools_summary = server.get_tools_summary(max_tools=5)
            servers_info.append(
                f"[{i+1}] ID: {server.server_id}, Name: {server.server_name}, "
                f"Tools ({server.tools_count}):\n{tools_summary}"
            )

        prompt = f"""You are an expert at categorizing MCP (Model Context Protocol) servers.

Classify the following {len(servers)} MCP servers into appropriate categories.

## Servers to Classify:
{chr(10).join(servers_info)}

## {self._taxonomy_desc}

## Instructions
- For each server, select the most appropriate major and minor category
- You MUST choose from the categories listed above only
- Provide confidence scores (0.0-1.0)
- Output ONLY a JSON array, no explanation outside JSON

## Output Format (JSON array only, no markdown):
[
  {{"server_id": "...", "category_major": "...", "category_minor": "...", "confidence": 0.XX}},
  ...
]
"""
        return prompt

    def _call_api(self, prompt: str, batch: bool = False) -> str:
        """Call the LLM API."""
        if self.provider == "openai" or self.provider == "upstage":
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"] * (5 if batch else 1),
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.config["model"],
                max_tokens=self.config["max_tokens"] * (5 if batch else 1),
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif self.provider == "google":
            model = self.client.GenerativeModel(self.config["model"])
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config["temperature"],
                    "max_output_tokens": self.config["max_tokens"] * (5 if batch else 1),
                }
            )
            return response.text

        raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse single classification response."""
        try:
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)

            data = json.loads(response)

            # Validate category
            major = data.get("category_major", "")
            minor = data.get("category_minor", "")

            if major not in CATEGORY_TAXONOMY:
                # Try to find closest match
                major = self._find_closest_category(major)

            if major and minor not in CATEGORY_TAXONOMY.get(major, []):
                # Use first subcategory as fallback
                minor = CATEGORY_TAXONOMY.get(major, ["Unknown"])[0]

            return {
                "category_major": major,
                "category_minor": minor,
                "confidence": float(data.get("confidence", 0.7)),
                "reasoning": data.get("reasoning", ""),
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing response: {e}")
            print(f"Response was: {response[:500]}")
            return None

    def _parse_batch_response(
        self, response: str, servers: List[MCPServer]
    ) -> List[Dict[str, Any]]:
        """Parse batch classification response."""
        try:
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)

            data = json.loads(response)

            if not isinstance(data, list):
                data = [data]

            results = []
            for item in data:
                major = item.get("category_major", "")
                minor = item.get("category_minor", "")

                if major not in CATEGORY_TAXONOMY:
                    major = self._find_closest_category(major)

                if major and minor not in CATEGORY_TAXONOMY.get(major, []):
                    minor = CATEGORY_TAXONOMY.get(major, ["Unknown"])[0]

                results.append({
                    "server_id": item.get("server_id", ""),
                    "category_major": major,
                    "category_minor": minor,
                    "confidence": float(item.get("confidence", 0.7)),
                })

            return results
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing batch response: {e}")
            print(f"Response was: {response[:1000]}")
            return []

    def _find_closest_category(self, category: str) -> str:
        """Find the closest matching category."""
        category_lower = category.lower()
        for cat in CATEGORY_TAXONOMY.keys():
            if cat.lower() in category_lower or category_lower in cat.lower():
                return cat
        # Default to Specialized if no match
        return "Specialized"

    def classify(self, server: MCPServer) -> Optional[ClassificationResult]:
        """Classify a single server using LLM."""
        prompt = self._build_prompt(server)
        response = self._call_api(prompt)
        parsed = self._parse_response(response)

        if not parsed:
            return None

        return ClassificationResult(
            server_id=server.server_id,
            server_name=server.server_name,
            category_major=parsed["category_major"],
            category_minor=parsed["category_minor"],
            tools_count=server.tools_count,
            confidence=parsed["confidence"],
            method=self.name,
            reasoning=parsed.get("reasoning"),
            source=server.source,
        )

    def classify_batch(
        self, servers: List[MCPServer], batch_size: int = BATCH_SIZE
    ) -> List[ClassificationResult]:
        """Classify multiple servers using LLM in batches."""
        results = []

        for i in range(0, len(servers), batch_size):
            batch = servers[i:i + batch_size]
            prompt = self._build_batch_prompt(batch)

            try:
                response = self._call_api(prompt, batch=True)
                parsed_results = self._parse_batch_response(response, batch)

                # Match results to servers
                server_id_map = {s.server_id: s for s in batch}

                for parsed in parsed_results:
                    server_id = parsed.get("server_id", "")
                    server = server_id_map.get(server_id)

                    if server:
                        results.append(ClassificationResult(
                            server_id=server.server_id,
                            server_name=server.server_name,
                            category_major=parsed["category_major"],
                            category_minor=parsed["category_minor"],
                            tools_count=server.tools_count,
                            confidence=parsed["confidence"],
                            method=self.name,
                            source=server.source,
                        ))

                # Handle servers that weren't matched
                matched_ids = {r.server_id for r in results[-len(parsed_results):]}
                for server in batch:
                    if server.server_id not in matched_ids:
                        # Classify individually as fallback
                        result = self.classify(server)
                        if result:
                            results.append(result)

            except Exception as e:
                print(f"Error in batch classification: {e}")
                # Fall back to individual classification
                for server in batch:
                    try:
                        result = self.classify(server)
                        if result:
                            results.append(result)
                    except Exception as e2:
                        print(f"Error classifying {server.server_id}: {e2}")

        return results

    def classify_all(
        self, servers: List[MCPServer], show_progress: bool = True
    ) -> List[ClassificationResult]:
        """Classify all servers, showing progress."""
        if show_progress:
            try:
                from tqdm import tqdm
                results = []
                for i in tqdm(range(0, len(servers), BATCH_SIZE), desc=f"Classifying with {self.provider}"):
                    batch = servers[i:i + BATCH_SIZE]
                    batch_results = self.classify_batch(batch, batch_size=len(batch))
                    results.extend(batch_results)
                return results
            except ImportError:
                pass

        return self.classify_batch(servers)
