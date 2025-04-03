# ods_agent.py
from typing import Optional, Dict, Any, Literal
from opendeepsearch.serp_search.serp_search import create_search_api, SearchAPI
from opendeepsearch.context_building.process_sources_pro import SourceProcessor
from opendeepsearch.context_building.build_context import build_context
from litellm import completion, utils
from dotenv import load_dotenv
import os
from opendeepsearch.prompts import SEARCH_SYSTEM_PROMPT
import asyncio
import nest_asyncio
load_dotenv()

class OpenDeepSearchAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        search_provider: Literal["serper", "searxng"] = "serper",
        serper_api_key: Optional[str] = None,
        searxng_instance_url: Optional[str] = None,
        searxng_api_key: Optional[str] = None,
        source_processor_config: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        top_p: float = 0.3,
        reranker: Optional[str] = "None",
    ):
        self.serp_search = create_search_api(
            search_provider=search_provider,
            serper_api_key=serper_api_key,
            searxng_instance_url=searxng_instance_url,
            searxng_api_key=searxng_api_key
        )
        if source_processor_config is None:
            source_processor_config = {}
        if reranker:
            source_processor_config['reranker'] = reranker
        self.source_processor = SourceProcessor(**source_processor_config)
        self.model = model if model is not None else os.getenv("LITELLM_SEARCH_MODEL_ID", os.getenv("LITELLM_MODEL_ID", "ollama/gemma3:27b"))
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt or """
        You are a helpful AI assistant tasked with providing detailed, in-depth summaries based on the provided context. Use the search results to give a comprehensive answer, including specific examples, explanations, and details from the sources. Aim for a thorough response that fully addresses the query, summarizing key points and elaborating where relevant.
        """
        openai_base_url = os.environ.get("OPENAI_BASE_URL")
        if openai_base_url:
            utils.set_provider_config("openai", {"base_url": openai_base_url})

    async def search_and_build_context(
        self,
        query: str,
        max_sources: int = 2,
        pro_mode: bool = False
    ) -> str:
        sources = self.serp_search.get_sources(query, max_sources)
        if not sources.success or sources.data is None:
            print(f"Search failed: {sources.error or 'No data returned'}")
            return build_context({'organic': []})  # Fallback to empty results
        processed_sources = await self.source_processor.process_sources(
            sources,
            max_sources,
            query,
            pro_mode
        )
        return build_context(processed_sources)

    async def ask(
        self,
        query: str,
        max_sources: int = 2,
        pro_mode: bool = False,
    ) -> str:
        context = await self.search_and_build_context(query, max_sources, pro_mode)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p
        )
        return response.choices[0].message.content

    def ask_sync(
        self,
        query: str,
        max_sources: int = 2,
        pro_mode: bool = False,
    ) -> tuple[str, str]:  # Return tuple of answer and context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        context = loop.run_until_complete(self.search_and_build_context(query, max_sources, pro_mode))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p
        )
        answer = response.choices[0].message.content
        return answer, context  # Return both