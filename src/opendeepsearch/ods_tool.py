# ods_tool.py
from typing import Optional, Literal
from smolagents import Tool
from opendeepsearch.ods_agent import OpenDeepSearchAgent

class OpenDeepSearchTool(Tool):
    name = "web_search"
    description = """
    Performs web search based on your query (think a Google search) then returns the final answer that is processed by an llm."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform",
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_name: Optional[str] = None,
        reranker: str = "infinity",
        search_provider: Literal["serper", "searxng"] = "searxng",
        serper_api_key: Optional[str] = None,
        searxng_instance_url: Optional[str] = "http://localhost:8080/",
        searxng_api_key: Optional[str] = None,
        source_processor_config: Optional[dict] = None,
        max_sources: int = 2,
        pro_mode: bool = True,
        system_prompt: Optional[str] = None  # Add system_prompt parameter
    ):
        super().__init__()
        self.search_model_name = model_name
        self.reranker = reranker
        self.search_provider = search_provider
        self.serper_api_key = serper_api_key
        self.searxng_instance_url = searxng_instance_url
        self.searxng_api_key = searxng_api_key
        self.source_processor_config = source_processor_config
        self.max_sources = max_sources
        self.pro_mode = pro_mode
        self.system_prompt = system_prompt  # Store it

    def forward(self, query: str) -> tuple[str, str]:
        answer, context = self.search_tool.ask_sync(
            query,
            max_sources=self.max_sources,
            pro_mode=self.pro_mode
        )
        return answer, context

    def setup(self):
        self.search_tool = OpenDeepSearchAgent(
            model=self.search_model_name,
            reranker=self.reranker,
            search_provider=self.search_provider,
            serper_api_key=self.serper_api_key,
            searxng_instance_url=self.searxng_instance_url,
            searxng_api_key=self.searxng_api_key,
            source_processor_config=self.source_processor_config,
            system_prompt=self.system_prompt  # Pass it to OpenDeepSearchAgent
        )