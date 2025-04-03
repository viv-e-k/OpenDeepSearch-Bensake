# process_sources_pro.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from opendeepsearch.context_scraping.crawl4ai_scraper import WebScraper
from opendeepsearch.ranking_models.infinity_rerank import InfinitySemanticSearcher
from opendeepsearch.ranking_models.jina_reranker import JinaReranker
from opendeepsearch.ranking_models.chunker import Chunker 
from opendeepsearch.serp_search.serp_search import SearchResult

@dataclass
class Source:
    link: str
    html: str = ""

class SourceProcessor:
    def __init__(
        self, 
        top_results: int = 5,
        strategies: List[str] = ["no_extraction"],
        filter_content: bool = True,
        reranker: str = "infinity"
    ):
        self.strategies = strategies
        self.filter_content = filter_content
        self.scraper = WebScraper(
            strategies=self.strategies, 
            filter_content=self.filter_content
        )
        self.top_results = top_results
        self.chunker = Chunker()
        if reranker.lower() == "jina":
            self.semantic_searcher = JinaReranker()
            print("Using Jina Reranker")
        else:
            self.semantic_searcher = InfinitySemanticSearcher()
            print("Using Infinity Reranker")

    async def process_sources(
        self, 
        sources: Union[SearchResult, List[dict]], 
        num_elements: int, 
        query: str, 
        pro_mode: bool = False
    ) -> dict:
        try:
            if isinstance(sources, SearchResult):
                if not sources.success or sources.data is None:
                    print(f"Search failed in process_sources: {sources.error or 'No data'}")
                    return {'organic': []}
                source_data = sources.data
                print("Raw search results from SearXNG:")
                for i, result in enumerate(source_data.get('organic', [])[:num_elements], 1):
                    print(f"{i}. {result.get('title', 'No title')} - {result.get('link', 'No link')}")
            else:
                source_data = sources

            valid_sources = self._get_valid_sources(source_data, num_elements)
            if not valid_sources:
                return {'organic': source_data.get('organic', []) if isinstance(source_data, dict) else source_data}

            if not pro_mode:
                wiki_sources = [(i, source) for i, source in valid_sources 
                              if 'wikipedia.org' in source['link']]
                if not wiki_sources:
                    return {'organic': source_data.get('organic', [])[:num_elements] if isinstance(source_data, dict) else source_data}
                valid_sources = wiki_sources[:1]

            html_contents = await self._fetch_html_contents([s[1]['link'] for s in valid_sources])
            processed_sources = self._update_sources_with_content(
                source_data.get('organic', []) if isinstance(source_data, dict) else source_data,
                valid_sources, html_contents, query
            )
            print("Reranked sources:")
            for i, source in enumerate(processed_sources[:num_elements], 1):
                print(f"{i}. {source.get('title', 'No title')} - {source.get('link', 'No link')}")
            return {'organic': processed_sources}
        except Exception as e:
            print(f"Error in process_sources: {e}")
            return {'organic': []}

    def _get_valid_sources(self, sources: Union[dict, List[dict]], num_elements: int) -> List[Tuple[int, dict]]:
        if isinstance(sources, dict):
            source_list = sources.get('organic', [])
        else:
            source_list = sources if sources is not None else []
        
        if source_list is None:
            return []
        
        return [(i, source) for i, source in enumerate(source_list[:num_elements]) if source]

    async def _fetch_html_contents(self, links: List[str]) -> List[str]:
        raw_contents = await self.scraper.scrape_many(links)
        return [x['no_extraction'].content for x in raw_contents.values()]

    def _process_html_content(self, html: str, query: str) -> str:
        if not html:
            return ""
        try:
            documents = self.chunker.split_text(html)
            reranked_content = self.semantic_searcher.get_reranked_documents(
                query,
                documents,
                top_k=self.top_results
            )
            return reranked_content
        except Exception as e:
            print(f"Error in content processing: {e}")
            return ""

    def _update_sources_with_content(
        self, 
        sources: List[dict],
        valid_sources: List[Tuple[int, dict]], 
        html_contents: List[str],
        query: str
    ) -> List[dict]:
        for (i, source), html in zip(valid_sources, html_contents):
            source['html'] = self._process_html_content(html, query)
            sources[i] = source
        return sources