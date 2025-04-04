import os
import sys
import argparse
from opendeepsearch import OpenDeepSearchTool

# Set environment variables for Ollama
os.environ["LITELLM_MODEL_ID"] = "ollama/llama3.1:8b-instruct-q5_K_M"
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

# Initialize OpenDeepSearchTool with pro mode and reranking
search_agent = OpenDeepSearchTool(
    model_name="ollama/llama3.1:8b-instruct-q5_K_M",
    search_provider="searxng",
    searxng_instance_url="http://localhost:8080/",
    reranker="infinity",
    source_processor_config={"top_results": 5},
    max_sources=3,
    pro_mode=True,
    system_prompt="Provide a detailed summary of the context, including examples and explanations."
)

# Call setup to initialize self.search_tool
search_agent.setup()

def search_query(query):
    print(f"Searching for: {query}")
    answer, context = search_agent.forward(query)
    print("\nRaw Context Passed to LLM:")
    print(context)
    print("\nSearch Results (Final Answer):")
    print(answer)
    return answer, context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search and summarize with OpenDeepSearchTool")
    parser.add_argument("query", type=str, help="The search query to process")
    args = parser.parse_args()
    
    search_query(args.query)
