# OpenDeepSearch - Local and Free - leveraging Ollama, SearXNG, Infinity-emb

This project uses `OpenDeepSearch` with Ollama and SearXNG to search, rank and summarize search results. Ollama allows for easy use of various LLMs locally. Custom system prompt was also added to control the presentation of search results (e.g. "conscise", "detailed", "summarized" or any other prompt). Fixed parameter issues. Added logs to see search results.

## Prerequisites
- Git
- Python 3.11 (tested)
- Docker
- Ollama
- Internet connection

## Setup Instructions

1. **Clone this Repository**
   
   ```git clone https://github.com/Bensake/OpenDeepSearch-Bensake.git```
   ``` cd OpenDeepSearch-Bensake```
   
2. **Create and Activate a Virtual Environment**

	```python -m venv venv```
   ```source venv/bin/activate```
   
3. **Install Python Requirements**   

	```pip install -e .```
4. **Install Torch**
   For Nvidia GPUs run:
   	```pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html```    # Check you CUDA version (install if not present), cu121 stands for cuda 12.1
   
5. **Install Docker**	
	
	Download and install Docker from docker.com
	
6. **Run Docker**	

	Start Docker Desktop, make sure it's running.
	
7. **Set Up SearXNG with Docker**

	```docker pull searxng/searxng```
	
8. **Run the SearXNG container**	

	```docker run -d -p 8080:8080 searxng/searxng```
	
9. **Install Ollama**	

	Download and install Ollama from ollama.ai
	
10. **Download Ollama Model**	

	Go to Ollama.ai and download a model that fits into your GPU memory. For example, if your GPU RAM is 8 GB, make sure the model size is under 8 GB.
	
	```ollama pull llama3.1:8b-instruct-q5_K_M```      # Change the model name to your desired model

11. **Set Model Name, System prompt, Max Sources**

	Open search_web.py and set:
	model_name    to the Ollama model you downloaded
	system_prompt    what you want LLM to do with the search results
	max_sources    how many search sources to analyze
	
## Usage	

Once you adjust search_web.py based on your needs, you can use command prompt to execute searches:

	 python search_web.py "your search query here"
	
## Notes

Ensure Docker and Ollama are running before executing the script.

	
