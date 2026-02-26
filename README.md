# 🚀 rag-toolkit
A library designed to evaluate Retrieval-Augmented Generation (RAG) pipelines. 

## 🛠 Installation & Setup

### 1. Install the Python Library
```
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Start Local Search Infrastructure
`rag-toolkit` leverages Elasticsearch for high-performance indexing and retrieval testing. Run this script to spin up a local instance:
```
bash elastic-start-local/start.sh
```
Once the script finishes, it will generate an .env file containing your ES_LOCAL_URL and ES_LOCAL_API_KEY. rag-toolkit will automatically look for these variables to connect to your cluster.