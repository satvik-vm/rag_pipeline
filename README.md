# Introduction
A basic rag pipeline

# Models
Embeddings => [jinaai/jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2)  
Ranking => [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)  
LLM => [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

# How to Use
1. Create ingest_data fodler to add pdf to it or paste the URLs in ingest.py to ingest data
2. Create an empty directory for data for vector db.
3. Download the models and save at models/.
4. Run ingest.py
5. Edit the query and add yours in query.py
6. Run query.py
