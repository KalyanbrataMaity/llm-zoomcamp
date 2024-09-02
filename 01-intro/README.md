Module 1: Introduction


In this module, we will learn what LLM and RAG are and implement a simple RAG pipeline to answer questions about the FAQ
documents from our Zoomcamp courses

Wha we will do:

- Index zoomcamp FAQ documents
    * DE Zoomcamp: https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit
    * ML Zoomcamp: https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit
    * MLOps Zoomcamp: https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit

- Create a Q&A system for answering questions about these documents


1.1 Introduction to LLM and RAG
- LLM
- RAG
- RAG architecture
- Course outcome

1.2 Preparing the Environment 
- installing libraries 
    ```bash
    pip install tqdm notebook==7.1.2 openai elasticsearch pandas scikit-learn ipywidgets
    ```
- Alternatives: installing anaconda or miniconda

1.3 Retrieval & Search
This part is for retrieval and search
- We will use the search engine that we built in the workshop: built your own search engine
- indexing the documents
- Performing the search

1.4 Genration with Open AI
- Invoking OpenAI api
- Building the prompt
- Getting the answer

    * 1.4.2 Open AI alternatives: Exploring OpenAI alternatives.

1.5 The RAG Flow of Cleaning and Modularizing code:
- cleaning the code we wrote so far
- Making it modular

1.6 Searching with Elastic Search
Replacing the toy search engine with Elastic Search
- Run ElasticSearch with Docker
-
-

### Running the Elastic Search Docker engine:
```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

If the previous command does not work (i.e. you see "error pulling image configuration"), try to run ElasticSearch directly from Docker Hub:
```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    elasticsearch:8.4.3
```

### Index Settings:
```python
{
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course":{"type": "keyword"}
        }
    }
}
```

### Query
```python
{
    "size": 5,
    "query": {
        "bool":{
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^3", "text", "section"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
}
```

We use "type": "best_fields". There are different types of multi_match search. Find here: elastic_search.md


