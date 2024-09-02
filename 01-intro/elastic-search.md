# Elastic Search

This document contains useful things about Elastic Search

## ```bash multi_match``` Query in Elasticsearch
This ```bash multi_match``` query is used to search for a given text across mutiple fields in an Elasticsearch index.

It provides various types of control how the matching is executed and scored.

There are multiple types of ```bash multi_match``` queries:
- ```bash best_fields```: Returns the highest score from any one field
- ```bash most_fields```: Combines the scores from all fields
- ```bash cross_fields```: Treats fields as one big field for scoring
- ```bash phrase```: Searches for the query as an exact phrase.
- ```bash phrase_prefix```: Search for the query as a prefix of a phrase.


### `best_fields`
The ```bash best_fields``` type searches each field separately and returns the highest score from any one of the fields.

This type is useful when you want to find documents where at least one field matches the query well.

```python
{
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": "How do I run docker on Windows?",
                    "fields": ["question", "text"],
                    "type": "best_fields"
                }
            }
        }
    }
}
           
```

### `most_fields`
The ```bash most_fields``` type searches each field and combines the scores from all fields.

This type is useful when the relevance of a document increases with more matching fields.

```python
{
    "multi_match": {
                    "query": "How do I run docker on Windows?",
                    "fields": ["question^4", "text"],
                    "type": "most_fields"
                }
}          
```

### `cross_fields`
The ```bash cross_fields``` type treats fields as though they were one big field.

It is suitable for cases where you have fields representing the same text in differnt ways, such as synonyms.
```python
{
    "multi_match": {
                    "query": "How do I run docker on Windows?",
                    "fields": ["question", "text"],
                    "type": "cross_fields"
                }
}          
```
### `phrase`
The ```bash phrase``` type looks for the query as an exact phrase within the fields.

It is useful for exact searches.
```python
{
    "multi_match": {
                    "query": "How do I run docker on Windows?",
                    "fields": ["question", "text"],
                    "type": "phrase"
                }
}          
```


### `phrase_prefix`
The ```bash phrase_prefix``` type searches the documents that contain the query as a prefix of a phrase.

This is useful for autocomplete or typehead functionality
```python
{
    "multi_match": {
                    "query": "How do I run docker on Windows?",
                    "fields": ["question", "text"],
                    "type": "phrase_prefix"
                }
}          
```