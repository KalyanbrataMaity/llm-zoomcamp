# minsearch

Minimalistic text search engine that uses sklearn and pandas

This is a simple search library implemented using ```bash sklearn``` and ```pandas``

It allows you to index documents with text and keyword fields and perform search queries with support for filtering and boosting.

### Installation

Make sure you have the required dependencies installed:
```bash pip install pandas scikit-learn```

### Usage

Here is how you can use the library:

#### Define your documents
Prepare your documents as a list of dictionaries. Each dictionary should have the text and keyword fields
you want to index.

```bash
docs = [
    {
        "question": "How do I join the course after it has started?",
        "text": "You can join the course at any time. We have recordings available.",
        "section": "General Information",
        "course": "data-engineering-zoomcamp"
    },
    {
        "question": "What are the prerequisites for the course?",
        "text": "You need to have basic knowledge of programming.",
        "section": "Course Requirements",
        "course": "data-engineering-zoomcamp"
    }
]
```

### Create the index
Create the instance of the index class, specifying the text and keyword fields.

```bash
from minsearch import Index

index = Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)
```

Fit the index with your documents
```bash index.fit(docs) ```

### Perform a search
Search the index with a query string, optional filter dictionary, and optional boost dictionary
```bash
query = "Can I join the course if it has already started?"

filter_dict = {"course": "data-engineering-zoomcamp"}
boost_dict = {"question": 3, "text": 1, "section": 1}

results = index.search(query, filter_dict, boost_dict)

for result in results:
    print(result)
```

