import pandas as pd 
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Index:
    """
    A simple search index using TF-IDF and cosine similarity for text fields and exact matching
    for keyword fields.

    Attributes:

    """
    def __init__(self, text_fields, keyword_fields, vectorizer_params={}):
        """
        Initialize the search index with specified text and keyword fields.

        Args:
            text_fields (List[str]): List of text fields to index.
            keyword_fields (List[str]): List of keyword fields to index.
            vectorizer_params (Dict, optional): Parameters for the vectorizer. Defaults to {}.
        """
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields

        self.vectorizers = {field: TfidfVectorizer(**vectorizer_params) for field in self.text_fields}
        self.text_matrices = {}

    def fit(self, docs):
        """
        Fits the index with provided documents
        
        Args:
            docs (List[Dict]): List of documents with text and keyword fields. Each document should be a dictionary.
        """
        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}

        for field in self.text_fields:
            texts = [doc.get(field, '') for doc in docs]
            self.text_matrices[field] = self.vectorizers[field].fit_transform(texts)

        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        self.keyword_df = pd.DataFrame(keyword_data)

        return self
    
    def search(self, query, filter_dict={}, boost_dict={}, num_results=10):
        """
        Searches the index with given query, filters, and boost parameters
        
        Args:
            query (str): Query string.
            num_results (int, optional): Number of results to return. Defaults to 10.
        
        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance
        """
        query_vecs = {field: self.vectorizers[field].transform([query]) for field in self.text_fields}
        scores = np.zeros(len(self.docs))

        # Compute cosine similarity for each text field and apply boost
        for field, query_vec in query_vecs.items():
            sim = cosine_similarity(query_vec, self.text_matrices[field]).flatten()
            boost = boost_dict.get(field, 1)
            scores += boost * sim

        # Apply keyword filters
        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                scores = scores * mask.to_numpy()

        # Use argpartition to get top num_results indices
        top_indices = np.argpartition(scores, -num_results)[-num_results:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        # Filter out zero-score results
        top_docs = [self.docs[i] for i in top_indices if scores[i]>0]

        return top_docs
