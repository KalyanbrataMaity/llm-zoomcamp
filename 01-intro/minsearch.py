import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


class Index:
    """
    A simple search index using Tf-Idf and cosine similarity for text fields and exact matching for keyword fields.
    
    Attributes:
    text_fields List[str]: List of text field names to index
    keyword_fields List[str]: List of keyword field names to index
    vectorizers Dict: Dicitonary of TfIdfVectorizer instances for each text field
    keyword_df pd.DataFrame: Dataframe containing keyword field data
    text_matrices Dict: Dictionary containing keyword field data
    docs list: List of documents indexed 
    """

    def __init__(self, text_fields, keyword_fields, vectorizer_params={}) -> None:
        """
        """
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields

        self.vectorizers = {field: TfidfVectorizer(**vectorizer_params) for field in text_fields}
        self.keyword_df = None
        self.text_matrices = {}
        self.docs = []

    def fit(self, docs):
        """
        """
        self.docs = docs 
        keyword_data = {field: [] for field in self.keyword_fields}

        for field in self.text_fields:
            