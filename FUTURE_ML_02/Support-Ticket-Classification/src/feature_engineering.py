from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.utils import save_model
import logging
from typing import Tuple, Union
import pandas as pd

def create_tfidf_vectorizer(texts: Union[pd.Series, list], max_features: int = 5000) -> Tuple:
    """
    Creates and fits a TF-IDF Vectorizer on the provided texts.
    
    Args:
        texts (Union[pd.Series, list]): The cleaned text data.
        max_features (int): Maximum number of features to keep.
        
    Returns:
        Tuple: (TF-IDF matrix, fitted TfidfVectorizer object)
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(texts)
    logging.info(f"TF-IDF Vectorizer fitted. Shape: {X_tfidf.shape}")
    save_model(vectorizer, 'tfidf.pkl')
    return X_tfidf, vectorizer

def create_bow_vectorizer(texts: Union[pd.Series, list], max_features: int = 5000) -> Tuple:
    """
    Creates and fits a Bag of Words Vectorizer on the provided texts.
    
    Args:
        texts (Union[pd.Series, list]): The cleaned text data.
        max_features (int): Maximum number of features to keep.
        
    Returns:
        Tuple: (BoW matrix, fitted CountVectorizer object)
    """
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_bow = vectorizer.fit_transform(texts)
    logging.info(f"BoW Vectorizer fitted. Shape: {X_bow.shape}")
    return X_bow, vectorizer