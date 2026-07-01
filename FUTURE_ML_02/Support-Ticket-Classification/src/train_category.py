import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import logging

from src.utils import get_data_path, save_model, setup_logging
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import create_tfidf_vectorizer, create_bow_vectorizer
from src.evaluate import evaluate_model

def load_and_prepare_data() -> pd.DataFrame:
    data_path = get_data_path()
    df = pd.read_csv(data_path)
    logging.info(f"Dataset loaded with {len(df)} records.")
    df = preprocess_dataframe(df, 'ticket_text')
    return df

def train_and_tune_model(model_class, param_grid: dict, X_train, y_train, model_name: str):
    logging.info(f"Tuning {model_name} with GridSearchCV...")
    # Now model_class is the actual class (e.g., LogisticRegression), so we can call it with ()
    base_model = model_class() 
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_models(X_train, y_train, X_test, y_test):
    # FIX: We pass the CLASSES here, not the instantiated objects!
    models_config = {
        'Logistic Regression': (LogisticRegression, {'max_iter': [1000], 'C': [0.1, 1.0, 10.0]}),
        'Multinomial Naive Bayes': (MultinomialNB, {'alpha': [0.1, 0.5, 1.0]}),
        'Linear SVM': (LinearSVC, {'max_iter': [10000], 'dual': ['auto'], 'C': [0.1, 1.0, 10.0]}),
        'Random Forest': (RandomForestClassifier, {'n_estimators': [50, 100], 'max_depth': [None, 10], 'random_state': [42]}),
        'Decision Tree': (DecisionTreeClassifier, {'max_depth': [None, 10, 20], 'random_state': [42]})
    }
    
    results = {}
    best_model = None
    best_score = 0.0
    best_model_name = ""
    
    for name, (model_class, param_grid) in models_config.items():
        best_model_instance = train_and_tune_model(model_class, param_grid, X_train, y_train, name)
        y_pred = best_model_instance.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'model': best_model_instance, 'accuracy': acc, 'predictions': y_pred}
        logging.info(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_score:
            best_score = acc
            best_model = best_model_instance
            best_model_name = name
            
    return results, best_model, best_model_name

def main() -> None:
    setup_logging()
    print("Starting Category Training...")
    df = load_and_prepare_data()
    
    X_tfidf, tfidf_vec = create_tfidf_vectorizer(df['cleaned_text'])
    X_bow, bow_vec = create_bow_vectorizer(df['cleaned_text'])
    
    y = df['category']
    
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
    X_train_bow, X_test_bow, _, _ = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info("--- Training with TF-IDF ---")
    results_tfidf, best_model_tfidf, best_name_tfidf = train_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    logging.info("--- Training with Bag of Words ---")
    results_bow, best_model_bow, best_name_bow = train_models(X_train_bow, y_train, X_test_bow, y_test)
    
    if results_tfidf[best_name_tfidf]['accuracy'] >= results_bow[best_name_bow]['accuracy']:
        logging.info(f"TF-IDF performs better or equal. Best model: {best_name_tfidf}")
        save_model(results_tfidf[best_name_tfidf]['model'], 'category_model.pkl')
        evaluate_model(results_tfidf[best_name_tfidf]['model'], X_test_tfidf, y_test, best_name_tfidf)
    else:
        logging.info(f"BoW performs better. Best model: {best_name_bow}")
        save_model(results_bow[best_name_bow]['model'], 'category_model.pkl')
        evaluate_model(results_bow[best_name_bow]['model'], X_test_bow, y_test, best_name_bow)
        
    print("Category Training Complete!")

if __name__ == "__main__":
    main()