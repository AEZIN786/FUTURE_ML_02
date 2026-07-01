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

from src.utils import get_data_path, save_model, load_model, setup_logging
from src.preprocessing import preprocess_dataframe
from src.evaluate import evaluate_model

def load_and_prepare_data() -> pd.DataFrame:
    data_path = get_data_path()
    df = pd.read_csv(data_path)
    df = preprocess_dataframe(df, 'ticket_text')
    return df

def train_and_tune_model(model_class, param_grid: dict, X_train, y_train, model_name: str):
    logging.info(f"Tuning {model_name} with GridSearchCV for Priority...")
    base_model = model_class()
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_models(X_train, y_train, X_test, y_test):
    models_config = {
        'Logistic Regression': (LogisticRegression, {'max_iter': [1000], 'class_weight': ['balanced'], 'C': [0.1, 1.0, 10.0]}),
        'Multinomial Naive Bayes': (MultinomialNB, {'alpha': [0.1, 0.5, 1.0]}),
        'Linear SVM': (LinearSVC, {'max_iter': [10000], 'class_weight': ['balanced'], 'dual': ['auto'], 'C': [0.1, 1.0, 10.0]}),
        'Random Forest': (RandomForestClassifier, {'n_estimators': [50, 100], 'max_depth': [None, 10], 'class_weight': ['balanced'], 'random_state': [42]}),
        'Decision Tree': (DecisionTreeClassifier, {'max_depth': [None, 10, 20], 'class_weight': ['balanced'], 'random_state': [42]})
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
    print("Starting Priority Training...")
    df = load_and_prepare_data()
    
    tfidf_vec = load_model('tfidf.pkl')
    X_tfidf = tfidf_vec.transform(df['cleaned_text'])
    y = df['priority']
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
    results, best_model, best_name = train_models(X_train, y_train, X_test, y_test)
    
    save_model(best_model, 'priority_model.pkl')
    logging.info(f"Best priority model ({best_name}) saved.")
    evaluate_model(best_model, X_test, y_test, best_name)
    print("Priority Training Complete!")

if __name__ == "__main__":
    main()