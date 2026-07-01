import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_model, setup_logging
from src.preprocessing import preprocess_text
import logging
from typing import Dict, Any

def predict_ticket(ticket_text: str) -> Dict[str, Any]:
    """
    Predicts the category and priority of a given support ticket.
    """
    tfidf_vec = load_model('tfidf.pkl')
    category_model = load_model('category_model.pkl')
    priority_model = load_model('priority_model.pkl')
    
    cleaned_text = preprocess_text(ticket_text)
    X_vec = tfidf_vec.transform([cleaned_text])
    
    cat_pred = category_model.predict(X_vec)[0]
    if hasattr(category_model, "predict_proba"):
        cat_probs = category_model.predict_proba(X_vec)[0]
        cat_confidence = max(cat_probs) * 100
        cat_prob_dict = dict(zip(category_model.classes_, cat_probs))
    else:
        cat_confidence = 100.0 
        cat_prob_dict = {cls: (1.0 if cls == cat_pred else 0.0) for cls in category_model.classes_}
    
    pri_pred = priority_model.predict(X_vec)[0]
    if hasattr(priority_model, "predict_proba"):
        pri_probs = priority_model.predict_proba(X_vec)[0]
        pri_confidence = max(pri_probs) * 100
        pri_prob_dict = dict(zip(priority_model.classes_, pri_probs))
    else:
        pri_confidence = 100.0
        pri_prob_dict = {cls: (1.0 if cls == pri_pred else 0.0) for cls in priority_model.classes_}
    
    return {
        'category': cat_pred,
        'category_confidence': cat_confidence,
        'category_probabilities': cat_prob_dict,
        'priority': pri_pred,
        'priority_confidence': pri_confidence,
        'priority_probabilities': pri_prob_dict
    }

def main() -> None:
    """CLI interface for predicting a single ticket."""
    setup_logging()
    if len(sys.argv) > 1:
        ticket = " ".join(sys.argv[1:])
    else:
        ticket = input("Enter Ticket: ")
        
    if not ticket.strip():
        logging.error("Ticket text cannot be empty.")
        return
        
    result = predict_ticket(ticket)
    print("\n--- Prediction Results ---")
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['category_confidence']:.1f}%")
    print(f"Priority: {result['priority']}")
    print(f"Confidence: {result['priority_confidence']:.1f}%")
    print("--------------------------\n")

if __name__ == "__main__":
    main()