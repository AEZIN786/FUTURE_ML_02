import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_curve, auc, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import os
import logging
from typing import Any, Union
from src.utils import get_screenshots_dir

def evaluate_model(model: Any, X_test: Any, y_test: Union[list, np.ndarray, pd.Series], model_name: str) -> dict:
    """
    Evaluates a trained model and prints/saves metrics and plots.
    """
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    report = classification_report(y_test, y_pred, zero_division=0)
    
    logging.info(f"\n--- Evaluation for {model_name} ---")
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Classification Report:\n{report}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    save_path = get_screenshots_dir()
    safe_name = model_name.replace(" ", "_")
    plt.savefig(os.path.join(save_path, f'confusion_matrix_{safe_name}.png'))
    plt.close()
    logging.info(f"Confusion matrix saved to screenshots/confusion_matrix_{safe_name}.png")
    
    # ROC Curve
    plot_roc_curve(model, X_test, y_test, model_name, save_path)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_roc_curve(model: Any, X_test: Any, y_test: Union[list, np.ndarray, pd.Series], model_name: str, save_path: str) -> None:
    """
    Plots and saves the ROC curve for binary or multiclass classification.
    """
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)
        classes = model.classes_
        safe_name = model_name.replace(" ", "_")
        
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_scores[:, 1], pos_label=classes[1])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
        else:
            y_test_bin = label_binarize(y_test, classes=classes)
            n_classes = len(classes)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.figure()
            colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'blue', 'purple', 'brown']
            for i, color in zip(range(n_classes), colors[:n_classes]):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Multiclass ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            
        plt.savefig(os.path.join(save_path, f'roc_curve_{safe_name}.png'))
        plt.close()
        logging.info(f"ROC curve saved to screenshots/roc_curve_{safe_name}.png")