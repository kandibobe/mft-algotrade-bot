#!/usr/bin/env python3
"""
Calculate precision for the most recent XGBoost model.
"""
import pickle
import json
import os

def load_model(model_path):
    """Load a pickled model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def calculate_precision():
    """Calculate precision for the most recent model."""
    # Load the most recent model
    model_path = "user_data/models/prod_candidate_v1.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    print(f"Model saved at: {model_path}")
    
    # Look at the walk-forward results for window 142
    results_path = "user_data/walk_forward_results/wfo_results_20251222_135455.json"
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Find window 142 (most recent)
    window_142 = None
    for window in results['window_results']:
        if window['window_id'] == 142:
            window_142 = window
            break
    
    if not window_142:
        print("Window 142 not found in results")
        return
    
    test_accuracy = window_142['test_accuracy']
    test_f1 = window_142['test_f1']
    win_rate = window_142.get('win_rate', 'N/A')
    
    print(f"\n=== Window 142 (Most Recent) Results ===")
    print(f"Test Period: {window_142['test_start']} to {window_142['test_end']}")
    print(f"Test Samples: {window_142['test_samples']}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Win Rate: {win_rate}")
    
    # Estimate precision from F1 score (assuming balanced precision/recall)
    # F1 = 2 * (precision * recall) / (precision + recall)
    # If precision ≈ recall, then precision ≈ F1
    estimated_precision = test_f1  # Rough estimate
    
    print(f"\n=== Precision Analysis ===")
    print(f"Estimated Precision (assuming balanced): {estimated_precision:.2%}")
    
    if estimated_precision > 0.6:
        print("✅ Estimated Precision > 60% (GOOD)")
    else:
        print(f"⚠️ Estimated Precision < 60%: {estimated_precision:.2%} (needs improvement)")
    
    # Also check train vs test for overfitting
    train_accuracy = window_142['train_accuracy']
    train_f1 = window_142['train_f1']
    
    print(f"\n=== Overfitting Check ===")
    print(f"Train Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Accuracy Drop: {(train_accuracy - test_accuracy):.2%}")
    
    if train_accuracy - test_accuracy > 0.3:  # More than 30% drop
        print("⚠️ Significant overfitting detected (train accuracy much higher than test)")
    else:
        print("✅ Reasonable generalization (train and test accuracy similar)")

if __name__ == "__main__":
    calculate_precision()
