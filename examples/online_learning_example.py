"""
Example: Online Learning (Model Retraining) for Trading Models
===============================================================

This example demonstrates how to use the OnlineLearner for:
1. Continuous model updates on new trading data
2. Model drift detection
3. A/B testing between production and online models
4. Gradual rollout of improved models

Author: Stoic Citadel Team
License: MIT
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import OnlineLearner
try:
    from src.ml.online_learner import OnlineLearner, OnlineLearningConfig
    print("✓ OnlineLearner imported successfully")
except ImportError as e:
    print(f"✗ Failed to import OnlineLearner: {e}")
    exit(1)


def create_sample_model():
    """Create a sample production model for demonstration."""
    from sklearn.linear_model import LogisticRegression
    
    # Create a simple model
    model = LogisticRegression(random_state=42)
    
    # Generate sample training data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model


def simulate_trading_data(n_samples=1000, n_features=10):
    """Simulate trading data with concept drift."""
    np.random.seed(42)
    
    # Generate base features
    X = np.random.randn(n_samples, n_features)
    
    # Simulate concept drift: change data distribution halfway
    drift_point = n_samples // 2
    X[drift_point:] += 0.5  # Add shift to simulate drift
    
    # Generate labels (simplified trading signals)
    # Complex pattern that changes with drift
    y = np.zeros(n_samples)
    for i in range(n_samples):
        if i < drift_point:
            # Pattern before drift
            if X[i, 0] > 0 and X[i, 1] < 0:
                y[i] = 1
        else:
            # Pattern after drift
            if X[i, 2] > 0.5 or X[i, 3] < -0.5:
                y[i] = 1
    
    return X, y


def main():
    """Main example demonstrating OnlineLearner functionality."""
    print("\n" + "="*70)
    print("ONLINE LEARNING EXAMPLE")
    print("="*70)
    
    # Create directories if they don't exist
    model_dir = Path("user_data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create and save a sample production model
    print("\n1. Creating sample production model...")
    prod_model = create_sample_model()
    prod_model_path = model_dir / "production_model.pkl"
    
    with open(prod_model_path, 'wb') as f:
        pickle.dump(prod_model, f)
    
    print(f"   Production model saved to: {prod_model_path}")
    
    # Step 2: Initialize OnlineLearner
    print("\n2. Initializing OnlineLearner...")
    
    # Custom configuration
    config = OnlineLearningConfig(
        base_model_path=str(prod_model_path),
        learning_rate=0.01,
        improvement_threshold=0.03,  # 3% improvement threshold
        ab_test_traffic_pct=0.2,     # 20% traffic to online model
        enable_drift_detection=True,
        drift_detection_window=500,
        save_interval=50
    )
    
    learner = OnlineLearner(str(prod_model_path), config=config)
    print(f"   OnlineLearner initialized with {learner.config.use_river and 'river' or 'sklearn'} backend")
    
    # Step 3: Simulate streaming trading data
    print("\n3. Simulating streaming trading data with concept drift...")
    X_stream, y_stream = simulate_trading_data(n_samples=1000)
    
    print(f"   Generated {len(X_stream)} samples with concept drift at sample {len(X_stream)//2}")
    
    # Step 4: Online learning loop
    print("\n4. Starting online learning loop...")
    print("   [Each '.' represents 10 updates, 'D' indicates drift detection, 'R' indicates replacement recommended]")
    
    replacements = 0
    drift_detections = 0
    
    for i in range(len(X_stream)):
        # Update online model with new data
        learner.update_online(X_stream[i], int(y_stream[i]))
        
        # Check for model drift
        if learner._check_drift_detected():
            if drift_detections == 0:  # Only log first detection
                print("\n   D", end="", flush=True)
            drift_detections += 1
        
        # Check if online model should replace production model
        if learner.should_replace_prod_model():
            if replacements == 0:  # Only log first recommendation
                print("\n   R", end="", flush=True)
            
            # Replace production model
            if learner.replace_production_model():
                replacements += 1
                print(f"\n   ✓ Production model replaced (replacement #{replacements})")
        
        # Progress indicator
        if i % 10 == 0:
            print(".", end="", flush=True)
        
        # Periodically show stats
        if i % 200 == 0 and i > 0:
            stats = learner.get_performance_stats()
            print(f"\n   [Sample {i}] Prod Acc: {stats['production_model']['accuracy']:.3f}, "
                  f"Online Acc: {stats['online_model']['accuracy']:.3f}, "
                  f"Improvement: {stats['comparison']['improvement']:.3f}")
    
    print("\n\n5. Online learning completed!")
    
    # Step 5: Display final statistics
    print("\n" + "-"*70)
    print("FINAL STATISTICS")
    print("-"*70)
    
    final_stats = learner.get_performance_stats()
    
    print(f"\nProduction Model:")
    print(f"  • Accuracy: {final_stats['production_model']['accuracy']:.3f}")
    print(f"  • Updates processed: {final_stats['production_model']['update_count']}")
    print(f"  • Average performance: {final_stats['production_model']['avg_performance']:.3f}")
    
    print(f"\nOnline Model:")
    print(f"  • Accuracy: {final_stats['online_model']['accuracy']:.3f}")
    print(f"  • Updates processed: {final_stats['online_model']['update_count']}")
    print(f"  • Average performance: {final_stats['online_model']['avg_performance']:.3f}")
    
    print(f"\nComparison:")
    print(f"  • Improvement: {final_stats['comparison']['improvement']:.3f}")
    print(f"  • Should replace production: {final_stats['comparison']['should_replace']}")
    print(f"  • Drift detected: {final_stats['comparison']['drift_detected']}")
    
    print(f"\nSummary:")
    print(f"  • Total samples processed: {len(X_stream)}")
    print(f"  • Model replacements: {replacements}")
    print(f"  • Drift detections: {drift_detections}")
    
    # Step 6: Demonstrate A/B testing
    print("\n" + "-"*70)
    print("A/B TESTING DEMONSTRATION")
    print("-"*70)
    
    print("\nStarting A/B test with 30% traffic to online model...")
    learner.start_ab_test(traffic_pct=0.3)
    
    # Simulate some predictions with A/B testing
    ab_test_predictions = []
    for i in range(100):
        pred = learner.predict(X_stream[i % len(X_stream)], use_ab_test=True)
        ab_test_predictions.append(pred)
    
    # Stop A/B test and show results
    ab_results = learner.stop_ab_test()
    
    print(f"\nA/B Test Results:")
    print(f"  • Duration: {ab_results['duration']:.1f} seconds")
    print(f"  • Total samples: {ab_results['total_samples']}")
    print(f"  • Traffic to online model: {ab_results['traffic_pct']*100:.1f}%")
    
    # Step 7: Batch evaluation demonstration
    print("\n" + "-"*70)
    print("BATCH EVALUATION")
    print("-"*70)
    
    # Create a test batch
    X_test = X_stream[-100:]
    y_test = y_stream[-100:]
    
    batch_results = learner.evaluate_on_batch(X_test, y_test)
    
    print(f"\nBatch Evaluation (last 100 samples):")
    print(f"  • Production accuracy: {batch_results['prod_accuracy']:.3f}")
    print(f"  • Online accuracy: {batch_results['online_accuracy']:.3f}")
    print(f"  • Production correct: {batch_results['prod_correct']}/{batch_results['total_samples']}")
    print(f"  • Online correct: {batch_results['online_correct']}/{batch_results['total_samples']}")
    
    # Step 8: Save the final online model
    print("\n" + "-"*70)
    print("MODEL PERSISTENCE")
    print("-"*70)
    
    if learner.save_online_model():
        print(f"\n✓ Online model saved to: {learner.config.online_model_path}")
    
    # Cleanup
    print("\n" + "="*70)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nKey features demonstrated:")
    print("1. ✓ Continuous online learning with new data")
    print("2. ✓ Model drift detection")
    print("3. ✓ A/B testing between production and online models")
    print("4. ✓ Gradual rollout of improved models")
    print("5. ✓ Performance metrics tracking")
    print("6. ✓ Model persistence (save/load)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error in example: {e}")
        import traceback
        traceback.print_exc()
