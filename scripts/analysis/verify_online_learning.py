
import numpy as np
import pandas as pd
import logging
from src.ml.online_learner import OnlineLearner, OnlineLearningConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OnlineLearningInfrastructureCheck")

def verify_online_learning_setup():
    logger.info("Verifying Online Learning Infrastructure...")
    
    # 1. Test Config
    config = OnlineLearningConfig(
        min_samples_for_comparison=10,
        save_interval=5
    )
    
    # 2. Initialize Learner with a dummy path
    # (It will create a dummy production model internally)
    learner = OnlineLearner("user_data/models/test_prod_model.pkl", config=config)
    
    # 3. Simulate streaming data
    n_features = 32 # Common feature count in our system
    n_samples = 20
    
    X_stream = np.random.randn(n_samples, n_features)
    y_stream = np.random.randint(0, 2, n_samples)
    
    logger.info(f"Simulating {n_samples} data points...")
    for i in range(n_samples):
        learner.update_online(X_stream[i], y_stream[i])
        
    # 4. Check results
    stats = learner.get_performance_stats()
    logger.info(f"Final Production Accuracy: {stats['production_model']['accuracy']:.2%}")
    logger.info(f"Final Online Accuracy: {stats['online_model']['accuracy']:.2%}")
    
    # 5. Check if saving works
    if Path(config.online_model_path).exists():
        logger.info(f"✅ Online model successfully saved to {config.online_model_path}")
    else:
        logger.error("❌ Online model was not saved!")
        
    logger.info("Online Learning Infrastructure Verification Complete.")

if __name__ == "__main__":
    from pathlib import Path
    verify_online_learning_setup()
