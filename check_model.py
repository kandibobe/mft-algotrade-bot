import pickle
import numpy as np

try:
    model = pickle.load(open('user_data/models/BTC_USDT_20251223_145507.pkl', 'rb'))
    print(f'Model type: {type(model)}')
    print(f'Model class: {model.__class__.__name__}')
    
    if hasattr(model, 'feature_importances_'):
        print(f'Number of features expected: {len(model.feature_importances_)}')
        print(f'Feature importance sum: {model.feature_importances_.sum():.4f}')
        print(f'Feature importance shape: {model.feature_importances_.shape}')
        
    if hasattr(model, 'n_features_in_'):
        print(f'n_features_in_: {model.n_features_in_}')
        
    if hasattr(model, 'n_estimators'):
        print(f'n_estimators: {model.n_estimators}')
        
    if hasattr(model, 'feature_names_in_'):
        print(f'Feature names available: {hasattr(model, "feature_names_in_")}')
        if hasattr(model, 'feature_names_in_'):
            print(f'Number of feature names: {len(model.feature_names_in_)}')
            print(f'First 10 feature names: {model.feature_names_in_[:10]}')
            
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
