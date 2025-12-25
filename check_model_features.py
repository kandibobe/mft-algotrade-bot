import pickle
import numpy as np

model = pickle.load(open('user_data/models/BTC_USDT_20251223_145507.pkl', 'rb'))

print("Model feature names:")
if hasattr(model, 'feature_names_in_'):
    for i, name in enumerate(model.feature_names_in_):
        print(f"{i}: {name}")
        
print(f"\nTotal features: {len(model.feature_names_in_)}")

# Group features by type
feature_groups = {}
for name in model.feature_names_in_:
    if 'open' in name:
        feature_groups.setdefault('price', []).append(name)
    elif 'high' in name:
        feature_groups.setdefault('price', []).append(name)
    elif 'low' in name:
        feature_groups.setdefault('price', []).append(name)
    elif 'close' in name:
        feature_groups.setdefault('price', []).append(name)
    elif 'volume' in name:
        feature_groups.setdefault('volume', []).append(name)
    elif 'rolling' in name:
        feature_groups.setdefault('rolling_stats', []).append(name)
    elif 'returns' in name:
        feature_groups.setdefault('returns', []).append(name)
    elif 'volatility' in name:
        feature_groups.setdefault('volatility', []).append(name)
    elif 'log' in name:
        feature_groups.setdefault('log', []).append(name)
    else:
        feature_groups.setdefault('other', []).append(name)

print("\nFeature groups:")
for group, features in feature_groups.items():
    print(f"{group}: {len(features)} features")
    if len(features) <= 5:
        print(f"  {features}")
    else:
        print(f"  {features[:3]} ... {features[-2:]}")
        
# Check if we have the basic OHLCV columns
basic_cols = ['open', 'high', 'low', 'close', 'volume']
missing_basic = [col for col in basic_cols if col not in model.feature_names_in_]
if missing_basic:
    print(f"\nWARNING: Missing basic columns: {missing_basic}")
else:
    print("\nAll basic OHLCV columns present")
