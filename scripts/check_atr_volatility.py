import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range (ATR)."""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def analyze_volatility_dynamics(pair="BTC/USDT"):
    """Analyze if volatility (ATR) increases during market crashes."""
    # Load data
    if pair == "BTC/USDT":
        filepath = "user_data/data/binance/BTC_USDT-5m.feather"
    else:
        filepath = f"user_data/data/binance/{pair.replace('/', '_')}-5m.feather"
    
    print(f"Analyzing {pair} from {filepath}")
    df = pd.read_feather(filepath)
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate ATR
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR as percentage of price
    
    # Identify market crashes (large negative returns)
    # Define crash as returns < -2% (for 5m timeframe this is significant)
    df['crash'] = df['returns'] < -0.02  # -2%
    
    # Calculate average ATR during crashes vs normal periods
    atr_during_crashes = df.loc[df['crash'], 'atr_pct'].mean()
    atr_normal = df.loc[~df['crash'], 'atr_pct'].mean()
    
    print(f"\n=== Volatility Analysis for {pair} ===")
    print(f"Total periods: {len(df)}")
    print(f"Crash periods (returns < -2%): {df['crash'].sum()} ({df['crash'].sum()/len(df)*100:.2f}%)")
    print(f"Average ATR % during crashes: {atr_during_crashes:.4f}%")
    print(f"Average ATR % during normal periods: {atr_normal:.4f}%")
    print(f"Volatility ratio (crash/normal): {atr_during_crashes/atr_normal:.2f}x")
    
    # Check if volatility is higher during crashes (should be > 1.0)
    if atr_during_crashes > atr_normal:
        print("✓ PASS: Volatility is higher during crashes (as expected)")
    else:
        print("✗ WARNING: Volatility is NOT higher during crashes (unexpected!)")
    
    # Additional analysis: Check correlation between returns magnitude and ATR
    df['returns_abs'] = df['returns'].abs()
    correlation = df[['returns_abs', 'atr_pct']].corr().iloc[0, 1]
    print(f"\nCorrelation between |returns| and ATR%: {correlation:.4f}")
    
    if correlation > 0:
        print("✓ Positive correlation: Volatility increases with larger price moves")
    else:
        print("✗ Negative correlation: Unexpected relationship")
    
    # Plot for visual inspection
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Price and ATR
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close Price', alpha=0.7)
    ax1.set_ylabel('Price (USD)')
    ax1.set_title(f'{pair} - Price and ATR%')
    ax1.legend(loc='upper left')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, df['atr_pct'], label='ATR%', color='orange', alpha=0.7)
    ax1_twin.set_ylabel('ATR %')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Returns and crash periods
    ax2 = axes[1]
    ax2.plot(df.index, df['returns'] * 100, label='Returns %', alpha=0.7)
    crash_periods = df[df['crash']]
    ax2.scatter(crash_periods.index, crash_periods['returns'] * 100, 
                color='red', s=10, label='Crash (< -2%)', alpha=0.5)
    ax2.axhline(y=-2, color='r', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Returns %')
    ax2.set_title('Returns with Crash Periods Highlighted')
    ax2.legend()
    
    # Plot 3: ATR distribution during crashes vs normal
    ax3 = axes[2]
    atr_crash = df.loc[df['crash'], 'atr_pct']
    atr_normal_vals = df.loc[~df['crash'], 'atr_pct'].sample(min(len(atr_crash)*10, len(df)), random_state=42)
    
    ax3.hist(atr_normal_vals, bins=50, alpha=0.5, label='Normal Periods', density=True)
    ax3.hist(atr_crash, bins=50, alpha=0.5, label='Crash Periods', density=True)
    ax3.set_xlabel('ATR %')
    ax3.set_ylabel('Density')
    ax3.set_title('ATR Distribution: Crash vs Normal Periods')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'volatility_analysis_{pair.replace("/", "_")}.png', dpi=150)
    print(f"\nPlot saved as 'volatility_analysis_{pair.replace('/', '_')}.png'")
    
    return df

if __name__ == "__main__":
    # Analyze both BTC and ETH
    for pair in ["BTC/USDT", "ETH/USDT"]:
        try:
            df = analyze_volatility_dynamics(pair)
            print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"Error analyzing {pair}: {e}")
            print("\n" + "="*60 + "\n")
