import cProfile
import pstats
import io
import asyncio
from src.ops.wfo_automation import WFOAutomation

def profile_wfo():
    print("🚀 Starting WFO Profiling (Target: 1 cycle for BTC/USDT)...")
    
    automation = WFOAutomation(pair="BTC/USDT")
    
    pr = cProfile.Profile()
    pr.enable()
    
    # Run a limited cycle for profiling
    # We might need to mock data or use a very small dataset to make profiling feasible
    try:
        automation.run_automated_cycle()
    except Exception as e:
        print(f"Profiling run partially failed (likely due to data): {e}")
        
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30) # Print top 30 functions
    
    print("\n=== WFO Profiling Report (Top 30 Cumulative Time) ===")
    print(s.getvalue())
    print("======================================================")

if __name__ == "__main__":
    profile_wfo()
