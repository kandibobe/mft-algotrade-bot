
import logging

def verify_imports():
    """
    Verifies that all critical libraries can be imported.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    libraries = [
        "freqtrade", "pandas", "talib", "xgboost", 
        "lightgbm", "river", "feast"
    ]
    
    all_ok = True
    for lib in libraries:
        try:
            __import__(lib)
            logging.info(f"[OK] Successfully imported '{lib}'")
        except ImportError as e:
            logging.error(f"[FAIL] Failed to import '{lib}': {e}")
            all_ok = False
            
    if all_ok:
        logging.info("\n✅ All critical libraries are installed and importable.")
    else:
        logging.error("\n❌ Some critical libraries are missing. Please install them.")

if __name__ == "__main__":
    verify_imports()
