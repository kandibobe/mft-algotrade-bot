import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
import logging
import sys
import os
from streamlit_autorefresh import st_autorefresh

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.database.db_manager import DatabaseManager
from src.database.models import TradeRecord, SignalRecord, ExecutionRecord
from src.config.unified_config import load_config # As requested, though we use DatabaseManager mostly

# --- Configuration ---
st.set_page_config(
    page_title="Stoic Citadel Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="data_refresh")

# --- Helper Functions ---
def get_safe_db_url():
    """Get DB URL and ensure it uses 127.0.0.1 instead of localhost for Windows compatibility."""
    # Try to get from env first (standard way)
    url = os.getenv("POSTGRES_URL")
    if not url:
        # Construct from components if full URL not set
        user = os.getenv('POSTGRES_USER', 'stoic_trader')
        password = os.getenv('POSTGRES_PASSWORD', '')
        host = os.getenv('POSTGRES_HOST', '127.0.0.1')
        port = os.getenv('POSTGRES_PORT', '5433')
        db = os.getenv('POSTGRES_DB', 'trading_analytics')
        url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    # Crucial fix: Replace localhost with 127.0.0.1
    if "localhost" in url:
        url = url.replace("localhost", "127.0.0.1")
    
    logger.info(f"Connecting to DB at: {url}")
        
    return url

def mask_db_url(url):
    """Mask password in DB URL for display."""
    try:
        if "@" in url:
            prefix = url.split("@")[0]
            suffix = url.split("@")[1]
            if ":" in prefix:
                user = prefix.split("//")[1].split(":")[0]
                return f"postgresql://{user}:***@{suffix}"
        return url
    except:
        return "postgresql://***:***@..."

def check_db_connection(url):
    """Check if DB is reachable."""
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB Connection check failed: {e}")
        return False

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # DB URL Display
    db_url = get_safe_db_url()
    st.text_input("Database URL", value=mask_db_url(db_url), disabled=True)
    
    # Connection Status
    if check_db_connection(db_url):
        st.success("🟢 Database Connected")
    else:
        st.error("🔴 Database Unreachable")
        
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
        
    st.divider()
    st.info(f"Dashboard Auto-Refresh: 60s")

# --- Data Loading ---
@st.cache_data(ttl=60)
def load_data():
    """
    Load data from the database, joining Trades, Signals, and Executions.
    Returns a pandas DataFrame.
    """
    # Use DatabaseManager (which should use the correct config/database.py settings now)
    session: Session = DatabaseManager.get_session_factory()()
    try:
        # Query trades with eager loading or join
        query = session.query(TradeRecord).outerjoin(SignalRecord).outerjoin(ExecutionRecord)
        trades = query.all()
        
        if not trades:
            return pd.DataFrame()

        data = []
        for trade in trades:
            record = {
                # Trade Info
                "trade_id": trade.id,
                "symbol": trade.symbol,
                "exchange": trade.exchange,
                "side": trade.side,
                "entry_price": float(trade.entry_price) if trade.entry_price else 0.0,
                "exit_price": float(trade.exit_price) if trade.exit_price else 0.0,
                "amount": float(trade.amount) if trade.amount else 0.0,
                "pnl_usd": float(trade.pnl_usd) if trade.pnl_usd else 0.0,
                "pnl_pct": float(trade.pnl_pct) if trade.pnl_pct else 0.0,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "strategy": trade.strategy_name,
                "status": trade.status,
                
                # Signal Info
                "model_confidence": float(trade.signal.model_confidence) if trade.signal and trade.signal.model_confidence else 0.0,
                "signal_regime": trade.signal.regime if trade.signal else "Unknown",
                
                # Execution Info
                "slippage_pct": float(trade.execution.slippage_pct) if trade.execution and trade.execution.slippage_pct else 0.0,
            }
            data.append(record)
            
        df = pd.DataFrame(data)
        
        if not df.empty:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
    finally:
        session.close()

# --- Main Dashboard ---
def main():
    st.title("🛡️ The Cockpit - Stoic Citadel")
    
    # Load Data
    df = load_data()
    
    if df.empty:
        st.warning("No trading data found in the database yet.")
        st.info("Waiting for trades to be recorded...")
        return

    # --- Tab 1: Live Monitor ---
    tab1, tab2, tab3 = st.tabs(["📊 Live Monitor", "🔬 Deep Dive", "📉 Drift Analysis"])
    
    with tab1:
        st.subheader("Market Pulse")
        
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_pnl = df['pnl_usd'].sum()
        win_rate = (df[df['pnl_usd'] > 0].shape[0] / df.shape[0]) * 100 if df.shape[0] > 0 else 0
        open_positions = df[df['status'] == 'open'].shape[0]
        trade_count = df.shape[0]

        col1.metric("Total PnL (USD)", f"${total_pnl:,.2f}", delta_color="normal")
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Open Positions", f"{open_positions}")
        col4.metric("Total Trades", f"{trade_count}")
        
        st.divider()
        
        # Recent Trades Dataframe
        st.subheader("Recent Trades (Last 20)")
        
        # Sort by date desc and take top 20
        recent_trades = df.sort_values('entry_time', ascending=False).head(20)
        
        # Styling
        def highlight_pnl(val):
            color = '#90EE90' if val > 0 else '#FFB6C1' if val < 0 else '' # Light green / Light red
            return f'background-color: {color}; color: black'

        display_cols = ['entry_time', 'symbol', 'side', 'strategy', 'entry_price', 'exit_price', 'pnl_usd', 'pnl_pct', 'model_confidence']
        
        st.dataframe(
            recent_trades[display_cols].style.map(highlight_pnl, subset=['pnl_usd', 'pnl_pct']),
            use_container_width=True,
            hide_index=True
        )

    # --- Tab 2: Deep Dive (Signals) ---
    with tab2:
        st.subheader("Signal & Execution Analysis")
        
        col_charts_1, col_charts_2 = st.columns(2)
        
        with col_charts_1:
            st.markdown("#### Prediction Confidence vs Outcome")
            if not df.empty:
                fig_conf = px.scatter(
                    df, 
                    x="model_confidence", 
                    y="pnl_pct", 
                    color="pnl_usd",
                    color_continuous_scale=px.colors.diverging.RdYlGn,
                    title="Does Higher Confidence Mean Higher PnL?",
                    hover_data=["symbol", "strategy"]
                )
                fig_conf.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_conf, use_container_width=True)
            else:
                st.info("No data for scatter plot.")

        with col_charts_2:
            st.markdown("#### Slippage Analysis")
            if not df.empty:
                fig_slip = px.histogram(
                    df, 
                    x="slippage_pct", 
                    nbins=20, 
                    title="Slippage Distribution",
                    color_discrete_sequence=['#FFA07A']
                )
                st.plotly_chart(fig_slip, use_container_width=True)
            else:
                st.info("No data for histogram.")

    # --- Tab 3: Drift Analysis ---
    with tab3:
        st.subheader("Reality Check (Drift Report)")
        
        st.markdown("""
        **What is Drift Analysis?**
        Verifies if the backtest simulation matches live execution results over the last 24 hours.
        """)
        
        # Placeholder for now, as requested
        st.info("ℹ️ To generate a fresh report, run: `python src/analysis/reality_check.py`")
        
        st.code("""
=== REALITY CHECK REPORT: BTC/USDT ===
Time Range: 2025-12-24 to 2025-12-25
Match Rate: 97.58%
Discrepancies: 7
...
        """, language="text")
        
        if st.button("Run Reality Check Now (This may take a minute)"):
            with st.spinner("Running simulation..."):
                # We could run the script via subprocess here
                import subprocess
                try:
                    result = subprocess.run([sys.executable, "src/analysis/reality_check.py"], capture_output=True, text=True)
                    st.text_area("Output", result.stdout + result.stderr, height=400)
                except Exception as e:
                    st.error(f"Failed to run script: {e}")

if __name__ == "__main__":
    main()
