import logging
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from streamlit_autorefresh import st_autorefresh

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.analysis.monte_carlo import MonteCarloSimulator
    from src.database.db_manager import DatabaseManager
    from src.database.models import ExecutionRecord, SignalRecord, TradeRecord
except ImportError:
    # Handle missing dependencies if needed or just let it fail later properly
    pass

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
        user = os.getenv("POSTGRES_USER", "stoic_trader")
        password = os.getenv("POSTGRES_PASSWORD", "")
        host = os.getenv("POSTGRES_HOST", "127.0.0.1")
        port = os.getenv("POSTGRES_PORT", "5433")
        db = os.getenv("POSTGRES_DB", "trading_analytics")
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
    st.info("Dashboard Auto-Refresh: 60s")


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
                "model_confidence": float(trade.signal.model_confidence)
                if trade.signal and trade.signal.model_confidence
                else 0.0,
                "signal_regime": trade.signal.regime if trade.signal else "Unknown",
                # Execution Info
                "slippage_pct": float(trade.execution.slippage_pct)
                if trade.execution and trade.execution.slippage_pct
                else 0.0,
                # Attribution Info
                "attribution": trade.meta_data if trade.meta_data else {},
            }
            data.append(record)

        df = pd.DataFrame(data)

        if not df.empty:
            df["entry_time"] = pd.to_datetime(df["entry_time"])
            df["exit_time"] = pd.to_datetime(df["exit_time"])

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

    # --- Tab 1: Live Monitor & Performance ---
    tab1, tab2, tab3 = st.tabs(["📊 Performance", "🔬 Deep Dive", "📉 Risk Analysis"])

    with tab1:
        st.subheader("Market Pulse")

        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)

        total_pnl = df["pnl_usd"].sum()
        win_rate = (df[df["pnl_usd"] > 0].shape[0] / df.shape[0]) * 100 if df.shape[0] > 0 else 0
        open_positions = df[df["status"] == "open"].shape[0]
        trade_count = df.shape[0]

        col1.metric("Total PnL (USD)", f"${total_pnl:,.2f}", delta_color="normal")
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Open Positions", f"{open_positions}")
        col4.metric("Total Trades", f"{trade_count}")

        st.divider()

        # Recent Trades Dataframe
        st.subheader("Recent Trades (Last 20)")

        # Sort by date desc and take top 20
        recent_trades = df.sort_values("entry_time", ascending=False).head(20)

        # Styling
        def highlight_pnl(val):
            color = (
                "#90EE90" if val > 0 else "#FFB6C1" if val < 0 else ""
            )  # Light green / Light red
            return f"background-color: {color}; color: black"

        display_cols = [
            "entry_time",
            "symbol",
            "side",
            "strategy",
            "entry_price",
            "exit_price",
            "pnl_usd",
            "pnl_pct",
            "model_confidence",
        ]

        st.dataframe(
            recent_trades[display_cols].style.map(highlight_pnl, subset=["pnl_usd", "pnl_pct"]),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # Equity Curve & Drawdown
        st.subheader("Equity Curve & Drawdown")
        if not df.empty:
            # Sort by time
            df_sorted = df.sort_values("exit_time")
            # Calculate cumulative PnL
            df_sorted["equity"] = df_sorted["pnl_usd"].cumsum()

            # Calculate Drawdown
            df_sorted["max_equity"] = df_sorted["equity"].cummax()
            df_sorted["drawdown"] = df_sorted["equity"] - df_sorted["max_equity"]

            # Plot Equity
            fig_equity = px.line(
                df_sorted,
                x="exit_time",
                y="equity",
                title="Portfolio Equity Curve (USD)",
                markers=True,
            )
            fig_equity.update_traces(line_color="#4CAF50")
            st.plotly_chart(fig_equity, use_container_width=True)

            # Plot Drawdown (Waterfall style using bar chart)
            fig_dd = px.bar(
                df_sorted,
                x="exit_time",
                y="drawdown",
                title="Drawdown Waterfall (USD)",
            )
            fig_dd.update_traces(marker_color="#FF5252")
            st.plotly_chart(fig_dd, use_container_width=True)

        else:
            st.info("Not enough data for equity curve.")

    # --- Tab 2: Deep Dive (Signals & Attribution) ---
    with tab2:
        st.subheader("Trade Attribution: Why was the deal opened?")

        if not df.empty:
            # Selector for specific trade
            trade_options = df.apply(
                lambda x: f"{x['entry_time']} - {x['symbol']} ({x['side']}) PnL: ${x['pnl_usd']:.2f}",
                axis=1,
            )
            selected_trade_str = st.selectbox("Select a Trade to Analyze", options=trade_options)

            if selected_trade_str:
                idx = trade_options[trade_options == selected_trade_str].index[0]
                trade_data = df.iloc[idx]

                st.markdown(
                    f"### Trade Analysis: {trade_data['symbol']} ({trade_data['side'].upper()})"
                )

                # Layout
                col_attr_1, col_attr_2, col_attr_3 = st.columns(3)

                with col_attr_1:
                    st.markdown("#### 🧠 Model Reasoning")
                    st.metric("Model Confidence", f"{trade_data['model_confidence']:.2f}")
                    st.metric("Market Regime", f"{trade_data['signal_regime']}")

                    # Feature Importance (if available in metadata)
                    attr_data = trade_data.get("attribution", {})
                    if attr_data and isinstance(attr_data, dict):
                        strategy_name = attr_data.get("strategy_name", "Unknown")
                        st.info(f"Strategy: {strategy_name}")
                        # Potentially more details here if we logged them

                with col_attr_2:
                    st.markdown("#### 🛡️ Risk Parameters")
                    # Assuming we can calculate or get these
                    entry = trade_data["entry_price"]
                    # We might need stop loss from metadata if not in main columns,
                    # but let's just show what we have
                    st.metric("Entry Price", f"{entry:.4f}")
                    st.metric("Exit Price", f"{trade_data['exit_price']:.4f}")

                with col_attr_3:
                    st.markdown("#### ⚡ Execution Quality")
                    st.metric("Slippage", f"{trade_data['slippage_pct']:.4f}%")
                    st.metric("Realized PnL", f"${trade_data['pnl_usd']:.2f}")

                st.divider()

        st.subheader("Aggregate Signal Analysis")

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
                    hover_data=["symbol", "strategy"],
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
                    color_discrete_sequence=["#FFA07A"],
                )
                st.plotly_chart(fig_slip, use_container_width=True)
            else:
                st.info("No data for histogram.")

    # --- Tab 3: Risk & Drift Analysis ---
    with tab3:
        st.subheader("Monte Carlo Simulation (Live)")

        if not df.empty:
            col_mc1, col_mc2 = st.columns(2)
            with col_mc1:
                initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
                iterations = st.slider("Iterations", 100, 5000, 1000)
            with col_mc2:
                max_dd_limit = st.slider("Max Drawdown Limit", 0.1, 0.9, 0.5)

            if st.button("Run Monte Carlo Simulation"):
                with st.spinner("Simulating..."):
                    # Prepare data
                    sim_df = df.copy()
                    sim_df["profit_ratio"] = sim_df["pnl_pct"]

                    simulator = MonteCarloSimulator(
                        trades_df=sim_df,
                        iterations=iterations,
                        initial_capital=initial_capital,
                        max_drawdown_limit=max_dd_limit,
                    )
                    simulator.run()
                    summary = simulator.get_summary()

                    # Display metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Probability of Ruin", f"{summary['probability_of_ruin']:.2f}%")
                    m2.metric("Mean Max Drawdown", f"{summary['mean_max_drawdown']:.2%}")
                    m3.metric("99th %ile Drawdown", f"{summary['99th_percentile_drawdown']:.2%}")

                    # Plot equity curves
                    st.markdown("#### Simulated Equity Curves")
                    fig_mc = plt.figure(figsize=(10, 6))

                    # Plot a subset of curves
                    import matplotlib.pyplot as plt
                    import numpy as np

                    subset_indices = np.random.choice(
                        len(simulator.all_equity_curves), size=min(100, iterations), replace=False
                    )
                    for i in subset_indices:
                        plt.plot(simulator.all_equity_curves[i], color="gray", alpha=0.1)

                    # Plot median
                    median_curve = np.median(simulator.all_equity_curves, axis=0)
                    plt.plot(median_curve, color="blue", linewidth=2, label="Median")

                    plt.title("Projected Equity Paths")
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig_mc)
        else:
            st.info("Need trade data to run simulation.")

        st.divider()

        st.subheader("Reality Check (Drift Report)")

        st.markdown("""
        **What is Drift Analysis?**
        Verifies if the backtest simulation matches live execution results over the last 24 hours.
        """)

        # Placeholder for now, as requested
        st.info("ℹ️ To generate a fresh report, run: `python src/analysis/reality_check.py`")

        st.code(
            """
=== REALITY CHECK REPORT: BTC/USDT ===
Time Range: 2025-12-24 to 2025-12-25
Match Rate: 97.58%
Discrepancies: 7
...
        """,
            language="text",
        )

        if st.button("Run Reality Check Now (This may take a minute)"):
            with st.spinner("Running simulation..."):
                # We could run the script via subprocess here
                import subprocess

                try:
                    result = subprocess.run(
                        [sys.executable, "src/analysis/reality_check.py"],
                        capture_output=True,
                        text=True,
                    )
                    st.text_area("Output", result.stdout + result.stderr, height=400)
                except Exception as e:
                    st.error(f"Failed to run script: {e}")


if __name__ == "__main__":
    main()