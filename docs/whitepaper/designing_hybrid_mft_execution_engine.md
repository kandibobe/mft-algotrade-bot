# Designing a Hybrid Async/Sync MFT Execution Engine in Python

*A deep dive into building Stoic Citadel: overcoming Freqtrade limitations with a dual-layer architecture.*

---

## The Problem: When "Fast Enough" Isn't Enough

In the world of Mid-Frequency Trading (MFT), the gap between signal generation and order execution is where profit evaporates.

Like many algorithmic traders, we started with **Freqtrade**. It's an excellent framework—robust, feature-rich, and perfect for strategy development. However, as we scaled towards higher frequency strategies, we hit a hard ceiling: **Synchronous Execution Latency**.

Freqtrade's standard loop operates sequentially:
1. Fetch Candles -> 2. Calculate Indicators -> 3. Check Signals -> 4. Place Orders.

This cycle, while safe, introduces milliseconds (or sometimes seconds) of delay. In volatile markets, that's the difference between a profitable fill and a slippage-induced loss. We needed the rich ecosystem of Freqtrade for strategy logic but the raw speed of an async engine for execution.

## The Solution: A Hybrid Async/Sync Architecture

We didn't want to rewrite the wheel. Instead, we built a **Hybrid Architecture** that splits the system into two specialized layers operating in parallel.

### 1. The Macro Layer (The Brain)
*   **Technology:** Synchronous Python (Freqtrade based)
*   **Role:** Heavy lifting. It handles data ingestion, complex feature engineering, machine learning inference, and broad market analysis.
*   **Output:** It doesn't place orders. It generates *intent*.

### 2. The Micro Layer (The Muscle)
*   **Technology:** `asyncio` + WebSockets
*   **Role:** Speed. It maintains real-time WebSocket connections to exchanges, aggregates order book data, and executes orders.
*   **Key Component:** `SmartOrderExecutor` implementing algorithms like `ChaseLimit` to work the order book dynamically.

### 3. The Bridge
The magic happens in the `HybridConnector`. It acts as the thread-safe interface between the synchronous strategy loop and the asynchronous execution event loop.

```mermaid
graph TD
    subgraph Macro Layer (Strategy)
        Strategy[Freqtrade Strategy]
        ML[ML Pipeline]
        Strategy --> ML
    end

    subgraph Bridge
        Connector[HybridConnector]
    end

    subgraph Micro Layer (Execution)
        Aggregator[WS Aggregator]
        Risk[Risk Manager]
        Executor[SmartOrderExecutor]
        
        Aggregator --> Risk
        Risk --> Executor
    end

    ML -- "Signal Intent" --> Connector
    Connector -- "Execution Task" --> Executor
    Aggregator -- "Real-time Orderbook" --> Connector
    Connector --> Strategy
```

## Key Technical Innovations

### Asynchronous Websocket Aggregator
Instead of polling REST APIs, our `Aggregator` (`src/websocket/aggregator.py`) maintains live WebSocket streams. It normalizes data from multiple exchanges into a unified internal format, providing the execution layer with the latest bid/ask spread and depth updates instantly.

### Smart Order Execution
We moved away from "Fire and Forget" market orders. Our `SmartOrderExecutor` uses a **ChaseLimit** algorithm:
1.  Place a Limit order at the best bid/ask.
2.  Watch the order book in real-time.
3.  If the price moves away, update the order price (within risk limits) to "chase" the fill.
4.  This reduces slippage significantly compared to market orders, while ensuring fills better than static limit orders.

### The Safety Net: Risk Manager
Speed is dangerous without brakes. Every transaction request from the Macro layer must pass through the `RiskManager` (`src/risk/risk_manager.py`). This component performs pre-trade checks:
*   **Circuit Breakers:** Stops trading if drawdown exceeds thresholds.
*   **Position Sizing:** Ensures no single trade exceeds risk limits (Kelly Criterion/Fixed Fraction).
*   **Liquidation Protection:** Checks distance to liquidation price.

## Results

By decoupling strategy from execution, we achieved:
*   **Latency Reduction:** Execution decision time dropped from ~500ms (poll-based) to <50ms (websocket-event based).
*   **Better Fills:** The ChaseLimit logic consistently saves 0.05% - 0.1% per trade in slippage.
*   **Reliability:** The strategy layer can crash or hang without affecting the management of open orders in the async layer.

## Conclusion

Python is often criticized for being "slow" for trading, but that's a misunderstanding of the tool. By leveraging `asyncio` for I/O-bound tasks (execution) and standard sync code for CPU-bound tasks (analysis), you can build systems that rival C++ in practical MFT performance while maintaining the development speed of Python.

---
*Stoic Citadel is an open-source project demonstrating these principles. Check out the code to see `HybridConnector` in action.*