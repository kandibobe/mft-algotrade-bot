#!/usr/bin/env python3
"""
Mock Exchange Server for Integration Testing
=============================================

Simulates exchange API endpoints for testing without real API calls.

Endpoints:
- /health - Health check
- /api/v1/ticker/{symbol} - Price ticker
- /api/v1/orderbook/{symbol} - Order book
- /api/v1/trades/{symbol} - Recent trades
- /api/v1/order - Place order (POST)
- /api/v1/balance - Account balance

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import random
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Mock Exchange API",
    description="Simulated exchange for testing",
    version="1.0.0"
)


# =============================================================================
# Data Models
# =============================================================================

class OrderRequest(BaseModel):
    symbol: str
    side: str  # 'buy' or 'sell'
    type: str  # 'limit' or 'market'
    quantity: float
    price: float | None = None


class OrderResponse(BaseModel):
    order_id: str
    symbol: str
    side: str
    type: str
    quantity: float
    price: float
    status: str
    timestamp: str


class TickerResponse(BaseModel):
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    change_24h: float
    timestamp: str


class BalanceResponse(BaseModel):
    currency: str
    available: float
    locked: float
    total: float


# =============================================================================
# Mock Data Storage
# =============================================================================

class MockDataStore:
    def __init__(self):
        self.base_prices = {
            "BTC/USDT": 95000.0,
            "ETH/USDT": 3500.0,
            "SOL/USDT": 200.0,
            "BNB/USDT": 600.0,
            "XRP/USDT": 2.20,
            "ADA/USDT": 1.00,
            "DOGE/USDT": 0.40,
            "AVAX/USDT": 45.0,
            "DOT/USDT": 8.5,
            "LINK/USDT": 25.0,
        }
        self.balances = {
            "USDT": {"available": 10000.0, "locked": 0.0},
            "BTC": {"available": 0.1, "locked": 0.0},
            "ETH": {"available": 1.0, "locked": 0.0},
        }
        self.orders: list[dict] = []
        self.order_counter = 0

    def get_price(self, symbol: str) -> float:
        """Get simulated price with random fluctuation."""
        base = self.base_prices.get(symbol, 100.0)
        # Add ±0.5% random fluctuation
        fluctuation = random.uniform(-0.005, 0.005)
        return round(base * (1 + fluctuation), 8)

    def create_order(self, order: OrderRequest) -> dict:
        """Create a mock order."""
        self.order_counter += 1
        price = order.price if order.price else self.get_price(order.symbol)

        order_data = {
            "order_id": f"MOCK-{self.order_counter:08d}",
            "symbol": order.symbol,
            "side": order.side,
            "type": order.type,
            "quantity": order.quantity,
            "price": price,
            "status": "filled" if order.type == "market" else "open",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        self.orders.append(order_data)
        return order_data


store = MockDataStore()


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "mock-exchange",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/api/v1/ticker/{symbol}", response_model=TickerResponse)
async def get_ticker(symbol: str):
    """Get price ticker for a symbol."""
    symbol = symbol.replace("-", "/").upper()

    if symbol not in store.base_prices:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    price = store.get_price(symbol)
    spread = price * 0.0001  # 0.01% spread

    return TickerResponse(
        symbol=symbol,
        bid=round(price - spread, 8),
        ask=round(price + spread, 8),
        last=price,
        volume_24h=random.uniform(1000000, 10000000),
        change_24h=random.uniform(-5, 5),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.get("/api/v1/orderbook/{symbol}")
async def get_orderbook(symbol: str, depth: int = 20):
    """Get order book for a symbol."""
    symbol = symbol.replace("-", "/").upper()

    if symbol not in store.base_prices:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    price = store.get_price(symbol)

    # Generate fake order book
    bids = []
    asks = []

    for i in range(depth):
        bid_price = price * (1 - 0.0001 * (i + 1))
        ask_price = price * (1 + 0.0001 * (i + 1))
        quantity = random.uniform(0.1, 10)

        bids.append([round(bid_price, 8), round(quantity, 8)])
        asks.append([round(ask_price, 8), round(quantity, 8)])

    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/api/v1/trades/{symbol}")
async def get_trades(symbol: str, limit: int = 100):
    """Get recent trades for a symbol."""
    symbol = symbol.replace("-", "/").upper()

    if symbol not in store.base_prices:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    trades = []
    base_price = store.get_price(symbol)

    for i in range(min(limit, 100)):
        price = base_price * random.uniform(0.999, 1.001)
        trades.append({
            "id": f"T{1000000 + i}",
            "price": round(price, 8),
            "quantity": round(random.uniform(0.001, 1), 8),
            "side": random.choice(["buy", "sell"]),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    return {
        "symbol": symbol,
        "trades": trades
    }


@app.post("/api/v1/order", response_model=OrderResponse)
async def create_order(order: OrderRequest):
    """Place a new order."""
    order.symbol = order.symbol.replace("-", "/").upper()

    if order.symbol not in store.base_prices:
        raise HTTPException(status_code=404, detail=f"Symbol {order.symbol} not found")

    if order.side not in ["buy", "sell"]:
        raise HTTPException(status_code=400, detail="Invalid side. Use 'buy' or 'sell'")

    if order.type not in ["limit", "market"]:
        raise HTTPException(status_code=400, detail="Invalid type. Use 'limit' or 'market'")

    if order.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")

    order_data = store.create_order(order)
    return OrderResponse(**order_data)


@app.get("/api/v1/balance")
async def get_balance():
    """Get account balances."""
    balances = []
    for currency, data in store.balances.items():
        balances.append(BalanceResponse(
            currency=currency,
            available=data["available"],
            locked=data["locked"],
            total=data["available"] + data["locked"]
        ))
    return {"balances": balances}


@app.get("/api/v1/orders")
async def get_orders(symbol: str | None = None, status: str | None = None):
    """Get order history."""
    orders = store.orders

    if symbol:
        symbol = symbol.replace("-", "/").upper()
        orders = [o for o in orders if o["symbol"] == symbol]

    if status:
        orders = [o for o in orders if o["status"] == status]

    return {"orders": orders[-100:]}  # Return last 100 orders


@app.delete("/api/v1/order/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order."""
    for order in store.orders:
        if order["order_id"] == order_id:
            if order["status"] == "open":
                order["status"] = "cancelled"
                return {"success": True, "order": order}
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel order with status: {order['status']}"
                )

    raise HTTPException(status_code=404, detail=f"Order {order_id} not found")


# =============================================================================
# WebSocket Simulation (for future use)
# =============================================================================

@app.websocket("/ws/ticker/{symbol}")
async def websocket_ticker(websocket, symbol: str):
    """WebSocket endpoint for real-time price updates."""
    await websocket.accept()
    symbol = symbol.replace("-", "/").upper()

    try:
        while True:
            price = store.get_price(symbol)
            await websocket.send_json({
                "type": "ticker",
                "symbol": symbol,
                "price": price,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
            await asyncio.sleep(1)  # Update every second
    except Exception:
        pass


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Starting Mock Exchange Server...")
    print("Available symbols:", list(store.base_prices.keys()))
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
