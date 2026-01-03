"""
Portfolio Rebalancer
====================

Implements dynamic portfolio rebalancing based on a target allocation strategy.
"""

import logging
from typing import Dict, List
import pandas as pd

from src.risk.hrp import get_hrp_weights
from src.order_manager.smart_order import SmartOrder, OrderSide

logger = logging.getLogger(__name__)

class Rebalancer:
    """
    Dynamically rebalances the portfolio to match a target allocation.
    """

    def __init__(self, risk_manager, order_executor, rebalancing_threshold=0.01):
        self.risk_manager = risk_manager
        self.order_executor = order_executor
        self.rebalancing_threshold = rebalancing_threshold

    def run(self, prices: pd.DataFrame):
        """
        Run the rebalancing logic.

        Args:
            prices: DataFrame of historical prices for all assets in the portfolio.
        """
        current_portfolio = self.risk_manager.get_status()['positions']
        account_balance = self.risk_manager.get_status()['account_balance']

        if not current_portfolio:
            logger.info("No open positions to rebalance.")
            return

        target_weights = get_hrp_weights(prices)
        
        current_weights = {
            symbol: pos['value'] / account_balance for symbol, pos in current_portfolio.items()
        }

        trades_to_execute = self._calculate_rebalancing_trades(current_weights, target_weights, account_balance)

        for trade in trades_to_execute:
            self.order_executor.submit_order(trade)

    def _calculate_rebalancing_trades(self, current_weights: Dict[str, float], target_weights: Dict[str, float], account_balance: float) -> List[SmartOrder]:
        """
        Calculate the trades needed to rebalance the portfolio.
        """
        trades = []
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > self.rebalancing_threshold:
                trade_value = weight_diff * account_balance
                
                # For simplicity, we'll assume we can get the current price from the RiskManager
                current_price = self.risk_manager.get_status()['positions'].get(symbol, {}).get('current_price')
                if not current_price:
                    logger.warning(f"No current price for {symbol}, cannot create rebalancing trade.")
                    continue

                trade_quantity = trade_value / current_price
                
                if trade_value > 0: # We need to buy
                    order = SmartOrder(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=abs(trade_quantity),
                        price=current_price # This should be a limit price
                    )
                    trades.append(order)
                else: # We need to sell
                    order = SmartOrder(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=abs(trade_quantity),
                        price=current_price
                    )
                    trades.append(order)
                    
        return trades
