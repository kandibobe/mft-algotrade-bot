import os
import sys
import argparse
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Добавляем путь к src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.logger import setup_structured_logging, get_logger
from src.ml.training.orchestrator import TrainingOrchestrator, OrchestratorConfig
from src.backtesting.wfo_engine import WalkForwardEngine
from src.backtesting.vectorized_backtester import BacktestConfig
from src.config.unified_config import load_config
from src.data.loader import get_ohlcv

# Initialize logging
setup_structured_logging(level="INFO")
logger = get_logger("full_cycle_launch")

class FullCycleLauncher:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        
        # Create OrchestratorConfig from TradingConfig
        orch_config = OrchestratorConfig(
            model_registry_dir=str(self.config.paths.models_dir / "registry")
        )
        self.orchestrator = TrainingOrchestrator(orch_config)
        
        # Create BacktestConfig from TradingConfig
        backtest_config = BacktestConfig(
            initial_capital=10000.0, # Default for WFO
            fee_rate=self.config.strategy.fee if self.config.strategy else 0.001,
            slippage_rate=self.config.strategy.slippage_entry if self.config.strategy else 0.001,
            take_profit=self.config.risk.take_profit_pct,
            stop_loss=self.config.risk.stop_loss_pct,
            position_size_pct=self.config.risk.max_position_pct
        )
        self.wfo_engine = WalkForwardEngine(self.config, backtest_config)

    async def run_wfo(self, strategy_name: str, pair: str, timerange: str):
        logger.info(f"Starting WFO for {strategy_name} on {pair} for {timerange}...")
        
        # Load data for WFO
        # Assuming 5m timeframe for WFO as per typical usage
        data = get_ohlcv(pair, timeframe="5m")
        
        # Filter data by timerange if needed, but for now we pass all loaded data
        # Real implementation would parse timerange string (YYYYMMDD-YYYYMMDD)
        
        # Note: WFOEngine run is synchronous (not async) based on its definition
        results = self.wfo_engine.run(
            data=data,
            pair=pair
        )
        logger.info(f"WFO completed.")
        return results

    async def train_models(self, strategy_name: str, pair: str):
        logger.info(f"Starting model training for {strategy_name} on {pair}...")
        # Используем оркестратор для обучения
        training_results = await self.orchestrator.run_training_pipeline(
            pair=pair,
            strategy=strategy_name
        )
        logger.info(f"Training completed. Model registered: {training_results.get('model_id')}")
        return training_results

    async def validate_and_deploy(self, training_results: Dict[str, Any]):
        logger.info("Validating and deploying models...")
        # Логика валидации (проверка метрик)
        metrics = training_results.get('metrics', {})
        precision = metrics.get('precision', 0)
        
        if precision > 0.52: # Пример порога
            logger.info(f"Model validated with precision {precision}. Deploying...")
            # В реальности оркестратор уже мог зарегистрировать модель, 
            # здесь мы можем пометить её как 'production_ready'
            return True
        else:
            logger.warning(f"Model precision {precision} too low. Deployment skipped.")
            return False

    async def execute_full_cycle(self, args):
        start_time = datetime.now()
        logger.info(f"Full cycle launch started at {start_time}")
        
        try:
            # 1. WFO
            wfo_results = await self.run_wfo(args.strategy, args.pair, args.timerange)
            
            # 2. Training
            training_results = await self.train_models(args.strategy, args.pair)
            
            # 3. Validation & Deployment
            success = await self.validate_and_deploy(training_results)
            
            if success:
                logger.info("Full cycle completed successfully.")
            else:
                logger.error("Full cycle failed at validation stage.")
                
        except Exception as e:
            logger.error(f"Error during full cycle: {e}", exc_info=True)
        finally:
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"Full cycle finished in {duration}")

def main():
    parser = argparse.ArgumentParser(description="Stoic Citadel Full Cycle Launcher")
    parser.add_argument("--config", type=str, default="user_data/config/config_dryrun.json", help="Path to config file")
    parser.add_argument("--strategy", type=str, default="StoicEnsembleStrategyV6", help="Strategy name")
    parser.add_argument("--pair", type=str, default="BTC/USDT:USDT", help="Trading pair")
    parser.add_argument("--timerange", type=str, default="20240101-20251231", help="Timerange for WFO")
    
    args = parser.parse_args()

    launcher = FullCycleLauncher(args.config)
    asyncio.run(launcher.execute_full_cycle(args))

if __name__ == "__main__":
    main()