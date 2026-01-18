"""
Stoic Citadel - ML Strategy Adapter
===================================

Decouples ML model inference and feature engineering from the Strategy class.
"""

import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.ml.meta_labeling import MetaModel
from src.ml.model_loader import get_production_model

logger = logging.getLogger(__name__)

class StrategyMLAdapter:
    """
    Adapter to handle ML operations for a strategy.
    """

    def __init__(self, pair: str):
        self.pair = pair
        self.model = None
        self.meta_model = None
        self.feature_engine = None
        self.feature_names = []
        self._load_model()

    def _load_model(self):
        try:
            # Main model loading
            self.model, self.feature_engine, self.feature_names = get_production_model(self.pair)

            # Meta-model loading (optional)
            try:
                # Assuming meta-model is stored with a '_meta' suffix or similar logic
                meta_obj, _, _ = get_production_model(f"{self.pair}_meta")
                self.meta_model = MetaModel(meta_obj)
                logger.info(f"Loaded Meta-Model for {self.pair}")
            except Exception:
                logger.debug(f"No Meta-Model found for {self.pair}, using default (0.5)")
                self.meta_model = MetaModel(None)

        except Exception as e:
            logger.error(f"Failed to load ML model for {self.pair}: {e}")

    def get_predictions(self, dataframe: DataFrame) -> pd.Series:
        if not self.model or not self.feature_engine:
            return pd.Series(0.5, index=dataframe.index)

        try:
            df_fe = dataframe.copy()
            if 'date' in df_fe.columns and not isinstance(df_fe.index, pd.DatetimeIndex):
                df_fe.set_index('date', inplace=True)

            X = self.feature_engine.transform(df_fe)

            if hasattr(self.model, 'predict_proba'):
                preds = self.model.predict_proba(X)[:, 1]
            else:
                preds = self.model.predict(X)

            pred_series = pd.Series(preds, index=X.index)

            if isinstance(dataframe.index, pd.DatetimeIndex):
                aligned_preds = pred_series.reindex(dataframe.index, fill_value=0.5)
            else:
                if 'date' in dataframe.columns:
                    temp_df = dataframe.set_index('date')
                    aligned_preds = pred_series.reindex(temp_df.index, fill_value=0.5)
                    aligned_preds = aligned_preds.values
                else:
                    if len(preds) < len(dataframe):
                        padding = np.full(len(dataframe) - len(preds), 0.5)
                        aligned_preds = np.concatenate([padding, preds])
                    else:
                        aligned_preds = preds[-len(dataframe):]
                    aligned_preds = pd.Series(aligned_preds, index=dataframe.index)

            return aligned_preds

        except Exception as e:
            logger.warning(f"ML Prediction failed for {self.pair}: {e}")
            return pd.Series(0.5, index=dataframe.index)

    def get_meta_predictions(self, dataframe: DataFrame) -> pd.Series:
        """
        Get success probability predictions from the meta-model.
        """
        if not self.meta_model or not self.feature_engine:
            return pd.Series(0.5, index=dataframe.index)

        try:
            df_fe = dataframe.copy()
            if 'date' in df_fe.columns and not isinstance(df_fe.index, pd.DatetimeIndex):
                df_fe.set_index('date', inplace=True)

            X = self.feature_engine.transform(df_fe)
            preds = self.meta_model.predict_proba(X)

            pred_series = pd.Series(preds, index=X.index)

            if isinstance(dataframe.index, pd.DatetimeIndex):
                aligned_preds = pred_series.reindex(dataframe.index, fill_value=0.5)
            else:
                if 'date' in dataframe.columns:
                    temp_df = dataframe.set_index('date')
                    aligned_preds = pred_series.reindex(temp_df.index, fill_value=0.5)
                    aligned_preds = aligned_preds.values
                else:
                    if len(preds) < len(dataframe):
                        padding = np.full(len(dataframe) - len(preds), 0.5)
                        aligned_preds = np.concatenate([padding, preds])
                    else:
                        aligned_preds = preds[-len(dataframe):]
                    aligned_preds = pd.Series(aligned_preds, index=dataframe.index)

            return aligned_preds

        except Exception as e:
            logger.warning(f"Meta-ML Prediction failed for {self.pair}: {e}")
            return pd.Series(0.5, index=dataframe.index)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state
        # but they will need reloading on the worker side.
