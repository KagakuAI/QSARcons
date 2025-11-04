from typing import Optional, List
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import RegressorMixin, ClassifierMixin

from .consensus import (
    ConsensusSearchRegressor,
    ConsensusSearchClassifier,
    calc_accuracy,
)

class StackingRegressor(ConsensusSearchRegressor):
    """Stacking regressor: uses **all filtered models** to train a meta-model.

    No consensus size parameter. After model filtering, the meta-model is
    trained on the full prediction matrix of the remaining models.
    """
    def __init__(
        self,
        meta_model: Optional[RegressorMixin] = None,
        metric: str = "mae"
    ):
        super().__init__(cons_size="auto", cons_size_candidates=None, metric=metric)
        self.meta_model = meta_model

    def run(self, x: pd.DataFrame, y: pd.Series) -> pd.Index:
        # Filter weak models first
        x_filtered = self._filter_models(x, y)
        # Train meta-model on remaining models
        self.meta_model.fit(x_filtered, y)
        # Return all models used
        return pd.Index(x_filtered.columns)

    def _consensus_predict(self, x_subset: pd.DataFrame) -> pd.Series:
        preds = self.meta_model.predict(x_subset)
        return pd.Series(preds, index=x_subset.index)


class StackingClassifier(ConsensusSearchClassifier):
    """Stacking classifier: uses **all filtered models** to train a meta-model.

    No consensus size parameter. After filtering, the meta-model predicts
    final class labels.
    """
    def __init__(
        self,
        meta_model: Optional[ClassifierMixin] = None,
        metric: str = "acc"
    ):
        super().__init__(cons_size="auto", cons_size_candidates=None, metric=metric)
        # If user passes a class, instantiate it
        # Instantiate only if a class was passed
        import inspect
        self.meta_model = meta_model() if inspect.isclass(meta_model) else (meta_model or LogisticRegression(max_iter=500))

    def run(self, x: pd.DataFrame, y: pd.Series) -> pd.Index:
        # Filter weak models
        x_filtered = self._filter_models(x, y)
        # Fit meta-model
        self.meta_model.fit(x_filtered, y)
        return pd.Index(x_filtered.columns)

    def _consensus_predict(self, x_subset: pd.DataFrame) -> pd.Series:
        pred = self.meta_model.predict(x_subset)
        return pd.Series(pred, index=x_subset.index)
