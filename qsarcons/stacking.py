from typing import Optional, List
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from sklearn.base import RegressorMixin
from qsarcons.consensus import ConsensusSearch, detect_task_type

from .hopt import StepwiseHopt, DEFAULT_PARAM_GRID_REGRESSORS, DEFAULT_PARAM_GRID_CLASSIFIERS
from qsarcons.logging import OutputSuppressor

class StackingAggregation(ConsensusSearch):
    """Stacking regressor: uses **all filtered models** to train a meta-model"""
    def __init__(
        self,
        meta_model: Optional[RegressorMixin] = None,
        hopt: bool = True,
    ):
        super().__init__()
        self.meta_model = meta_model
        self.hopt = hopt
        self._task_type = None

    def run(self, x: pd.DataFrame, y: pd.Series) -> List:

        self._task_type = detect_task_type(y)

        # 1. Filter weak models
        x_filtered = self._filter_models(x, y)

        # 2. Hyperparameters optimization
        if self.hopt:
            est_name = self.meta_model.__class__.__name__
            task_type = type_of_target(y)
            is_classification = task_type in ["binary", "multiclass"]

            param_grid = (
                DEFAULT_PARAM_GRID_CLASSIFIERS.get(est_name)
                if is_classification
                else DEFAULT_PARAM_GRID_REGRESSORS.get(est_name)
            )

            scoring = "roc_auc" if is_classification else "r2"

            stepwise_hopt = StepwiseHopt(self.meta_model, param_grid, scoring=scoring, verbose=False)
            stepwise_hopt.fit(x_filtered, y)
            self.meta_model = stepwise_hopt.estimator

        # 3. Train final meta-model
        with OutputSuppressor() as logger:
            self.meta_model.fit(x_filtered, y)

        # 3. Return filtered models used
        return list(x_filtered.columns)

    def predict(self, x_subset: pd.DataFrame) -> List:
        y_pred = list(self.meta_model.predict(x_subset))
        return y_pred

    def predict_proba(self, x_subset: pd.DataFrame) -> List:
        y_pred = list(self.meta_model.predict_proba(x_subset))
        return y_pred