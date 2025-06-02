from typing import  Callable
from pandas import DataFrame, Series


class StackingRegressor:
    """
    A wrapper for any scikit-learn compatible regression model to perform stacking.
    """

    def __init__(self, method: Callable = None) -> None:
        self.method = method

    def run(self, x: DataFrame, y: Series) -> "StackingRegressor":
        self.method.fit(x, y)
        return self

    def predict(self, x: DataFrame) -> Series:
        return self.method.predict(x)