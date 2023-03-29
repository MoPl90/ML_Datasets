import numpy as np
from typing import List


def rMAE(y_true: List[float], y_pred: List[float]) -> float:
    """Relative mean absolute iRT error

    Args:
        y_true (List[float]): Ground truth iRTs
        y_pred (List[float]): Predicted iRTs

    Returns:
        float: metric value
    """

    val = (
        np.sum(np.abs(y_true - y_pred))
        / len(y_pred)
        / (np.max(y_true) - np.min(y_true))
        * 100
    )

    # name, value, is higher better?
    return ("rMAE", val, False)


def rMSE(y_true: List[float], y_pred: List[float]) -> float:
    """Relative mean squared iRT error

    Args:
        y_true (List[float]): Ground truth iRTs
        y_pred (List[float]): Predicted iRTs

    Returns:
        float: metric value
    """
    val = (
        np.sqrt(np.sum((y_true - y_pred) ** 2))
        / len(y_pred)
        / (np.max(y_true) - np.min(y_true))
        * 100
    )

    # name, value, is higher better?
    return ("rMSE", val, False)
