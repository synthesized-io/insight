from typing import List

import pandas as pd

from ..insight.metrics import predictive_modelling_score as _predictive_modelling_score
from ..insight.metrics import predictive_modelling_comparison as _predictive_modelling_comparison


def predictive_modelling_score(data: pd.DataFrame, y_label: str, x_labels: List[str], model: str):
    """Calculates an R2 or ROC AUC score for a dataset for a given model and labels.

    This function will fit a regressor or classifier depending on the datatype of the y_label. All necessary
    preprocessing (standard scaling, one-hot encoding) is done in the function.

    Args:
        data: The input dataset.
        y_label: The name of the target variable column/response variable.
        x_labels: A list of the input column names/explanatory variables.
        model: One of 'Linear', 'GradientBoosting', 'RandomForrest', 'MLP', 'LinearSVM', or 'Logistic'. Note that
            'Logistic' only applies to categorical response variables.

    Returns:
        The score, metric ('r2' or 'roc_auc'), and the task ('regression', 'binary', or 'multinomial')
    """

    return _predictive_modelling_score(df=data, model=model, y_label=y_label, x_labels=x_labels)


def predictive_modelling_comparison(data: pd.DataFrame, synth_data: pd.DataFrame,
                                    y_label: str, x_labels: List[str], model: str):

    return _predictive_modelling_comparison(
        df_old=data, df_new=synth_data, model=model, y_label=y_label, x_labels=x_labels
    )
