import logging
from typing import Any, Dict, List, Union, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ..modelling import ModellingPreprocessor
from ..metrics.modelling_metrics import classifier_scores_from_df
from .sensitive_attributes import sensitive_attr_concat_name

logger = logging.getLogger(__name__)


class ClassificationBias:
    """Compute classification bias from given Dataframe, a list of sensitive attributes, and a target variable. It will
    be calculated by computing the difference of ROC AUC, Confusion Matrix, and Disparate Imact for the entire dataset
    and all unique values grouped by sensitive attributes.

    """
    def __init__(self, df, sensitive_attrs: Union[str, List[str]], target, min_count: Optional[int] = 50):
        """Classification Bias Constructor.

        Args:
            df: Dataframe to compute bias.
            sensitive_attrs: Sensitive attributes.
            target: Target variable
            min_count: Minimum sample size to compute classification tasks.

        """
        self.df = df.reset_index(drop=True)
        self.sensitive_attrs: List[str] = sensitive_attrs if isinstance(sensitive_attrs, list) else [sensitive_attrs]
        self.sensitive_attrs_name: str = sensitive_attr_concat_name(sensitive_attrs)
        self.target = target
        self.sensitive_attrs_and_target = np.concatenate((self.sensitive_attrs, [self.target]))

        self.preprocessor = ModellingPreprocessor(target=target)
        self.df_pre = self.preprocessor.fit_transform(self.df)

        try:
            train, test = train_test_split(self.df, test_size=0.4, random_state=42,
                                           stratify=self.df[self.sensitive_attrs_and_target])
        except ValueError:
            train, test = train_test_split(self.df, test_size=0.4, random_state=42)

        self.train_idx = train.index

        self.tests = {'All': test.index}
        for sensitive_values, indices in df.groupby(self.sensitive_attrs).indices.items():
            if min_count is not None and len(indices) < min_count:
                continue

            self.tests[sensitive_values] = indices

        if len(self.tests) == 1:
            logger.info(f"No sensitive populations found for sensitive_attrs='{self.sensitive_attrs_name}'.")

    def classifier_bias(self, threshold: float = 0.025, classifiers: Dict[str, BaseEstimator] = None,
                        remove_sensitive: bool = False, plot_results: bool = False,
                        plot_file_name: str = None) -> Tuple[float, List[Dict[str, Any]]]:

        if len(self.tests) == 1:
            return 0, []

        if classifiers is None:
            classifiers = {
                'LogisticRegression': LogisticRegression(max_iter=500),
                'RandomForest': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=1000)
            }

        if plot_results:
            fig, axs = plt.subplots(len(classifiers), 1, figsize=(6, 6 * len(classifiers)))
            fig_hm, axs_hm = plt.subplots(len(classifiers), len(self.tests),
                                          figsize=(4 * len(self.tests), 3 * len(classifiers)))

        classification_bias: Dict[str, Dict[str, Any]] = dict()

        for k, (name, test) in enumerate(self.tests.items()):
            classification_bias[name] = dict()

            # Remove sensitive column from columns?
            if remove_sensitive:
                columns = list(filter(lambda c: c not in self.sensitive_attrs, self.df.columns))
            else:
                columns = self.df.columns
            columns_pre = np.concatenate([self.preprocessor.columns_mapping[c] for c in columns])
            metrics_to_compute = ['roc_auc', 'roc_curve', 'confusion_matrix']

            results = []
            for clf_name, clf in classifiers.items():

                results_i = classifier_scores_from_df(
                    df_train=self.df_pre.iloc[self.train_idx][columns_pre],
                    df_test=self.df_pre.iloc[test][columns_pre],
                    target=self.target,
                    clf=clf,
                    metrics=metrics_to_compute,
                    return_predicted=True
                )
                results_i['clf_name'] = clf_name
                results.append(results_i)
            df_results = pd.DataFrame(results)

            if len(results) == 0:
                continue

            for i, clf_name in enumerate(df_results['clf_name'].unique()):
                # ROC Curves
                auc = df_results.loc[df_results['clf_name'] == clf_name, 'roc_auc'].values[0]

                if plot_results:
                    x, y, _ = df_results.loc[df_results['clf_name'] == clf_name, 'roc_curve'].values[0]
                    axs[i].plot(x, y, label=f"{name} (AUC={auc:.3f})")
                    axs[i].legend(loc='lower right')
                    axs[i].set_title(clf_name)
                    axs[i].set_xlabel('FPR')
                    axs[i].set_ylabel('FPR')

                # Confusion Matrix Heatmap
                cm = df_results.loc[df_results['clf_name'] == clf_name, 'confusion_matrix'].values[0]
                cm_norm = cm / np.sum(cm)

                if plot_results:
                    oh_encoder = self.preprocessor.column_encoders.get(self.target)
                    xticklabels = list(oh_encoder[0].categories_[0]) if oh_encoder is not None else None
                    yticklabels = list(oh_encoder[0].categories_[0]) if oh_encoder is not None else None

                    cm_annot = self.get_cm_annot(cm, cm_norm)
                    sns.heatmap(cm_norm, annot=cm_annot, fmt='', vmin=0, vmax=1, annot_kws={"size": 14}, cbar=False,
                                xticklabels=xticklabels, yticklabels=yticklabels, ax=axs_hm[i][k])

                    axs_hm[i][k].set_title(f"{clf_name} - {name}")
                    axs_hm[i][k].set_xlabel('Real')
                    axs_hm[i][k].set_ylabel('Predicted')

                # disparate_impact
                predicted_values = df_results.loc[df_results['clf_name'] == clf_name, 'predicted_values'].values[0]
                classification_bias[name][clf_name] = (auc, cm_norm, predicted_values)

        if plot_results:
            fig.tight_layout()
            fig_hm.tight_layout()
            if plot_file_name is None:
                fig.savefig(f"ROC_{self.sensitive_attrs_name}_{remove_sensitive}.png")
                fig_hm.savefig(f"CM_{self.sensitive_attrs_name}_{remove_sensitive}.png")
            else:
                fig.savefig(f"ROC_{plot_file_name}")
                fig_hm.savefig(f"CM_{plot_file_name}")
            plt.show()

        classification_score = dict()
        for name, name_results in classification_bias.items():
            if name == 'All':
                continue

            auc_diff = cm_diff = di = 0.
            for clf_name, clf_results in classification_bias[name].items():
                auc_all, cm_all, y_all = classification_bias['All'][clf_name]
                auc, cm, predicted_values = clf_results

                auc_diff += abs(auc_all - auc)
                cm_diff += np.average(abs(cm_all - cm)) if cm_all.shape == cm.shape else 0.
                di += self.disparate_impact(predicted_values, y_all)

            n_clfs = len(classification_bias[name]) - 1
            auc_diff = auc_diff / n_clfs
            cm_diff = cm_diff / n_clfs
            di = di / n_clfs

            classification_score[name] = np.nanmean((auc_diff, cm_diff, di))

        score_mean = np.nanmean(list(classification_score.values()))
        biases: List[Dict[str, Any]] = []
        for name, score in classification_score.items():
            distance = abs(score_mean - score)
            if distance > threshold:
                biases.append({
                    'name': self.sensitive_attrs_name,
                    'value': name,
                    'distance': distance,
                    'count': len(self.tests[name])
                })

        return np.std(list(classification_score.values())), biases

    @staticmethod
    def get_cm_annot(cm: np.array, cm_norm: np.array = None) -> List[List[str]]:
        if cm_norm is None:
            cm_norm = cm / np.sum(cm)

        cm_annot = []
        for i in range(len(cm)):
            row = []
            for j in range(len(cm[i])):
                row.append(f"{cm[i, j]} ({cm_norm[i, j] * 100:.2f}%)")
            cm_annot.append(row)

        return cm_annot

    @staticmethod
    def disparate_impact(y_sensitive: Union[List[float], np.array], y_all: Union[List[float], np.array]) -> float:
        if not isinstance(y_sensitive, np.ndarray):
            y_sensitive = np.array(y_sensitive)
        if not isinstance(y_all, np.ndarray):
            y_all = np.array(y_all)

        positives_sensitive = np.sum(y_sensitive >= 0.5)
        positives_all = np.sum(y_all >= 0.5)
        if positives_sensitive == 0 and positives_all == 0:
            di = 1.
        elif positives_sensitive != 0 and positives_all != 0:
            di = (np.sum(y_sensitive >= 0.5) / len(y_sensitive)) / (np.sum(y_all >= 0.5) / len(y_all))
        else:
            di = 0.

        return max(1, abs(1 - di))
