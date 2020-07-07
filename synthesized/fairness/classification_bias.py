from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ..testing.experiments import ExperimentalEstimator


class ClassificationBias:
    def __init__(self, df, sensitive_attr, target):
        self.df = df.reset_index(drop=True)
        self.sensitive_attr = sensitive_attr
        self.target = target

        self.ee = ExperimentalEstimator(target=target)

        try:
            train, test = train_test_split(self.df, test_size=0.4, random_state=42,
                                           stratify=self.df[[self.target, self.sensitive_attr]])
        except ValueError:
            train, test = train_test_split(self.df, test_size=0.4, random_state=42)

        self.train_idx = train.index

        self.tests = {'All': test.index}
        for sensitive_value in self.df[sensitive_attr].unique():
            test_idx_i = test[test[sensitive_attr] == sensitive_value].index
            # if len(test_i) > 100:
            self.tests[sensitive_value] = test_idx_i

    def classifier_bias(self, classifiers: Dict[str, BaseEstimator] = None, remove_sensitive: bool = False,
                        plot_results: bool = False):

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
                columns = list(filter(lambda c: c != 'sensitive_attr', self.df.columns))
            else:
                columns = self.df.columns

            results = self.ee.preprocess_classify(self.df[columns], other_dfs=dict(), train_idx=self.train_idx,
                                                  test_idx=test, classifiers=classifiers, copy_clf=False)
            if len(results) == 0:
                continue

            for i, clf in enumerate(results['Classifier'].unique()):
                # ROC Curves
                auc = results.loc[results['Classifier'] == clf, 'AUC Test'].values[0]

                if plot_results:
                    x, y, _ = results.loc[results['Classifier'] == clf, 'ROC Curve'].values[0]
                    axs[i].plot(x, y, label=f"{name} (AUC={auc:.3f})")
                    axs[i].legend(loc='lower right')
                    axs[i].set_title(clf)
                    axs[i].set_xlabel('FPR')
                    axs[i].set_ylabel('FPR')

                # Confusion Matrix Heatmap
                cm = results.loc[results['Classifier'] == clf, 'Confusion Matrix'].values[0]
                cm_norm = cm / np.sum(cm)

                if plot_results:
                    cm_annot = self.get_cm_annot(cm, cm_norm)
                    sns.heatmap(cm_norm, annot=cm_annot, fmt='', vmin=0, vmax=1, annot_kws={"size": 14}, cbar=False,
                                xticklabels=list(self.ee.oh_encoders['income'][0].categories_[0]),
                                yticklabels=list(self.ee.oh_encoders['income'][0].categories_[0]),
                                ax=axs_hm[i][k])

                    axs_hm[i][k].set_title(f"{clf} - {name}")
                    axs_hm[i][k].set_xlabel('Real')
                    axs_hm[i][k].set_ylabel('Predicted')

                # disparate_impact
                predicted_values = results.loc[results['Classifier'] == clf, 'Predicted Values'].values[0]

                classification_bias[name][clf] = (auc, cm_norm, predicted_values)

        if plot_results:
            fig.tight_layout()
            fig_hm.tight_layout()
            fig.savefig(f"/Users/tonbadal/Pictures/fairness/ROC_{self.sensitive_attr}_{remove_sensitive}.png")
            fig_hm.savefig(f"/Users/tonbadal/Pictures/fairness/CM_{self.sensitive_attr}_{remove_sensitive}.png")
            plt.show()

        classification_score = dict()
        for name, name_results in classification_bias.items():
            if name == 'All':
                continue

            auc_diff = cm_diff = di = 0.
            for clf, clf_results in classification_bias[name].items():
                auc_all, cm_all, y_all = classification_bias['All'][clf]
                auc, cm, predicted_values = clf_results

                auc_diff += abs(auc_all - auc)
                cm_diff += np.average(abs(cm_all - cm)) if cm_all.shape == cm.shape else 0.
                di += self.disparate_impact(predicted_values, y_all)

            n_clfs = len(classification_bias[name]) - 1
            auc_diff = auc_diff / n_clfs
            cm_diff = cm_diff / n_clfs
            di = di / n_clfs

            classification_score[name] = (auc_diff + cm_diff + di) / 3

        return np.std(list(classification_score.values()))

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

        return (np.sum(y_sensitive >= 0.5) / len(y_sensitive)) / (np.sum(y_all >= 0.5) / len(y_all))
