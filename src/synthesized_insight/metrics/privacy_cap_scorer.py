from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class CAPScorer(ABC):
    """Calculates the privacy score using CAP (Correct Attribution Probability) model
<<<<<<< HEAD
=======

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
    Fits the model by mapping the predictors keys to the sensitive values using the synthetic data.
    These mappings are stored in predictors_to_sensitive_val_map dictionary.
    Then, evaluates the privacy of the original dataset using specific implementation of score(..) method
    in the child classes.
    """

    def __init__(self) -> None:
        self.predictors_to_sensitive_val_map: Dict[Tuple, List[Any]] = {}

    def fit(self, predictors_data: pd.DataFrame, sensitive_data: pd.Series):
        """Fit the privacy scorer on the synthetic data.
<<<<<<< HEAD
=======

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
        Go through each row of the predictors_data and map the tuple of the predictor values to
        corresponding sensitive column value. Tuple of the predictor values is the key for the dictionary.
        If multiple rows contain same predictor tuple, then add their corresponding sensitive atribute value
        to the list of sensitive values corresponding to the key.
<<<<<<< HEAD
        Args:
            predictors_data: Values of the predictor columns
            sensitive_data: Values of the sensitive column
=======

        Args:
            predictors_data: Values of the predictor columns
            sensitive_data: Values of the sensitive column

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
        Returns:
            self
        """
        for (pval, sval) in zip(predictors_data.values, sensitive_data.values):
            if sval == np.nan:
                continue

            key = tuple(pval)
            if key in self.predictors_to_sensitive_val_map:
                self.predictors_to_sensitive_val_map[key].append(sval)
            else:
                self.predictors_to_sensitive_val_map[key] = [sval]
        return self

    @abstractmethod
    def score(self, predictors_data: pd.DataFrame, true_senstive_vals: pd.Series) -> float:
        """Score the privacy of the sensitive column of the original df using synthetic data
<<<<<<< HEAD
        Args:
            predictors_data: Values of predictor columns in original df
            true_senstive_vals: Values of the sensitive column in original df
=======

        Args:
            predictors_data: Values of predictor columns in original df
            true_senstive_vals: Values of the sensitive column in original df

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
        Returns:
            Privacy score between 0 and 1, 0 means no privacy and 1 means absolute privacy
        """
        pass


def get_frequency(samples, element) -> float:
    """Find the frequency of the given element in given samples
<<<<<<< HEAD
    Args:
        samples: List of elements
        element: Element whose frequency needs to be determined
=======

    Args:
        samples: List of elements
        element: Element whose frequency needs to be determined

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
    Returns:
        Frequency of the element
    """
    return samples.count(element) / len(samples)


def get_most_frequent_element(elements):
    """Find the element with the highest frequency
<<<<<<< HEAD
    Args:
        elements: List of element
=======

    Args:
        elements: List of element

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
    Returns:
        The element with highest frequency
    """
    freq_map = defaultdict(int)
    most_freq_elem = None
    highest_freq = 0
    for elem in elements:
        if elem is None:
            continue

        freq = freq_map[elem] + 1
        freq_map[elem] = freq
        if freq > highest_freq:
            highest_freq = freq
            most_freq_elem = elem

    return most_freq_elem


def get_hamming_dist(tuple1: Tuple[Any], tuple2: Tuple[Any]) -> int:
    """Find the hamming distance between two tuples.
<<<<<<< HEAD
=======

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
    The hamming distance is the number of respective elements in the tuples that are not equal.
    E.g. tuple1 = (1, 23, 'a'), tuple2 = (1, 1, 'a')
    Hamming distance for the above example is 1,
    since the 2nd element of the two tuples don't match.
<<<<<<< HEAD
    Args:
        tuple1: First tuple
        tuple2: Second tuple
=======

    Args:
        tuple1: First tuple
        tuple2: Second tuple

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
    Returns:
        Number of respective elements that don't match
    """
    dist = 0
    assert len(tuple1) == len(tuple2), ('Tuples must have the same length')

    for val1, val2 in zip(tuple1, tuple2):
        if val1 != val2:
            dist += 1

    return dist


def get_closest_neighbours(samples, target) -> List[Any]:
    """Find elements in a given list that are closest to the given target using hamming distance.
<<<<<<< HEAD
    Arguments:
        samples: The list from which to select neighbours to the given target.
        target: The target element
=======

    Arguments:
        samples: The list from which to select neighbours to the given target.
        target: The target element

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
    Returns:
        Elements in samples that are closest to the target.
    """
    dist = float('inf')
    closest_nei = []
    for sample in samples:
        hamming_dist = get_hamming_dist(target, sample)
        if hamming_dist < dist:
            dist = hamming_dist
            closest_nei = [sample, ]
        elif hamming_dist == dist:
            closest_nei.append(sample)

    return closest_nei


class GeneralizedCAPScorer(CAPScorer):
    """Calculates the privacy score using CAP (Correct Attribution Probability) model using
    exact match of the original df key in the synthetic table
<<<<<<< HEAD
=======

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
    Fits the model by mapping the predictors keys to the sensitive values using the synthetic data.
    Then, evaluates the privacy of the original dataset by trying to find the sensitive attribute of the
    original data in the dictionary corresponding to the original df predictor key.
    """

    def __init__(self) -> None:
        super().__init__()

    def score(self, orig_predictors: pd.DataFrame, orig_sensitive: pd.Series) -> float:
        """Gives the privacy score based on evaluation of the original data using synthetic data
<<<<<<< HEAD
=======

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
        For each row of predictors_data, calculate as follows:
        find the list of sensitive values corresponding to predictor key in the fitted
        predictors_to_sensitive_val_map dictionary, and then find the frequency of true sensitive
        value in this list. Add this frequency to the score.
<<<<<<< HEAD
        Average out the score to get the final privacy score.
        Args:
            orig_predictors: Values of predictor columns in original df
            orig_sensitive: Values of the sensitive column in original df
=======

        Average out the score to get the final privacy score.

        Args:
            orig_predictors: Values of predictor columns in original df
            orig_sensitive: Values of the sensitive column in original df

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
        Returns:
            Privacy score between 0 and 1, 0 means no privacy and 1 means absolute privacy
        """

        score = 0.0
        count = 0

        for (pval, sval) in zip(orig_predictors.values, orig_sensitive.values):
            key = tuple(pval)
            if key in self.predictors_to_sensitive_val_map:
                score += get_frequency(self.predictors_to_sensitive_val_map[key], sval)
                count += 1

        if count == 0:
            return 0

        return 1.0 - score / count


class DistanceCAPScorer(CAPScorer):
    """Calculates the privacy score using CAP (Correct Attribution Probability) distance model
    which uses neighbouring keys of the original df key to evaluate privacy
<<<<<<< HEAD
=======

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
    Fits the model by mapping the predictors keys to the sensitive values using the synthetic data.
    Then, evaluates the privacy of the original dataset by trying to find the sensitive attribute of the
    original data in the dictionary corresponding to the neighbouring keys of the original df predictor key.
    """
    def __init__(self) -> None:
        super().__init__()

    def score(self, orig_predictors: pd.DataFrame, orig_sensitive: pd.Series) -> float:
        """Gives the privacy score based on evaluation of the original data using synthetic data
<<<<<<< HEAD
=======

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
        For each row of predictors_data, calculate as follows:
        find the closest neighbouring keys of original df predictor key from predictors_to_sensitive_val_map,
        get the list of sensitive values corresponding to these neighbouring keys from
        predictors_to_sensitive_val_map dictionary, and then find the frequency of true sensitive
        value in this list. Add this frequency to the score.
<<<<<<< HEAD
        Average it out to get the final privacy score.
        Args:
            orig_predictors: Values of predictor columns in original df
            orig_sensitive: Values of the sensitive column in original df
=======

        Average it out to get the final privacy score.

        Args:
            orig_predictors: Values of predictor columns in original df
            orig_sensitive: Values of the sensitive column in original df

>>>>>>> e8862ccd9b4c98ab34b851ed0e67a790608f71b7
        Returns:
            Privacy score between 0 and 1, 0 means no privacy and 1 means absolute privacy
        """
        score = 0.0
        count = 0

        for (pval, sval) in zip(orig_predictors.values, orig_sensitive.values):
            key = tuple(pval)

            closest_neighbours = get_closest_neighbours(self.predictors_to_sensitive_val_map.keys(), key)
            sensitive_vals = []
            for nei in closest_neighbours:
                sensitive_vals.extend(self.predictors_to_sensitive_val_map[nei])

            if sval in sensitive_vals:
                score += get_frequency(sensitive_vals, sval)
                count += 1

        if count == 0:
            return 0

        return 1.0 - score / count
