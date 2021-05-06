from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd

from .exceptions import AssociatedRuleOverlap
from ...metadata import DataFrameMeta, Nominal


class Association:
    """ Define a relationship between two categorical variables that the synthesizer must obey, the index to value
    mapping is determined from the order of the categories in the appropriate meta class

    Attributes:
        associations: list of column names to associate with
        nan_associations: list of nan indicators to associate with
        binding_mask: masking tensor to prevent certain outputs of the associated columns
    """

    def __init__(self, binding_mask: np.ndarray,
                 associations: List[str] = None, nan_associations: Optional[List[str]] = None):
        self.associations = associations or []
        self.nan_associations = nan_associations or []

        if set(self.associations).intersection(self.nan_associations):
            raise AssociatedRuleOverlap("Cannot associated a column with its nan counterpart, "
                                        "we currently don't support the functionality where we associate "
                                        "a column A with B such that A_nan and B are also associated")

        self.binding_mask = binding_mask

    def __repr__(self):
        return f"Association(associations={self.associations}, nan_associations={self.nan_associations})"

    @classmethod
    def detect_association(cls, df: pd.DataFrame, df_meta: DataFrameMeta,
                           associations: List[str] = None, nan_associations: List[str] = None) -> 'Association':
        """
        Constructor that automatically generates a binding mask for the association based on a dataframe,
        masks all combinations of inputs that don't appear in the dataframe.

        Arguments:
            df: input DataFrame.
            df_meta: extracted DataFrameMeta
            associations: list of regular columns to check for associations
            nan_associations: list of columns to check nan value associations

        Returns:
            association: new Association object with automatically generated binding mask.

        """
        associations = associations or []
        nan_associations = nan_associations or []

        df_associated = df[associations].copy()
        df_associated[[f"{name}_nan" for name in nan_associations]] = df[nan_associations].isna()

        association_metas: List[Nominal] = list()
        for association in associations:
            meta = df_meta[association]
            assert isinstance(meta, Nominal)
            association_metas.append(meta)

        categories_to_idx = create_categories_to_idx(association_metas)
        for name, idx_mapping in categories_to_idx.items():
            df_associated[name] = df_associated[name].map(idx_mapping)

        counts = np.zeros(shape=[len(association.categories) + 1 for association in association_metas] + [2 for _ in nan_associations])

        for _, row in df_associated.iterrows():
            idx = tuple(int(v) for v in row.values)
            counts[idx] += 1

        binding_mask = (counts > 0).astype(np.int32)
        for axis in range(len(associations)):
            # remove the nan part of the binding mask (dealt with separately)
            binding_mask = np.delete(binding_mask, 0, axis)

        return cls(binding_mask=binding_mask, associations=associations,
                   nan_associations=nan_associations)

    @staticmethod
    def validate_association_rules(association_rules: List['Association']):
        """
        Helper function that checks whether a list of association_rules are valid together.

        Raises:
            AssociatedRuleOverlap if any column or nan column occurs in more than one association
        """
        associated_names = [associated_col for association in association_rules for associated_col in association.associations]
        if len(associated_names) != len(set(associated_names)):
            repeated_elements = [associated_name for associated_name in associated_names if associated_names.count(associated_name) > 1]
            raise AssociatedRuleOverlap(f"Column {repeated_elements[0]} occurs in more than one association which is currently not supported")

        associated_nans = [associated_col for association in association_rules for associated_col in association.nan_associations]
        if len(associated_nans) != len(set(associated_nans)):
            repeated_elements = [associated_nan for associated_nan in associated_nans if associated_nans.count(associated_nan) > 1]
            raise AssociatedRuleOverlap(f"Column {repeated_elements[0]} occurs in more than one association which is currently not supported")


def create_categories_to_idx(association_metas: List[Nominal]):
    categories_to_idx = {}
    for meta in association_metas:
        category_to_idx = defaultdict(int)
        for idx, category in enumerate(meta.categories):
            category_to_idx[category] = idx + 1
        categories_to_idx[meta.name] = category_to_idx

    return categories_to_idx
