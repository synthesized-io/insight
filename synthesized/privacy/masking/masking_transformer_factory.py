from typing import Dict, List

from synthesized.metadata import DataFrameMeta
from synthesized.transformer.base import Transformer
from synthesized.transformer.data_frame import DataFrameTransformer
import os
print(os.getcwd())
from synthesized.privacy.masking import (NullTransformer, PartialTransformer, RandomTransformer, RoundingTransformer, SwappingTransformer)


class MaskingTransformerFactory:
    def __init__(self):
        # Init is empty but we still create the factory as a class to imitate the TransformerFactory class.
        pass

    def _create_random_transformer(self, masked_column, arg_val) -> RandomTransformer:
        str_length = 10
        if arg_val:
            str_length = int(arg_val)
        return RandomTransformer(name=masked_column, str_length=str_length)

    def _create_partial_transformer(self, masked_column, arg_val) -> PartialTransformer:
        masking_proportion = 0.75
        if arg_val:
            masking_proportion = float(arg_val)
        return PartialTransformer(name=masked_column, masking_proportion=masking_proportion)

    def _create_rounding_transformer(self, masked_column, arg_val) -> RoundingTransformer:
        n_bins = 20
        if arg_val:
            n_bins = int(arg_val)
        return RoundingTransformer(name=masked_column, n_bins=n_bins)

    def create_transformers(self, config: Dict[str, str]) -> DataFrameTransformer:
        """
        Creates masking transformers and returns them wrapped up in a DataFrameTransformer

        Args:
            config: dictionary mapping columns to the respective masking techniques

        Returns:
            A DataFrameTransformer instance.
        """
        transformers: List[Transformer] = []
        for masked_column, masking_technique in config.items():
            split = masking_technique.split('|', 1)
            arg_val = None
            if len(split) > 1:
                arg_val = split[1]

            if masking_technique.startswith('random'):
                transformers.append(self._create_random_transformer(masked_column, arg_val))
            elif masking_technique.startswith('partial_masking'):
                transformers.append(self._create_partial_transformer(masked_column, arg_val))
            elif masking_technique.startswith('rounding'):
                transformers.append(self._create_rounding_transformer(masked_column, arg_val))
            elif masking_technique.startswith('swapping'):
                transformers.append(SwappingTransformer(name=masked_column))
            elif masking_technique.startswith('null'):
                transformers.append(NullTransformer(name=masked_column))
            else:
                raise ValueError(f"Given masking technique '{masking_technique}' for "
                                 f"column '{masked_column}' not supported")

        dataframe_meta = DataFrameMeta(name='masked_columns_meta', columns=list(config.keys()))
        return DataFrameTransformer(meta=dataframe_meta, transformers=transformers)
