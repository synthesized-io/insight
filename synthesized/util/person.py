import re
from typing import Dict, List, Optional, Sequence

import numpy as np


def collections_from_mapping(
        seq: Sequence[str], mapping: Dict[str, str], ambiguous_key: Optional[str] = None
) -> Dict[str, List[str]]:
    unique = np.unique(seq)
    discrete_mapping = {
        key: [val for val in unique if re.match(pattern, val, flags=re.I) is not None]
        for key, pattern in mapping.items()
    }
    ambiguous = discrete_mapping.get(ambiguous_key, []) if ambiguous_key is not None else []

    for val in unique:
        if len([key for key, values in discrete_mapping.items() if val in values]) > 1 and val not in ambiguous:
            ambiguous.append(val)

    if len(ambiguous) > 0:
        if ambiguous_key is None:
            raise ValueError("Found ambiguous regex collections using map: f{mapping} but no ambiguous_key specified.")

        for key, collection in discrete_mapping.items():
            discrete_mapping[key] = [v for v in collection if v not in ambiguous]

        discrete_mapping[ambiguous_key] = ambiguous

    return discrete_mapping
