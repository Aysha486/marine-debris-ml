"""
Simple Marine Debris Risk Index (MDRI) utilities.

compute_mdri(df, weights=None)
- Normalizes selected indicators and computes a weighted sum.
- Returns a copy of df with an additional `mdri` column in range [0,1].

Adjust `DEFAULT_WEIGHTS` to match your project design and domain knowledge.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Default features and weights — modify after reviewing available columns
DEFAULT_FEATURES = ['Total Items Collected', 'items_per_person', 'PEOPLE']
DEFAULT_WEIGHTS = {'Total Items Collected': 0.6, 'items_per_person': 0.3, 'PEOPLE': 0.1}


def compute_mdri(df, features_weights=None):
    """
    Compute a normalized Marine Debris Risk Index (MDRI).

    Parameters
    - df: pandas.DataFrame with relevant indicators
    - features_weights: dict mapping column -> weight. If None, DEFAULT_WEIGHTS is used.

    Returns: copy of df with `mdri` column (0..1)
    """
    if features_weights is None:
        features_weights = DEFAULT_WEIGHTS

    # pick only features that exist and are numeric
    available = [f for f in features_weights.keys() if f in df.columns]
    if not available:
        raise ValueError('No MDRI features available in DataFrame')

    df_out = df.copy()
    numeric_df = df_out[available].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # normalize each feature 0..1
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns, index=numeric_df.index)

    # weighted sum
    total_weight = sum(features_weights.get(f, 0) for f in available)
    if total_weight <= 0:
        raise ValueError('Sum of weights must be > 0')

    mdri_vals = np.zeros(len(df_out))
    for f in available:
        w = features_weights.get(f, 0) / total_weight
        mdri_vals += scaled[f].values * w

    # store in df_out
    df_out['mdri'] = mdri_vals
    return df_out


if __name__ == '__main__':
    print('mdri.py module loaded. Use compute_mdri(df, features_weights)')
