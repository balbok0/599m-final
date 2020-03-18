import pandas as pd
import numpy as np
from imblearn.over_sampling import \
    ADASYN, BorderlineSMOTE, KMeansSMOTE, \
    RandomOverSampler, SMOTE, SMOTENC, SVMSMOTE

from imblearn.under_sampling import \
    TomekLinks, RandomUnderSampler, CondensedNearestNeighbour

import os


def __load_data_first_time():
    """Actually read the data, and return it
    """
    df: pd.DataFrame = pd.read_csv(
        'adult.data',
        names=[
            'age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'martial-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country',
            'earnings'
        ]
    )
    # Remove unnecessary columns
    del df['fnlwgt']

    # Split to hidden and visible data
    hidden_columns = ['race', 'sex']
    hidden_df = df[hidden_columns]

    for x in hidden_columns:
        del df[x]

    # Turning categorical data to numerical
    initial_columns = df.columns
    for column in initial_columns:
        if column == 'earnings':
            df[column] = 1.0 * (df[column] == ' >50K')
        elif df[column].dtype == 'object':
            for unique_value in np.unique(df[column]):
                df[unique_value] = 1.0 * (df[column] == unique_value)
            del df[column]
        else:
            df[column] *= 1.0

    return df, hidden_df


def load_data(mode: str, normalize: bool = True):
    df, hidden_df = __load_data_first_time()

    # Extract x and y
    y = np.array(df['earnings'].to_numpy(), dtype=int)
    del df['earnings']

    x = np.array(df.to_numpy(), dtype=float)

    # Hidden to numpy
    hidden = hidden_df.to_numpy()

    if mode == 'vanilla':
        pass

    elif mode == 'smote':
        x, y = SMOTE().fit_sample(x, y)

    elif mode == 'adasyn':
        x, y = ADASYN().fit_sample(x, y)

    elif mode == 'bordersmote':
        x, y = BorderlineSMOTE().fit_sample(x, y)

    elif mode == 'randomover':
        x, y, idxs = RandomOverSampler(return_indices=True).fit_sample(x, y)
        hidden = hidden[idxs]

    elif mode == 'randomunder':
        x, y, idxs = RandomUnderSampler(return_indices=True).fit_sample(x, y)
        hidden = hidden[idxs]

    elif mode == 'tomek':
        x, y, idxs = TomekLinks(return_indices=True).fit_sample(x, y)
        hidden = hidden[idxs]

    elif mode == 'knn':
        x, y, idxs = CondensedNearestNeighbour(return_indices=True, n_neighbors=3).fit_sample(x, y)
        hidden = hidden[idxs]

    if normalize:
        x -= np.mean(x, axis=0)
        x /= np.std(x, axis=0)

    return x, y, hidden
