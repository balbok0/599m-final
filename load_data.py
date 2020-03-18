import pandas as pd
import numpy as np

def load_data(normalize: bool = True):
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
    print(df.dtypes)
    initial_columns = df.columns
    for column in initial_columns:
        if column == 'earnings':
            df[column] = df[column] == '>=50k'
        elif df[column].dtype == 'object':
            for unique_value in np.unique(df[column]):
                df[unique_value] = 1.0 * (df[column] == unique_value)
                if normalize:
                    df[unique_value] -= np.mean(df[unique_value])
                    df[unique_value] /= np.std(df[unique_value])
            del df[column]
        else:
            df[column] *= 1.0
            if normalize:
                df[column] -= np.mean(df[column])
                df[column] /= np.std(df[column])

    return df, hidden_df
