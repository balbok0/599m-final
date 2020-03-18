from imblearn.over_sampling import \
    ADASYN, BorderlineSMOTE, KMeansSMOTE, \
    RandomOverSampler, SMOTE, SMOTENC, SVMSMOTE

import numpy as np
from matplotlib import pyplot as plt

from load_data import load_data

from tqdm import tqdm


def main():
    dataset, hidden_dataset = load_data()

    dataset = np.array(dataset.to_numpy(), dtype=float)

    dists_men = []
    dists_women = []
    for idx, row in enumerate(tqdm(dataset)):
        dist = np.sqrt(np.sum(np.square(dataset - row), axis=1))
        dist = np.delete(dist, idx)
        if hidden_dataset['sex'][idx].lower() == 'male':
            dists_men.extend(dist)
        else:
            dists_women.extend(dist)

    plt.hist(dists_men)
    plt.hist(dists_women)
    plt.show()

    print(f'\n\n\n\n\n{dataset}')

if __name__ == "__main__":
    main()
