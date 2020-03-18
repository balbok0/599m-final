import numpy as np
from matplotlib import pyplot as plt

from load_data import load_data

from keras.models import Sequential
from keras.layers import Dense, ReLU, Softmax
from keras.optimizers import Adam

from tqdm import tqdm

from itertools import product as cartesian_product

import pickle as pkl

def train(x: np.ndarray, y: np.ndarray):
    tmp = np.zeros((len(y), 2), dtype=int)
    tmp[np.arange(len(y), dtype=int), y] = 1
    y = tmp

    model = Sequential()
    model.add(Dense(2, input_dim=98))
    model.add(Softmax())

    opt = Adam(lr=3e-4)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    model.fit(x, y, epochs=5, batch_size=100)

    return model


def evaluate_method(mode: str, bootstrap_size: int = 20):
    fpr_boot = {
        'sex': {
            ' Male': [],
            ' Female': []
        },
        'race': {
            ' Amer-Indian-Eskimo': [],
            ' Asian-Pac-Islander': [],
            ' Black': [],
            ' Other': [],
            ' White': [],
        },
        'combos': {
        }
    }

    for bootstrap_idx in range(bootstrap_size):
        x, y, hidden = load_data(mode, normalize=True)

        model = train(x, y)
        y_pred = model.predict(x)

        fprs = np.logical_and(np.argmax(y_pred, axis=1) == 1, y == 0)

        # Calculate fprs for races
        for race in fpr_boot['race'].keys():
            race_idxs = np.where(hidden[:, 0] == race)[0]
            fpr_boot['race'][race].append(np.sum(fprs[race_idxs]) / len(race_idxs))

        # Calculate fprs for sex
        for sex in fpr_boot['sex'].keys():
            sex_idxs = np.where(hidden[:, 1] == sex)[0]
            fpr_boot['sex'][sex].append(np.sum(fprs[sex_idxs]) / len(sex_idxs))


        # Calculate combinations
        for race, sex in cartesian_product(fpr_boot['race'].keys(), fpr_boot['sex'].keys()):
            key = race + sex
            combo_idxs = np.where(np.logical_and(hidden[:, 0] == race, hidden[:, 1] == sex))[0]
            if not key in fpr_boot['combos']:
                fpr_boot['combos'][key] = []
            fpr_boot['combos'][key].append(np.sum(fprs[combo_idxs]) / len(combo_idxs))

    return fpr_boot


def main():
    for mode in [
        'vanilla',
        'smote',
        'adasyn',
        'bordersmote',
        'randomover',
        'randomunder',
        'tomek',
        'knn'
    ]:
        result = evaluate_method(mode, bootstrap_size=100)
        with open(f'fprs_{mode}.pkl', mode='wb') as f:
            pkl.dump(result, f)


def analyze_results():
    for mode in [
        'vanilla',
        'smote',
        'adasyn',
        'bordersmote',
        'randomover',
        'randomunder',
        'tomek',
        # 'knn',
    ]:
        with open(f'fprs_{mode}.pkl', mode='rb') as f:
            d = pkl.load(f)


            print(f'Mode: {mode}')
            for key, key_dict in d.items():
                for subkey, val in key_dict.items():
                    print(f'\t{subkey}')
                    print(f'\t\tMean: {np.mean(val)}')
                    print(f'\t\tStd Dev: {np.std(val)}')


if __name__ == "__main__":
    # main()

    analyze_results()