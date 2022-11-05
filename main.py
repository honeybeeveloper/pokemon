from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from common import convert_to_int


class PokemonPrediction(object):
    def __init__(self):
        self.totals = None
        # shuffle
        combats = pd.read_csv('./data/combats.csv', encoding='utf-8')
        self.combats = combats.sample(frac=1)
        self.pokemon = pd.read_csv('./data/pokemon.csv', encoding='utf-8')

    def start(self):
        self.preprocessing()
        self.first_player()
        self.second_player()
        x_train, x_test, y_train, y_test = self.handle_totals()
        self.run_decision_tree(x_train, x_test, y_train, y_test)

    def preprocessing(self):
        self.pokemon.pop('Name')
        self.pokemon['Legendary'] = np.where(self.pokemon['Legendary'] == False, 0, 1)
        # self.pokemon['Legendary'].value_counts()

        type_1_dit, type_2_dit = self.get_type_1_2_sort()
        self.convert_type_to_int(type_1_dit, type_2_dit)

    def get_type_1_2_sort(self):
        type_1_dit = defaultdict(int)
        type_2_dit = defaultdict(int)
        for idx, key in enumerate(np.unique(self.pokemon['Type 1'].values.astype(str))):
            type_1_dit[key] = idx
        for idx, key in enumerate(np.unique(self.pokemon['Type 2'].values.astype(str))):
            type_2_dit[key] = idx
        return type_1_dit, type_2_dit

    def convert_type_to_int(self, type_1_dit, type_2_dit):
        self.pokemon['Type 1'] = self.pokemon['Type 1'].apply(convert_to_int, args=(type_1_dit,))
        self.pokemon['Type 2'] = self.pokemon['Type 2'].apply(convert_to_int, args=(type_2_dit,))

    def first_player(self):
        # first-player
        first_combats = self.combats.rename(columns={'First_pokemon': '#'}, inplace=False)
        self.totals = first_combats.merge(self.pokemon, on='#', how='left')
        first_col_name = {'#': 'First pokemon', 'Type 1': 'First Type 1', 'Type 2': 'First Type 2',
                          'HP': 'First HP', 'Attack': 'First Attack', 'Defense': 'First Defense',
                          'Sp. Atk': 'First Sp. Atk', 'Sp. Def': 'First Sp. Def',
                          'Speed': 'First Speed', 'Generation': 'First Generation', 'Legendary': 'First Legendary'}
        self.totals.rename(columns=first_col_name, inplace=True)
        print(f'first_player - total columns length :{len(self.totals.columns)}')

    def second_player(self):
        # second-player
        second_combats = self.totals.rename(columns={'Second_pokemon': '#'}, inplace=False)
        self.totals = second_combats.merge(self.pokemon, on='#', how='left')
        second_col_name = {'#': 'Second pokemon', 'Type 1': 'Second Type 1', 'Type 2': 'Second Type 2',
                           'HP': 'Second HP', 'Attack': 'Second Attack', 'Defense': 'Second Defense',
                           'Sp. Atk': 'Second Sp. Atk', 'Sp. Def': 'First Sp. Def',
                           'Speed': 'Second Speed', 'Generation': 'Second Generation', 'Legendary': 'Second Legendary'}
        self.totals.rename(columns=second_col_name, inplace=True)
        print(f'second_player - total columns length :{len(self.totals.columns)}')

    def handle_totals(self):
        label = self.totals['Winner']
        self.totals = self.totals.drop(['First pokemon', 'Second pokemon', 'Winner'], axis=1)

        # convert to ndarray
        x_train = self.totals.to_numpy()[:35000]
        x_test = self.totals.to_numpy()[35000:]
        y_train = label.to_numpy()[:35000]
        y_test = label.to_numpy()[35000:]
        return x_train, x_test, y_train, y_test

    def run_decision_tree(self, x_train, x_test, y_train, y_test):
        tree = DecisionTreeClassifier(random_state=0)
        tree = tree.fit(x_train, y_train)

        accuracy_train = tree.score(x_train, y_train)
        accuracy_test = tree.score(x_test, y_test)
        print(f'훈련 세트 정확도: {accuracy_train:.3f}')
        print(f'테스트 세트 정확도: {accuracy_test:.3f}')


if __name__ == '__main__':
    pokemon = PokemonPrediction()
    pokemon.start()