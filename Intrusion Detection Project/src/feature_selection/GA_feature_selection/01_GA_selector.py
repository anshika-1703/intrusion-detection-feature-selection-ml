from abc import ABC, abstractmethod
from enum import Enum
from sklearn.preprocessing import minmax_scale
class ResultType(Enum):
    WEIGHTS = "weights"
    RANK = "rank"
    SUBSET = "subset"


class BaseSelector(ABC):
    def __init__(self, n_features: int):
        self._n_features = n_features

        self._X = None
        self._support_mask = None
        self._selected = None
        self._rank = None
        self._weights = None
        self._fitted = False

    @abstractmethod
    def fit(X, y):
        raise NotImplementedError()

    def _check_fit(self):
        if not self._fitted:
            raise Exception("Model is not fitted!")

    def check_already_fitted(self):
        if self._fitted:
            raise Exception("Model is already fitted!")

    @property
    @classmethod
    @abstractmethod
    def result_type(self):
        raise NotImplementedError()

    def get_features(self, k=None):
        self._check_fit()
        feats = self._X[:, self._support_mask]
        return feats[:, :k] if k and k is not None < self._n_features else feats

    def get_selected(self):
        self._check_fit()
        if self._selected is not None:
            return self._selected
        raise Exception("This selector does not return a subset of features!")

    def get_weights(self):
        self._check_fit()
        if self._weights is not None:
            return minmax_scale(self._weights)
        raise Exception("This selector does not return feature weights!")

    def get_rank(self):
        self._check_fit()
        if self._rank is not None:
            return self._rank
        raise Exception("This selector does not return feature ranks!")

    def get_top_k_rank(self, k):
        self._check_fit()
        if self._rank is not None:
            if self._n_features is None or k < self._n_features:
                return self._rank[:k]
            else:
                raise ValueError("Given `k` should be lower than the number of selected `n_features!")
        raise Exception("This selector does not return feature ranks!")

    def get_mask(self):
        self._check_fit()
        return self._support_mask

    def get_support(self, indices=False):
        self._check_fit()
        return self._selected if indices else self._support_mask

    def transform(self, X):
        return X[:, self._support_mask]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


import math
import time
import shutil
import numpy as np
from numpy.random import random, randint, uniform, permutation
from sklearn.preprocessing import minmax_scale


class GeneticAlgorithmFeatureSelector(BaseSelector):
    result_type = ResultType.SUBSET

    def __init__(
            self,
            n_features=20,
            num_individuals=20,
            max_generations=25,
            num_elite=0.05,
            crossover_rate=0.5,
            mutation_rate=0.001,
            max_fitness=None,
            fitness_function=None,
            verbose=1
    ):
        super().__init__(n_features)
        self._n = n_features
        self._num_individuals = num_individuals
        self._max_generations = max_generations
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._verbose = verbose
        self._max_fitness = max_fitness
        self._fitness_function = fitness_function
        self._fitted = False

        if fitness_function is None:
            raise Exception("fitness_function `f(numpy.array, numpy.array) -> float` must be provided")

        def check_type(val, name, types=[int]):
            if type(val) not in types:
                raise ValueError(f"{name} must be one of {types}")

        check_type(num_elite, 'num_elite', [float, int])
        self._num_elite = num_elite

        check_type(num_individuals, 'num_individuals', [int, str])
        if type(num_individuals) is str and not num_individuals == 'auto':
            raise ValueError("Unknown value for num_individuals, must be `int` or 'auto'")

    def _auto_population(self, n_features):
        feature_pool = permutation(n_features)
        remaining = n_features % self._n
        if remaining > 0:
            padding = np.random.choice(feature_pool[:-remaining], self._n - remaining, replace=False)
            feature_pool = np.concatenate((feature_pool, padding))

        num_individuals = len(feature_pool) / self._n

        return np.split(feature_pool, num_individuals)

    def _initial_population(self, n_features):
        """Generate initial population as diverse as possible, if
        self._num_individuals is set to 'auto', then population
        size is gonna be the minimum needed so that each gene
        happens at least once in first generation"""

        if self._num_individuals == 'auto':
            return self._auto_population(n_features)

        num_ind_to_all = math.ceil(n_features / self._n)

        if self._num_individuals < num_ind_to_all:
            feature_pool = permutation(n_features)
            cut_point = self._num_individuals * self._n
            return np.split(feature_pool[:cut_point], self._num_individuals)

        else:
            needed_pops = math.ceil(self._num_individuals / math.ceil(n_features / self._n))
            population = np.concatenate([self._auto_population(n_features) for x in range(needed_pops)])
            return population[:self._num_individuals]

    def _mutate(self, individual, n_features):
        for i in range(self._n):
            if random() < self._mutation_rate:
                value = randint(0, n_features)
                loop = True
                while (loop):
                    if value not in individual:
                        loop = False
                        individual[i] = value
                    value = randint(0, n_features)
        return individual

    def _crossover(self, parent1, parent2):
        diff_p1 = np.setdiff1d(parent1, parent2)
        num_diff = len(diff_p1)

        if num_diff == 0:
            return parent1, parent2

        diff_p2 = np.setdiff1d(parent2, parent1)

        diff_p1 = permutation(diff_p1)
        diff_p2 = permutation(diff_p2)

        for i in range(num_diff):
            if random() < self._crossover_rate:
                diff_p1[i], diff_p2[i] = diff_p2[i], diff_p1[i]

        if num_diff < self._n:
            common = np.intersect1d(parent1, parent2)
            return np.concatenate((common, diff_p1)), np.concatenate((common, diff_p2))
        else:
            return diff_p1, diff_p2

    def _selection(self, population, fitness):
        scaled_fitness = minmax_scale(fitness)
        fitness_sum = scaled_fitness.sum()
        selected = []
        for _ in range(self._num_to_select):
            current_sum = uniform(0, fitness_sum)
            i = 0
            while (current_sum < fitness_sum):
                current_sum += scaled_fitness[i]
                i += 1
            selected.append(np.copy(population[i - 1]))

        return selected

    def fit(self, X, y):
        import os
        import pickle
        print("starting GA fit....")
        self.check_already_fitted()
        self._X = X
        n_samples, n_features = X.shape
        self._total_features = n_features

        if n_features < self._n_features:
            raise ValueError("Given dataset has less features than the number that should be selected.")

        # Load checkpoint if exists
        checkpoint_path = "/kaggle/working/ga_checkpoint.pkl"  # ✅ permanent directory

        # checkpoint_path = "ga_checkpoint.pkl"
        start_gen = 0
        if os.path.exists(checkpoint_path):
            print("🔄 Resuming from checkpoint...")
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            population = checkpoint["population"]
            fitness = checkpoint["fitness"]
            start_gen = checkpoint["generation"] + 1
        else:
            population = np.array(self._initial_population(n_features))
            print(f"✅ Initial population of {len(population)} individuals created")
            print(f"🧠 Starting initial fitness evaluation...")

            print("🔍 Testing one individual manually...")

            # Take the first individual
            individual = population[0]
            X_individual = X[:, individual]

            # Try SVM fitness on it
            score = self._fitness_function(X_individual, y)
            print("✅ Score for one individual:", score)

            fitness = []
            for idx, x in enumerate(population):
                print(f"🧪 Evaluating individual {idx + 1}/{len(population)}...")
                fit_score = self._fitness_function(X[:, x], y)
                print(f"✅ Fitness for individual {idx + 1}: {fit_score}")
                fitness.append(fit_score)
            fitness = np.array(fitness)

            print(f"✅ Initial fitness evaluation complete")

        num_individuals = len(population)
        self.num_individuals_ = num_individuals

        num_elite = int(self._num_elite * num_individuals) if type(self._num_elite) is float else self._num_elite

        num_to_select = num_individuals - num_elite
        self._num_to_select = num_to_select + (num_to_select % 2)

        for gen in range(self._max_generations):
            start_time = time.time()
            print(f"📈 Starting generation {gen + 1}/{self._max_generations}...")
            if self._verbose > 0:
                print(f"→ Starting generation {gen + 1}")

            fitness_index = np.argsort(fitness)[::-1]
            elite_index = fitness_index[:num_elite]
            elite_fitness = fitness[elite_index]
            elite = population[elite_index]

            best_fitness = fitness[fitness_index[0]]

            if self._verbose > 0:
                print(f"Generation [{gen + 1}/{self._max_generations}]: Best fitness: {fitness[fitness_index[0]]}")

            if self._max_fitness is not None and best_fitness >= self._max_fitness:
                if self._verbose > 0:
                    print("Early stop, found optimal solution!")
                break

            new_population = self._selection(population, fitness)

            offspring_pairs = [
                self._crossover(new_population[i], new_population[i + 1])
                for i in range(0, self._num_to_select, 2)
            ]

            offsprings = [offspring for osp in offspring_pairs for offspring in osp]

            [self._mutate(individual, n_features) for individual in offsprings]

            offsprings_fitness = [self._fitness_function(X[:, x], y) for x in offsprings]

            population = np.concatenate((elite, offsprings))
            fitness = np.concatenate((elite_fitness, offsprings_fitness))
            # ⏱️ Save checkpoint
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "generation": gen,
                    "population": population,
                    "fitness": fitness
                }, f)

            # 📁 Also save a backup version to /kaggle/working for download
            shutil.copy(checkpoint_path, "/kaggle/working/ga_checkpoint_download.pkl")
            print(f"💾 Progress saved after generation {gen + 1} (⏱️ {time.time() - start_time:.2f}s)")

        best_index = np.argsort(fitness)[-1]
        self.last_generation_ = population
        self.last_generation_fitness_ = fitness
        self._selected_fitness = fitness[best_index]
        self._selected = population[best_index]
        self._support_mask = np.zeros(self._total_features, dtype=bool)
        self._support_mask[self._selected] = True
        self._fitted = True
        print("✅ GA completed!")

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        return self
