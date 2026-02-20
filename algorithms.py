import numpy as np
import random
from copy import deepcopy
import time


class TSPSolver:
    """Classe de base pour tous les solveurs TSP"""

    def __init__(self, instance):
        self.instance = instance
        self.n_cities = instance.n_cities
        self.eval_count = 0
        self.max_evals = None

    def evaluate(self, tour):
        """Évalue un tour et incrémente le compteur"""
        self.eval_count += 1
        return self.instance.tour_length(tour)

    def random_solution(self):
        """Génère une solution aléatoire"""
        return list(np.random.permutation(self.n_cities))

    def swap_neighbors(self, tour):
        """Génère tous les voisins par swap de 2 villes"""
        neighbors = []
        for i in range(len(tour)):
            for j in range(i + 1, len(tour)):
                neighbor = tour.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors

    def two_opt_neighbors(self, tour):
        """Génère tous les voisins par 2-opt"""
        neighbors = []
        n = len(tour)
        for i in range(n - 1):
            for j in range(i + 2, n):
                neighbor = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
                neighbors.append(neighbor)
        return neighbors


class HillClimbing(TSPSolver):
    """Hill Climbing avec first et best improvement"""

    def solve(self, initial_solution=None, mode='best', max_evals=10000, neighborhood='swap'):
        """
        Résout le TSP avec Hill Climbing

        Args:
            initial_solution: solution initiale (None = aléatoire)
            mode: 'best' ou 'first' improvement
            max_evals: budget maximum d'évaluations
            neighborhood: 'swap' ou '2opt'

        Returns:
            meilleure solution, son coût, temps
        """
        start_time = time.time()
        self.eval_count = 0
        self.max_evals = max_evals

        # Solution initiale
        current = initial_solution if initial_solution else self.random_solution()
        current_cost = self.evaluate(current)

        improved = True
        while improved and self.eval_count < max_evals:
            improved = False

            # Générer les voisins
            if neighborhood == '2opt':
                neighbors = self.two_opt_neighbors(current)
            else:
                neighbors = self.swap_neighbors(current)

            if mode == 'best':
                # Best improvement: évaluer tous les voisins
                best_neighbor = None
                best_cost = current_cost

                for neighbor in neighbors:
                    if self.eval_count >= max_evals:
                        break
                    cost = self.evaluate(neighbor)
                    if cost < best_cost:
                        best_cost = cost
                        best_neighbor = neighbor

                if best_neighbor is not None:
                    current = best_neighbor
                    current_cost = best_cost
                    improved = True

            else:  # first improvement
                for neighbor in neighbors:
                    if self.eval_count >= max_evals:
                        break
                    cost = self.evaluate(neighbor)
                    if cost < current_cost:
                        current = neighbor
                        current_cost = cost
                        improved = True
                        break

        elapsed_time = time.time() - start_time
        return current, current_cost, elapsed_time


class MultiStartHC(TSPSolver):
    """Multi-start Hill Climbing"""

    def solve(self, n_starts=30, hc_mode='best', max_evals=10000, neighborhood='swap'):
        """
        Résout le TSP avec Multi-start HC

        Args:
            n_starts: nombre de démarrages
            hc_mode: mode de HC ('best' ou 'first')
            max_evals: budget total d'évaluations
            neighborhood: 'swap' ou '2opt'

        Returns:
            meilleure solution, son coût, temps
        """
        start_time = time.time()
        self.eval_count = 0
        self.max_evals = max_evals

        best_solution = None
        best_cost = float('inf')
        evals_per_start = max_evals // n_starts

        hc = HillClimbing(self.instance)

        for _ in range(n_starts):
            if self.eval_count >= max_evals:
                break

            # Nouvelle solution initiale
            initial = self.random_solution()

            # Lancer HC
            solution, cost, _ = hc.solve(
                initial_solution=initial,
                mode=hc_mode,
                max_evals=evals_per_start,
                neighborhood=neighborhood
            )

            self.eval_count += hc.eval_count

            if cost < best_cost:
                best_cost = cost
                best_solution = solution

        elapsed_time = time.time() - start_time
        return best_solution, best_cost, elapsed_time


class SimulatedAnnealing(TSPSolver):
    """Recuit Simulé"""

    def solve(self, initial_solution=None, T0=100, alpha=0.95, Tmin=0.01,
              max_evals=10000, neighborhood='swap'):
        """
        Résout le TSP avec Recuit Simulé

        Args:
            initial_solution: solution initiale (None = aléatoire)
            T0: température initiale
            alpha: facteur de refroidissement
            Tmin: température minimale
            max_evals: budget maximum d'évaluations
            neighborhood: 'swap' ou '2opt'

        Returns:
            meilleure solution, son coût, temps
        """
        start_time = time.time()
        self.eval_count = 0
        self.max_evals = max_evals

        # Solution initiale
        current = initial_solution if initial_solution else self.random_solution()
        current_cost = self.evaluate(current)

        best_solution = current.copy()
        best_cost = current_cost

        T = T0

        while T > Tmin and self.eval_count < max_evals:
            # Générer un voisin aléatoire
            if neighborhood == '2opt':
                i, j = sorted(random.sample(range(self.n_cities), 2))
                if j - i < 2:
                    continue
                neighbor = current[:i + 1] + current[i + 1:j + 1][::-1] + current[j + 1:]
            else:  # swap
                i, j = random.sample(range(self.n_cities), 2)
                neighbor = current.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            neighbor_cost = self.evaluate(neighbor)
            delta = neighbor_cost - current_cost

            # Acceptation
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current = neighbor
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = current.copy()
                    best_cost = current_cost

            # Refroidissement
            T *= alpha

        elapsed_time = time.time() - start_time
        return best_solution, best_cost, elapsed_time


class TabuSearch(TSPSolver):
    """Recherche Tabou (BONUS)"""

    def solve(self, initial_solution=None, tabu_tenure=10, max_evals=10000,
              neighborhood='swap'):
        """
        Résout le TSP avec Recherche Tabou

        Args:
            initial_solution: solution initiale
            tabu_tenure: durée tabou
            max_evals: budget maximum d'évaluations
            neighborhood: 'swap' ou '2opt'

        Returns:
            meilleure solution, son coût, temps
        """
        start_time = time.time()
        self.eval_count = 0
        self.max_evals = max_evals

        # Solution initiale
        current = initial_solution if initial_solution else self.random_solution()
        current_cost = self.evaluate(current)

        best_solution = current.copy()
        best_cost = current_cost

        tabu_list = []

        while self.eval_count < max_evals:
            # Générer voisins
            if neighborhood == '2opt':
                neighbors_moves = [(i, j) for i in range(self.n_cities - 1)
                                   for j in range(i + 2, self.n_cities)]
            else:
                neighbors_moves = [(i, j) for i in range(self.n_cities)
                                   for j in range(i + 1, self.n_cities)]

            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None

            # Trouver le meilleur voisin non-tabou
            for move in neighbors_moves:
                if self.eval_count >= max_evals:
                    break

                i, j = move
                if neighborhood == '2opt':
                    neighbor = current[:i + 1] + current[i + 1:j + 1][::-1] + current[j + 1:]
                else:
                    neighbor = current.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                cost = self.evaluate(neighbor)

                # Accepter si non-tabou ou critère d'aspiration
                if (move not in tabu_list or cost < best_cost) and cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost
                    best_move = move

            if best_neighbor is None:
                break

            current = best_neighbor
            current_cost = best_neighbor_cost

            # Mettre à jour la liste tabou
            tabu_list.append(best_move)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

            # Mettre à jour la meilleure solution
            if current_cost < best_cost:
                best_solution = current.copy()
                best_cost = current_cost

        elapsed_time = time.time() - start_time
        return best_solution, best_cost, elapsed_time