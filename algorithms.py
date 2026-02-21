import numpy as np
import random
from copy import deepcopy
import time
from collections import deque

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

    def solve(self, n_starts=None, hc_mode='best', max_evals=10000, neighborhood='swap'):
        # On peut ignorer n_starts si on veut consommer TOUT le budget max_evals

        start_time = time.time()
        self.eval_count = 0
        self.max_evals = max_evals

        best_solution = None
        best_cost = float('inf')

        hc = HillClimbing(self.instance)

        # On boucle TANT QU'IL NOUS RESTE DU BUDGET
        while self.eval_count < max_evals:
            initial = self.random_solution()

            # Le budget alloué à ce lancement est ce qu'il reste du budget total
            remaining_evals = max_evals - self.eval_count

            solution, cost, _ = hc.solve(
                initial_solution=initial,
                mode=hc_mode,
                max_evals=remaining_evals,
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
              max_evals=10000, neighborhood='swap', palier=100):

        start_time = time.time()
        self.eval_count = 0  # Compteur global d'évaluations
        self.max_evals = max_evals

        # Initialisation
        current = initial_solution if initial_solution else self.random_solution()
        current_cost = self.evaluate(current)
        self.eval_count += 1  # On compte la première évaluation

        best_solution = current.copy()
        best_cost = current_cost

        T = T0
        iter_count = 0  # Compteur d'itérations au palier actuel

        import math  # Utilisé pour math.exp, plus rapide que np.exp sur des scalaires

        while T > Tmin and self.eval_count < max_evals:
            # 1. Générer un voisin aléatoire
            if neighborhood == '2opt':
                i, j = sorted(random.sample(range(self.n_cities), 2))
                if j - i < 2:
                    continue  # Ignore les mouvements invalides, on recommence la boucle
                neighbor = current[:i + 1] + current[i + 1:j + 1][::-1] + current[j + 1:]
            else:
                i, j = random.sample(range(self.n_cities), 2)
                neighbor = current.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            # 2. Évaluer le voisin
            neighbor_cost = self.evaluate(neighbor)
            self.eval_count += 1  # CORRECTION : On incrémente le budget total à chaque évaluation

            delta = neighbor_cost - current_cost

            # 3. Critère d'acceptation (Metropolis)
            # Si c'est meilleur (delta <= 0) OU accepté par probabilité
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current = neighbor
                current_cost = neighbor_cost

                # Mise à jour du meilleur global
                if current_cost < best_cost:
                    best_solution = current.copy()
                    best_cost = current_cost

            # 4. Gestion du palier et refroidissement
            iter_count += 1

            if iter_count >= palier:
                T *= alpha
                iter_count = 0  # reset du compteur de palier

        elapsed_time = time.time() - start_time
        return best_solution, best_cost, elapsed_time


class TabuSearch(TSPSolver):
    """Recherche Tabou"""

    def solve(self, initial_solution=None, tabu_tenure=10, max_evals=10000,
              neighborhood='swap'):

        start_time = time.time()
        self.eval_count = 0
        self.max_evals = max_evals

        # ── Solution initiale ──────────────────────────────────────────
        current = initial_solution if initial_solution else self.random_solution()
        current_cost = self.evaluate(current)   # evaluate() gère eval_count

        best_solution = current.copy()
        best_cost = current_cost

        # ── Liste tabou (deque à taille fixe) ─────────────────────────
        # maxlen gère automatiquement la suppression des anciens mouvements
        tabu_list = deque(maxlen=tabu_tenure)

        # ── Boucle principale ──────────────────────────────────────────
        while self.eval_count < max_evals:

            # Générer tous les mouvements possibles
            if neighborhood == '2opt':
                moves = [(i, j) for i in range(self.n_cities - 1)
                         for j in range(i + 2, self.n_cities)]
            else:
                moves = [(i, j) for i in range(self.n_cities)
                         for j in range(i + 1, self.n_cities)]

            # Variables pour le meilleur voisin non-tabou
            best_neighbor      = None
            best_neighbor_cost = float('inf')
            best_move          = None

            # Variables de secours : le moins pire des voisins tabous
            # (utilisé seulement si TOUS les mouvements sont tabous)
            fallback_neighbor      = None
            fallback_cost          = float('inf')
            fallback_move          = None

            # ── Évaluation des voisins ─────────────────────────────────
            for move in moves:
                if self.eval_count >= max_evals:
                    break

                i, j = move

                # Construire le voisin selon le type de voisinage
                if neighborhood == '2opt':
                    neighbor = (current[:i + 1] +
                                current[i + 1:j + 1][::-1] +
                                current[j + 1:])
                else:
                    neighbor = current.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                # evaluate() incrémente eval_count automatiquement
                cost = self.evaluate(neighbor)

                # ── Critère d'acceptation ──────────────────────────────
                # Un mouvement est accepté si :
                #   1) il n'est pas tabou  →  candidat normal
                #   2) il est tabou MAIS bat le record global  →  aspiration
                if move not in tabu_list or cost < best_cost:
                    if cost < best_neighbor_cost:
                        best_neighbor      = neighbor
                        best_neighbor_cost = cost
                        best_move          = move

                # Sinon on garde quand même le moins pire (filet de sécurité)
                elif cost < fallback_cost:
                    fallback_neighbor = neighbor
                    fallback_cost     = cost
                    fallback_move     = move

            # ── Choisir le mouvement à effectuer ──────────────────────
            if best_neighbor is not None:
                # Cas normal : on a au moins un voisin non-tabou
                current      = best_neighbor
                current_cost = best_neighbor_cost
                chosen_move  = best_move
            elif fallback_neighbor is not None:
                # Cas rare : tout est tabou → on prend le moins pire
                current      = fallback_neighbor
                current_cost = fallback_cost
                chosen_move  = fallback_move
            else:
                # Cas impossible en pratique, mais sécurité totale
                break

            # ── Mise à jour de la liste tabou ──────────────────────────
            tabu_list.append(chosen_move)

            # ── Mise à jour de la meilleure solution globale ───────────
            if current_cost < best_cost:
                best_solution = current.copy()
                best_cost     = current_cost

        elapsed_time = time.time() - start_time
        return best_solution, best_cost, elapsed_time