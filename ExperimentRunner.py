import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tsp_model import TSPInstance
from algorithms import HillClimbing, MultiStartHC, SimulatedAnnealing, TabuSearch
import time
import os


class ExperimentRunner:
    """Classe pour exécuter les expériences et analyser les résultats"""

    def __init__(self, instance):
        self.instance = instance
        self.results = {}

    def run_algorithm(self, algorithm_name, algorithm_class, params, n_runs=30):
        """
        Exécute un algorithme plusieurs fois

        Args:
            algorithm_name: nom de l'algorithme
            algorithm_class: classe de l'algorithme
            params: paramètres de l'algorithme
            n_runs: nombre d'exécutions indépendantes

        Returns:
            dictionnaire avec les statistiques
        """
        print(f"\n{'=' * 60}")
        print(f"Exécution de {algorithm_name}...")
        print(f"{'=' * 60}")

        costs = []
        times = []
        solutions = []

        for run in range(n_runs):
            solver = algorithm_class(self.instance)
            solution, cost, elapsed_time = solver.solve(**params)

            costs.append(cost)
            times.append(elapsed_time)
            solutions.append(solution)

            if (run + 1) % 10 == 0:
                print(f"  Run {run + 1}/{n_runs} terminé - Coût: {cost:.2f}")

        results = {
            'algorithm': algorithm_name,
            'params': params,
            'n_runs': n_runs,
            'costs': costs,
            'times': times,
            'best_cost': min(costs),
            'worst_cost': max(costs),
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'median_cost': np.median(costs),
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'best_solution': solutions[np.argmin(costs)]
        }

        self.results[algorithm_name] = results

        print(f"\n  Résultats:")
        print(f"    Meilleur coût : {results['best_cost']:.2f}")
        print(f"    Coût moyen    : {results['mean_cost']:.2f} ± {results['std_cost']:.2f}")
        print(f"    Temps moyen   : {results['mean_time']:.4f}s")

        return results

    def print_summary_table(self):
        """Affiche un tableau récapitulatif des résultats"""
        print(f"\n{'=' * 100}")
        print(f"TABLEAU RÉCAPITULATIF - Instance de {self.instance.n_cities} villes")
        print(f"{'=' * 100}")
        print(f"{'Algorithme':<30} {'Meilleur':<12} {'Moyen':<15} {'Écart-type':<12} {'Temps (s)':<12}")
        print(f"{'-' * 100}")

        for name, res in self.results.items():
            print(f"{name:<30} {res['best_cost']:<12.2f} {res['mean_cost']:<15.2f} "
                  f"{res['std_cost']:<12.2f} {res['mean_time']:<12.4f}")

        print(f"{'=' * 100}\n")

    def plot_results(self, save_dir='results'):
        """Génère des graphiques de comparaison"""
        os.makedirs(save_dir, exist_ok=True)

        # 1. Boxplot des coûts
        plt.figure(figsize=(12, 6))
        data = [res['costs'] for res in self.results.values()]
        labels = list(self.results.keys())

        plt.boxplot(data, labels=labels)
        plt.ylabel('Coût du tour', fontsize=12)
        plt.title(f'Distribution des coûts - Instance {self.instance.n_cities} villes',
                  fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/boxplot_{self.instance.n_cities}.png', dpi=300)
        plt.close()

        # 2. Barplot comparaison
        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.results))
        width = 0.35

        means = [res['mean_cost'] for res in self.results.values()]
        bests = [res['best_cost'] for res in self.results.values()]

        plt.bar(x - width / 2, means, width, label='Coût moyen', alpha=0.8)
        plt.bar(x + width / 2, bests, width, label='Meilleur coût', alpha=0.8)

        plt.xlabel('Algorithme', fontsize=12)
        plt.ylabel('Coût', fontsize=12)
        plt.title(f'Comparaison des performances - Instance {self.instance.n_cities} villes',
                  fontsize=14, fontweight='bold')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/barplot_{self.instance.n_cities}.png', dpi=300)
        plt.close()

        # 3. Visualisation de la meilleure solution
        best_algo = min(self.results.items(), key=lambda x: x[1]['best_cost'])
        best_solution = best_algo[1]['best_solution']

        plt.figure(figsize=(10, 10))
        coords = self.instance.coords
        tour_coords = coords[best_solution + [best_solution[0]]]

        plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', linewidth=2, alpha=0.6)
        plt.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=5)
        plt.scatter(coords[best_solution[0], 0], coords[best_solution[0], 1],
                    c='green', s=200, marker='s', zorder=6, label='Départ')

        for i, (x, y) in enumerate(coords):
            plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')

        plt.title(f'Meilleur tour trouvé par {best_algo[0]}\nCoût: {best_algo[1]["best_cost"]:.2f}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/best_tour_{self.instance.n_cities}.png', dpi=300)
        plt.close()

        print(f"\nGraphiques sauvegardés dans {save_dir}/")

    def _convert_to_python_type(self, obj):
        """Convertit les types NumPy en types Python natifs"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_python_type(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_to_python_type(v) for k, v in obj.items()}
        else:
            return obj

    def save_results(self, filename):
        """Sauvegarde les résultats en JSON"""
        results_serializable = {}
        for name, res in self.results.items():
            results_serializable[name] = {
                k: self._convert_to_python_type(v)
                for k, v in res.items()
            }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        print(f"Résultats JSON sauvegardés dans {filename}")


def run_complete_experiment():
    """Exécute l'expérimentation complète sur les 3 instances"""

    instances_files = [
        ('data/instance_A_20.json', 20),
        ('data/instance_B_50.json', 50),
        ('data/instance_C_80.json', 80)
    ]

    for instance_file, n_cities in instances_files:
        print(f"\n{'#' * 100}")
        print(f"# INSTANCE: {n_cities} VILLES")
        print(f"{'#' * 100}")

        # Charger l'instance
        instance = TSPInstance.load(instance_file)
        runner = ExperimentRunner(instance)

        # Budget adaptatif selon la taille
        if n_cities == 20:
            max_evals = 5000
            n_runs = 30
        elif n_cities == 50:
            max_evals = 10000
            n_runs = 20
        else:  # 80 villes
            max_evals = 15000
            n_runs = 15

        # 1. Hill Climbing - Best Improvement
        runner.run_algorithm(
            'HC-Best-Swap',
            HillClimbing,
            {'mode': 'best', 'max_evals': max_evals, 'neighborhood': 'swap'},
            n_runs=n_runs
        )

        # 2. Hill Climbing - First Improvement
        runner.run_algorithm(
            'HC-First-Swap',
            HillClimbing,
            {'mode': 'first', 'max_evals': max_evals, 'neighborhood': 'swap'},
            n_runs=n_runs
        )

        # 3. Multi-Start Hill Climbing
        runner.run_algorithm(
            'Multi-Start-HC',
            MultiStartHC,
            {'n_starts': 20, 'hc_mode': 'best', 'max_evals': max_evals, 'neighborhood': 'swap'},
            n_runs=n_runs
        )

        # 4. Recuit Simulé
        runner.run_algorithm(
            'Simulated-Annealing',
            SimulatedAnnealing,
            {'T0': 100, 'alpha': 0.95, 'Tmin': 0.01, 'max_evals': max_evals, 'neighborhood': 'swap', 'palier': 100},
            n_runs=n_runs
        )

        # 5. BONUS: Recherche Tabou
        runner.run_algorithm(
            'Tabu-Search',
            TabuSearch,
            {'tabu_tenure': 10, 'max_evals': max_evals, 'neighborhood': 'swap'},
            n_runs=n_runs
        )

        # Afficher et sauvegarder les résultats
        runner.print_summary_table()
        runner.plot_results(f'results/instance_{n_cities}')
        runner.save_results(f'results/results_{n_cities}.json')


if __name__ == "__main__":
    # Créer le dossier de résultats
    os.makedirs('results', exist_ok=True)

    # Lancer l'expérimentation complète
    run_complete_experiment()

    print("\n" + "=" * 100)
    print("EXPÉRIMENTATION TERMINÉE!")
    print("=" * 100)