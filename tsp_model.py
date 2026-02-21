import numpy as np
import json
import os


class TSPInstance:
    """Classe pour gérer les instances du TSP"""

    def __init__(self, n_cities, seed=None):
        """
        Initialise une instance TSP

        Args:
            n_cities: nombre de villes
            seed: graine aléatoire pour reproductibilité
        """
        self.n_cities = n_cities
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Générer les coordonnées aléatoires
        self.coords = np.random.uniform(0, 100, (n_cities, 2))

        # Calculer la matrice de distances
        self.distances = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        """Calcule la matrice de distances euclidiennes"""
        n = self.n_cities
        dist = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((self.coords[i] - self.coords[j]) ** 2))
                dist[i, j] = dist[j, i] = round(d)

        return dist

    def tour_length(self, tour):
        """
        Calcule la longueur totale d'un tour

        Args:
            tour: liste représentant l'ordre des villes

        Returns:
            longueur totale du tour
        """
        length = 0
        n = len(tour)
        for i in range(n):
            length += self.distances[tour[i], tour[(i + 1) % n]]
        return length

    def save(self, filename):
        """Sauvegarde l'instance dans un fichier JSON"""
        data = {
            'n_cities': self.n_cities,
            'seed': self.seed,
            'coords': self.coords.tolist(),
            'distances': self.distances.tolist()
        }
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filename):
        """Charge une instance depuis un fichier JSON"""
        with open(filename, 'r') as f:
            data = json.load(f)

        instance = cls.__new__(cls)
        instance.n_cities = data['n_cities']
        instance.seed = data['seed']
        instance.coords = np.array(data['coords'])
        instance.distances = np.array(data['distances'])
        return instance


def generate_instances():
    """Génère les 3 instances requises pour le projet"""
    instances = [
        ('data/instance_A_20.json', 20, 42),
        ('data/instance_B_50.json', 50, 123),
        ('data/instance_C_80.json', 80, 456)
    ]

    for filename, n_cities, seed in instances:
        instance = TSPInstance(n_cities, seed)
        instance.save(filename)
        print(f"Instance {filename} créée: {n_cities} villes")


if __name__ == "__main__":
    generate_instances()