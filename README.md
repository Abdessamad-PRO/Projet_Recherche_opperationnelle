# Projet TSP — Comparaison de Métaheuristiques

**Université Hassan II de Casablanca — ENSET Mohammedia**  
**Masters : SDIA | UE : Optimisation / Métaheuristiques**  
**Enseignant : Prof. MESTARI | Année universitaire 2025-2026**

---

## Description

Ce projet compare plusieurs métaheuristiques appliquées au **Problème du Voyageur de Commerce (TSP)** sur trois instances de tailles croissantes (20, 50 et 80 villes).

Les algorithmes implémentés sont :
- Hill Climbing Best Improvement
- Hill Climbing First Improvement
- Multi-Start Hill Climbing
- Recuit Simulé (Simulated Annealing)
- Recherche Tabou *(bonus)*

---

## Structure du projet

```
projet_tsp/
│
├── data/                        # Instances TSP générées
│   ├── instance_A_20.json       # Instance A : 20 villes
│   ├── instance_B_50.json       # Instance B : 50 villes
│   └── instance_C_80.json       # Instance C : 80 villes
│
├── results/                     # Résultats générés automatiquement
│   ├── instance_20/             # Graphiques pour 20 villes
│   ├── instance_50/             # Graphiques pour 50 villes
│   ├── instance_80/             # Graphiques pour 80 villes
│   ├── results_20.json          # Données complètes pour 20 villes
│   ├── results_50.json          # Données complètes pour 50 villes
│   └── results_80.json          # Données complètes pour 80 villes
│
├── tsp_model.py                 # Modélisation du TSP (instances, distances)
├── algorithms.py                # Implémentation des métaheuristiques
├── ExperimentRunner.py          # Protocole expérimental et analyse
├── main.py                      # Point d'entrée principal
└── README.md                    # Ce fichier
```

---

## Prérequis

Python 3.8 ou supérieur est requis. Installez les dépendances avec :

```bash
pip install numpy matplotlib seaborn
```

---

## Exécution

### Option 1 — Lancer tout automatiquement (recommandé)

```bash
python main.py
```

Ce script effectue dans l'ordre :
1. Génération des 3 instances TSP dans `data/`
2. Lancement de tous les algorithmes sur les 3 instances
3. Sauvegarde des résultats dans `results/`

### Option 2 — Étape par étape

**Étape 1 : Générer les instances**
```bash
python tsp_model.py
```

**Étape 2 : Lancer les expériences**
```bash
python ExperimentRunner.py
```

---

## Instances TSP

Les instances sont générées aléatoirement avec des coordonnées dans le carré [0, 100] × [0, 100]. Les seeds sont fixées pour garantir la reproductibilité des résultats.

| Instance | Nombre de villes | Seed |
|----------|-----------------|------|
| A        | 20              | 42   |
| B        | 50              | 123  |
| C        | 80              | 456  |

---

## Algorithmes et paramètres

| Algorithme | Paramètres principaux |
|---|---|
| HC Best Improvement | max_evals adaptatif, voisinage swap |
| HC First Improvement | max_evals adaptatif, voisinage swap |
| Multi-Start HC | 20 départs, HC best, budget dynamique |
| Recuit Simulé | T0=100, α=0.95, Tmin=0.01, palier=100 |
| Recherche Tabou | tenure=20, critère d'aspiration activé |

**Budget d'évaluations selon la taille de l'instance :**

| Instance | Budget (max_evals) | Nombre de runs |
|----------|-------------------|----------------|
| 20 villes | 5 000            | 30             |
| 50 villes | 10 000           | 20             |
| 80 villes | 15 000           | 15             |

---

## Résultats générés

Après exécution, le dossier `results/` contiendra pour chaque instance :

- **`boxplot_XX.png`** : Distribution des coûts par algorithme
- **`barplot_XX.png`** : Comparaison meilleur coût vs coût moyen
- **`best_tour_XX.png`** : Visualisation du meilleur tour trouvé
- **`results_XX.json`** : Toutes les statistiques (meilleur, moyen, écart-type, temps)

---

## Mesures collectées

Pour chaque algorithme et chaque instance, les métriques suivantes sont calculées sur l'ensemble des runs indépendants :

- Meilleur coût obtenu
- Coût moyen et écart-type
- Temps moyen d'exécution

---

## Auteurs

Projet réalisé dans le cadre du module **Optimisation / Métaheuristiques**  
Par: AIT EL MAHJOUB ABDESSAMAD.
     EL MAKAOUI ACHRAF.
     IDHAMIDE ABOUBAKER.
ENSET Mohammedia — Année universitaire 2025-2026
