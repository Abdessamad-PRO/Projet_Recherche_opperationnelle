"""
FICHIER PRINCIPAL DU PROJET TSP
Exécutez ce fichier pour générer les instances et lancer toutes les expériences
"""

import os
import sys



def main():
    print("=" * 100)
    print("PROJET TSP - COMPARAISON DE MÉTAHEURISTIQUES")
    print("Université Hassan II - ENSET Mohammedia")
    print("=" * 100)

    # ===== ÉTAPE 1 : GÉNÉRER LES INSTANCES =====
    print("\n[ÉTAPE 1/2] Génération des instances TSP...")
    print("-" * 100)

    try:
        from tsp_model import generate_instances
        generate_instances()
        print("✅ Instances générées avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors de la génération des instances: {e}")
        return

    # ===== ÉTAPE 2 : LANCER LES EXPÉRIMENTATIONS =====
    print("\n[ÉTAPE 2/2] Lancement des expérimentations...")
    print("-" * 100)
    print("\nCela peut prendre plusieurs minutes selon votre ordinateur...")
    print("Appuyez sur Entrée pour continuer ou Ctrl+C pour annuler...")

    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Expérimentation annulée par l'utilisateur.")
        return

    try:
        from ExperimentRunner import run_complete_experiment
        run_complete_experiment()

        print("\n" + "=" * 100)
        print("✅ EXPÉRIMENTATION TERMINÉE AVEC SUCCÈS!")
        print("=" * 100)
        print("\n📁 Résultats disponibles dans le dossier 'results/':")
        print("   - results_XX.json     : Données complètes")

    except ImportError as e:
        print(f"\n❌ Erreur d'importation: {e}")
        print("\n📌 Vérifiez que tous les fichiers sont présents:")
        print("   - tsp_instances.py")
        print("   - tsp_algorithms.py")
        print("   - tsp_experiments_light.py")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'expérimentation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()