"""
main.py
Point d'entrée du simulateur de cerveau de Drosophila melanogaster
Lancement de simulation + monde virtuel
"""

import sys
import time
import numpy as np

# Ajouter le répertoire courant au path
sys.path.insert(0, '/mnt/agents/output/drosophila_brain')

from config import DT, T_MAX, N_STEPS
from core.network import BrainNetwork
from data.connectome_synthetic import SyntheticConnectome
from learning.reinforcement_learning import DANModulatedLearning
from visualization.real_time_plot import RealTimeVisualizer
from world.world import VirtualWorld


class DrosophilaSimulator:
    """
    Simulateur complet: cerveau + monde + apprentissage + visualisation.

    Pipeline:
        1. Créer le réseau cérébral (3016 neurones)
        2. Connecter avec le monde virtuel
        3. Boucle: stimulus → cerveau → action → récompense → apprentissage
        4. Visualiser en temps réel
    """

    def __init__(self, use_synthetic_connectome=True, seed=42):
        print("=" * 60)
        print("  SIMULATEUR CERVEAU DE DROSOPHILA MELANOGASTER")
        print("  Basé sur Winding et al. 2023 (Science)")
        print("  3016 neurones | 548000 synapses | Apprentissage DAN-modulé")
        print("=" * 60)

        np.random.seed(seed)

        # 1. Créer le réseau cérébral
        print("\n[1/4] Initialisation du réseau cérébral...")
        self.network = BrainNetwork(seed=seed)

        # 2. Charger la matrice de connectivité
        if use_synthetic_connectome:
            print("[2/4] Génération de la matrice de connectivité synthétique...")
            self.connectome = SyntheticConnectome(seed=seed)
            stats = self.connectome.get_stats()
            print(f"      Statistiques de la matrice:")
            for key, val in stats.items():
                print(f"        {key}: {val}")
            self.connectome.export_to_network(self.network)

        # 3. Initialiser l'apprentissage
        print("[3/4] Initialisation de l'apprentissage DAN-modulé...")
        self.learning = DANModulatedLearning(self.network)

        # 4. Créer le monde virtuel
        print("[4/4] Création du monde virtuel...")
        self.world = VirtualWorld(seed=seed)

        # 5. Visualisation (optionnelle)
        self.visualizer = None

        print("\n✓ Simulateur prêt!")
        print(f"  Neurones: {len(self.network.neurons)}")
        print(f"  Synapses: {len(self.network.synapses)}")

    def run_simulation(self, duration_ms=1000.0, with_visualization=False, 
                      stimulus_sequence=None):
        """
        Exécute une simulation complète.

        Args:
            duration_ms: Durée en millisecondes
            with_visualization: Active l'affichage temps réel
            stimulus_sequence: Liste de stimuli à appliquer
                [{'type': 'olfactory', 'intensity': 0.8, 'time': 100}, ...]
        """
        n_steps = int(duration_ms / DT)

        print(f"\n▶ Démarrage simulation: {duration_ms}ms ({n_steps} pas)")

        # Initialiser la visualisation si demandée
        if with_visualization:
            try:
                self.visualizer = RealTimeVisualizer(self.network)
            except Exception as e:
                print(f"⚠ Visualisation non disponible: {e}")
                with_visualization = False

        # Séquence de stimuli par défaut si non fournie
        if stimulus_sequence is None:
            stimulus_sequence = [
                {'type': 'olfactory', 'intensity': 0.8, 'time': 100, 'duration': 200},
                {'type': 'gustatory', 'intensity': 0.6, 'time': 400, 'duration': 100},
            ]

        # Boucle principale
        start_time = time.time()

        for step in range(n_steps):
            current_time = step * DT

            # 1. Appliquer les stimuli programmés
            for stim in stimulus_sequence:
                if abs(current_time - stim['time']) < DT:
                    self.network.apply_stimulus(
                        stim['type'],
                        stim.get('intensity', 0.8),
                        stim.get('duration', 50.0)
                    )
                    print(f"  [t={current_time:.1f}ms] Stimulus: {stim['type']} "
                          f"(I={stim['intensity']})")

            # 2. Obtenir les stimuli du monde virtuel
            world_stimuli = self.world.get_sensory_input()

            # 3. Propagation dans le cerveau
            self.network.step(DT)

            # 4. Vérifier les interactions monde
            events = self.world.check_interactions()

            # 5. Apprentissage si récompense/punition
            if events['reward'] > 0:
                result = self.learning.apply_reward(events['reward'])
                print(f"  [t={current_time:.1f}ms] 🍬 RÉCOMPENSE! "
                      f"DAN={result['dan_signal']:.3f}, "
                      f"Δw={result['total_delta']:.6f}")

            if events['punishment'] > 0:
                result = self.learning.apply_punishment(events['punishment'])
                print(f"  [t={current_time:.1f}ms] 🔥 PUNITION! "
                      f"DAN={result['dan_signal']:.3f}, "
                      f"Δw={result['total_delta']:.6f}")

            # 6. Mettre à jour la visualisation
            if with_visualization and self.visualizer and step % 10 == 0:
                self.visualizer.record_state()
                self.visualizer.update()
                plt.pause(0.001)

        elapsed = time.time() - start_time
        print(f"\n✓ Simulation terminée en {elapsed:.2f}s")

        # Résultats
        final_state = self.network.get_state()
        print(f"\n📊 Résultats:")
        print(f"  Temps simulé: {final_state['time']:.1f}ms")
        print(f"  Neurones actifs: {final_state['active_neurons']}")
        print(f"  Activité moyenne: {final_state['mean_activity']:.4f}")

        region_act = self.network.get_region_activity()
        print(f"\n  Activité par région:")
        for region, act in region_act.items():
            print(f"    {region:12s}: {act:.4f}")

        # Résumé apprentissage
        learning_summary = self.learning.get_learning_summary()
        print(f"\n  Apprentissage:")
        print(f"    Signaux DAN: {learning_summary['n_dan_signals']}")
        print(f"    DAN moyen: {learning_summary['mean_dan']:.4f}")

    def run_olfactory_learning_task(self, n_trials=5):
        """
        Tâche d'apprentissage olfactif classique:
        - Odeur A + sucre → approche
        - Odeur B + choc → fuite
        """
        print("\n" + "=" * 60)
        print("  TÂCHE D'APPRENTISSAGE OLFACTIF")
        print("=" * 60)

        odors = ['olfactory_A', 'olfactory_B']
        outcomes = ['reward', 'punishment']

        for trial in range(n_trials):
            print(f"\n--- Essai {trial + 1}/{n_trials} ---")

            for odor, outcome in zip(odors, outcomes):
                print(f"\n  Présentation: {odor}")

                # Phase 1: Stimulus olfactif
                self.network.reset()
                self.network.apply_stimulus('olfactory', intensity=0.8, duration=100)
                self.network.run(200)

                # Phase 2: Résultat
                if outcome == 'reward':
                    print(f"  → Récompense (sucre)")
                    self.learning.apply_reward(1.0)
                else:
                    print(f"  → Punition (choc)")
                    self.learning.apply_punishment(1.0)

                self.network.run(100)

        print("\n✓ Tâche d'apprentissage terminée")

    def interactive_mode(self):
        """Mode interactif avec commandes utilisateur."""
        print("\n" + "=" * 60)
        print("  MODE INTERACTIF")
        print("  Commandes: stimulus <type> <intensité> | reward | punish | step | quit")
        print("=" * 60)

        while True:
            try:
                cmd = input("\n> ").strip().lower()

                if cmd == 'quit' or cmd == 'q':
                    break
                elif cmd.startswith('stimulus'):
                    parts = cmd.split()
                    stim_type = parts[1] if len(parts) > 1 else 'olfactory'
                    intensity = float(parts[2]) if len(parts) > 2 else 0.8
                    self.network.apply_stimulus(stim_type, intensity, 100)
                    print(f"  Stimulus {stim_type} appliqué (I={intensity})")

                elif cmd == 'reward' or cmd == 'r':
                    result = self.learning.apply_reward(1.0)
                    print(f"  Récompense! DAN={result['dan_signal']:.3f}")

                elif cmd == 'punish' or cmd == 'p':
                    result = self.learning.apply_punishment(1.0)
                    print(f"  Punition! DAN={result['dan_signal']:.3f}")

                elif cmd == 'step' or cmd == 's':
                    self.network.step(DT)
                    state = self.network.get_state()
                    print(f"  t={state['time']:.1f}ms | "
                          f"actifs={state['active_neurons']} | "
                          f"moy={state['mean_activity']:.4f}")

                elif cmd == 'status':
                    state = self.network.get_state()
                    for k, v in state.items():
                        print(f"  {k}: {v}")

                else:
                    print("  Commandes: stimulus <type> <I> | reward | punish | step | status | quit")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  Erreur: {e}")

        print("\n✓ Mode interactif terminé")


def demo():
    """Démonstration rapide du simulateur."""
    sim = DrosophilaSimulator(use_synthetic_connectome=True)

    # Simulation simple avec stimulus olfactif
    sim.run_simulation(
        duration_ms=500.0,
        with_visualization=False,
        stimulus_sequence=[
            {'type': 'olfactory', 'intensity': 0.9, 'time': 50, 'duration': 150},
            {'type': 'gustatory', 'intensity': 0.7, 'time': 300, 'duration': 100},
        ]
    )

    # Tâche d'apprentissage
    sim.run_olfactory_learning_task(n_trials=3)

    print("\n" + "=" * 60)
    print("  DÉMONSTRATION TERMINÉE")
    print("=" * 60)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Vérifier les arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        sim = DrosophilaSimulator()
        sim.interactive_mode()
    elif len(sys.argv) > 1 and sys.argv[1] == '--visual':
        sim = DrosophilaSimulator()
        sim.run_simulation(duration_ms=1000.0, with_visualization=True)
    else:
        demo()
