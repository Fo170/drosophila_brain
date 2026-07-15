"""
main_advanced.py
Simulateur complet intégré avec circuits, pathways, monde 3D
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/mnt/agents/output/drosophila_brain')

from config import DT, T_MAX
from core.network import BrainNetwork
from data.connectome_synthetic import SyntheticConnectome
from learning.reinforcement_learning import DANModulatedLearning

# Circuits
from circuits.antennal_lobe import AntennalLobeCircuit
from circuits.mushroom_body import MushroomBodyCircuit
from circuits.lateral_horn import LateralHornCircuit
from circuits.interhemispheric import InterhemisphericCircuit
from circuits.brain_vnc_interface import BrainVNCInterface

# Pathways
from pathways.sensory_tracts import SensoryTracts
from pathways.motor_tracts import MotorTracts
from pathways.feedback_loops import FeedbackLoops

# Visualisation
from visualization.real_time_plot import RealTimeVisualizer
from visualization.brain_3d import Brain3DVisualizer

# Monde
from world.world_3d import VirtualWorld3D

# Utils
from utils.stats import ConnectomeStats
from utils.io import NetworkIO


class AdvancedDrosophilaSimulator:
    """
    Simulateur avancé avec tous les modules intégrés.
    """

    def __init__(self, seed=42):
        print("=" * 70)
        print("  SIMULATEUR AVANCÉ - Drosophila melanogaster")
        print("  Cerveau complet + Circuits + Monde 3D + Apprentissage")
        print("=" * 70)

        # 1. Réseau cérébral
        print("\n[1/8] Réseau cérébral...")
        self.network = BrainNetwork(seed=seed)

        # 2. Connectome
        print("[2/8] Matrice de connectivité...")
        self.connectome = SyntheticConnectome(seed=seed)
        self.connectome.export_to_network(self.network)

        # 3. Circuits anatomiques
        print("[3/8] Circuits anatomiques...")
        self.al = AntennalLobeCircuit(self.network)
        self.mb = MushroomBodyCircuit(self.network)
        self.lh = LateralHornCircuit(self.network)
        self.interhemi = InterhemisphericCircuit(self.network)
        self.brain_vnc = BrainVNCInterface(self.network)

        # 4. Pathways
        print("[4/8] Faisceaux sensoriels et moteurs...")
        self.sensory_tracts = SensoryTracts(self.network)
        self.motor_tracts = MotorTracts(self.network)
        self.feedback = FeedbackLoops(self.network)

        # 5. Apprentissage
        print("[5/8] Apprentissage DAN-modulé...")
        self.learning = DANModulatedLearning(self.network)

        # 6. Monde 3D
        print("[6/8] Monde virtuel 3D...")
        self.world = VirtualWorld3D(seed=seed)

        # 7. Stats
        print("[7/8] Statistiques...")
        self.stats = ConnectomeStats(self.network)
        self.io = NetworkIO(self.network)

        # 8. Visualisation
        print("[8/8] Visualisation...")
        self.viz_2d = None
        self.viz_3d = None

        print("\n✓ Simulateur avancé prêt!")
        self._print_summary()

    def _print_summary(self):
        """Affiche un résumé du réseau."""
        summary = self.stats.network_summary()
        print(f"\n  Résumé du réseau:")
        print(f"    Neurones: {summary['n_neurons']}")
        print(f"    Synapses: {summary['n_synapses']}")
        print(f"    Degré moyen (in): {summary['mean_in_degree']:.1f}")
        print(f"    Degré moyen (out): {summary['mean_out_degree']:.1f}")
        print(f"    Hubs in-out: {summary['n_in_out_hubs']}")
        print(f"    Clustering: {summary['clustering_coeff']:.4f}")
        print(f"    Densité: {summary['density']:.6f}")

    def run_closed_loop(self, duration_ms=2000.0, with_viz=False):
        """
        Boucle fermée: monde → cerveau → action → monde
        """
        print(f"\n▶ Boucle fermée: {duration_ms}ms")

        n_steps = int(duration_ms / DT)

        if with_viz:
            try:
                self.viz_2d = RealTimeVisualizer(self.network)
            except:
                print("⚠ Visualisation 2D non disponible")

        for step in range(n_steps):
            # 1. Stimuli du monde
            world_stimuli = self.world.get_sensory_input_3d()

            # 2. Appliquer au cerveau
            self._apply_world_stimuli(world_stimuli)

            # 3. Propagation cérébrale
            self.network.step(DT)

            # 4. Commande motrice
            motor_cmd = self.brain_vnc.get_motor_output()

            # 5. Déplacer l'insecte
            self.world.move_insect_3d(
                speed=motor_cmd['speed'] * 0.5,
                turn_yaw=motor_cmd['turn_angle'] * 0.3,
                turn_pitch=0.0
            )

            # 6. Interactions
            events = self.world.check_interactions_3d()

            # 7. Apprentissage
            if events['reward'] > 0:
                self.learning.apply_reward(events['reward'])
            if events['punishment'] > 0:
                self.learning.apply_punishment(events['punishment'])

            # 8. Visualisation
            if with_viz and self.viz_2d and step % 10 == 0:
                self.viz_2d.record_state()
                self.viz_2d.update()
                plt.pause(0.001)

        # Résultats
        world_state = self.world.get_state()
        print(f"\n📊 Résultats de la boucle fermée:")
        print(f"  Distance parcourue: {world_state['distance_traveled']:.1f}")
        print(f"  Nourriture consommée: {world_state['food_consumed']}")
        print(f"  Menaces rencontrées: {world_state['threats_encountered']}")
        print(f"  Interactions: {world_state['n_interactions']}")

    def _apply_world_stimuli(self, stimuli):
        """Applique les stimuli du monde au cerveau."""
        # Olfaction
        if stimuli['olfactory']['total'] > 0.1:
            self.network.apply_stimulus('olfactory', 
                                       stimuli['olfactory']['total'], 50)

        # Gustation
        if stimuli['gustatory'] > 0.1:
            self.network.apply_stimulus('gustatory', stimuli['gustatory'], 50)

        # Thermo
        if stimuli['thermal'] > 0.1:
            self.network.apply_stimulus('thermal', stimuli['thermal'], 50)

        # Visuel
        if stimuli['visual'] > 0.3:
            self.network.apply_stimulus('visual', stimuli['visual'], 50)

        # Mécano
        if stimuli['mechanosensory'] > 0.1:
            self.network.apply_stimulus('mechano', stimuli['mechanosensory'], 50)

    def run_olfactory_learning(self, n_trials=10):
        """Tâche d'apprentissage olfactif avancée."""
        print("\n" + "=" * 70)
        print("  TÂCHE D'APPRENTISSAGE OLFACTIF")
        print("=" * 70)

        odors = ['A', 'B']
        results = {odor: [] for odor in odors}

        for trial in range(n_trials):
            print(f"\n--- Essai {trial + 1}/{n_trials} ---")

            for i, odor in enumerate(odors):
                print(f"\n  Odeur {odor}:")

                # Phase 1: Présentation
                self.network.reset()

                # Motif d'odeur spécifique
                pattern = [0.8 if j % 2 == i else 0.2 for j in range(15)]
                self.al.present_odor(pattern, intensity=0.8)

                # Propagation
                for _ in range(100):
                    self.network.step(DT)

                # Phase 2: Résultat
                if i == 0:  # Odeur A = récompense
                    print("    → Récompense (sucre)")
                    result = self.learning.apply_reward(1.0)
                else:  # Odeur B = punition
                    print("    → Punition (choc)")
                    result = self.learning.apply_punishment(1.0)

                # Mesurer la valence apprise
                valence = self.mb.get_mbon_valence()
                results[odor].append(valence)
                print(f"    Valence MBON: {valence:.3f}")

        # Analyse
        print(f"\n📊 Résultats de l'apprentissage:")
        for odor, vals in results.items():
            print(f"  Odeur {odor}: moyenne={np.mean(vals):.3f}, "
                  f"tendance={vals[-1] - vals[0]:+.3f}")

    def visualize_brain_3d(self, save_path=None):
        """Visualise le cerveau en 3D."""
        print("\n▶ Génération de la visualisation 3D...")
        self.viz_3d = Brain3DVisualizer(self.network)
        self.viz_3d.plot_neurons()
        self.viz_3d.plot_region_boundaries()

        if save_path:
            self.viz_3d.save(save_path)
        else:
            self.viz_3d.show()

    def save_simulation(self, filepath):
        """Sauvegarde l'état complet."""
        self.io.save_state(filepath)

    def get_full_report(self):
        """Génère un rapport complet."""
        report = {
            'network': self.stats.network_summary(),
            'circuits': {
                'AL_glomeruli': len(self.al.glomeruli),
                'MB_compartments': len(self.mb.kc_compartments),
                'LH_valence': self.lh.get_innate_response('olfactory'),
                'interhemispheric_pairs': len(self.interhemi.contralateral_pairs)
            },
            'pathways': {
                'sensory_tracts': list(self.sensory_tracts.tracts.keys()),
                'motor_commands': list(self.motor_tracts.locomotion_commands.keys())
            },
            'learning': self.learning.get_learning_summary(),
            'world': self.world.get_state()
        }
        return report


def demo_advanced():
    """Démonstration complète du simulateur avancé."""
    sim = AdvancedDrosophilaSimulator(seed=42)

    # 1. Boucle fermée simple
    print("\n" + "=" * 70)
    print("  DÉMONSTRATION 1: Boucle fermée monde-cerveau")
    print("=" * 70)
    sim.run_closed_loop(duration_ms=500.0, with_viz=False)

    # 2. Apprentissage olfactif
    print("\n" + "=" * 70)
    print("  DÉMONSTRATION 2: Apprentissage olfactif")
    print("=" * 70)
    sim.run_olfactory_learning(n_trials=5)

    # 3. Rapport final
    print("\n" + "=" * 70)
    print("  RAPPORT FINAL")
    print("=" * 70)
    report = sim.get_full_report()

    print(f"\n  Réseau:")
    for k, v in report['network'].items():
        print(f"    {k}: {v}")

    print(f"\n  Circuits:")
    for k, v in report['circuits'].items():
        print(f"    {k}: {v}")

    print(f"\n  Apprentissage:")
    for k, v in report['learning'].items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 70)
    print("  DÉMONSTRATION TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    demo_advanced()
