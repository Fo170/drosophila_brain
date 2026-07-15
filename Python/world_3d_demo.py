"""
world_3d_demo.py
Démonstration du monde virtuel 3D avec insecte en mouvement
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/mnt/agents/output/drosophila_brain')

from config import DT
from core.network import BrainNetwork
from learning.reinforcement_learning import DANModulatedLearning
from world.world_3d import VirtualWorld3D


def run_world_3d_demo(duration_ms=5000.0, verbose=True):
    """
    Lance une démonstration du monde 3D avec l'insecte qui se déplace.

    L'insecte:
        1. Reçoit des stimuli sensoriels du monde (odeur, nourriture, danger)
        2. Le cerveau traite ces stimuli et génère des commandes motrices
        3. L'insecte se déplace dans le monde
        4. Si nourriture trouvée → récompense → apprentissage
        5. Si danger rencontré → punition → apprentissage
    """

    print("=" * 70)
    print("  DÉMONSTRATION MONDE 3D - Drosophila Larva")
    print("=" * 70)

    # 1. Créer le monde
    print("\n[1/4] Création du monde virtuel 3D...")
    world = VirtualWorld3D(size=(50, 50, 20), seed=42)

    print(f"  Taille du monde: {world.size}")
    print(f"  Sources d'odeur: {len(world.odor_sources)}")
    print(f"  Sources de nourriture: {len(world.food_sources)}")
    print(f"  Zones de danger: {len(world.threat_zones)}")
    print(f"  Obstacles: {len(world.obstacles)}")

    # 2. Créer le cerveau
    print("\n[2/4] Création du cerveau...")
    network = BrainNetwork(seed=42)
    learner = DANModulatedLearning(network)

    # 3. Boucle de simulation
    print("\n[3/4] Simulation en cours...")
    print("  (L'insecte explore le monde, apprend des récompenses/punitions)")
    print()

    n_steps = int(duration_ms / DT)
    log_interval = 500  # Afficher un log tous les 500 pas

    for step in range(n_steps):
        # --- PHASE 1: Stimuli du monde vers le cerveau ---
        stimuli = world.get_sensory_input_3d()

        # Appliquer les stimuli au cerveau
        if stimuli['olfactory']['total'] > 0.1:
            network.apply_stimulus('olfactory', 
                                 stimuli['olfactory']['total'], 10)
        if stimuli['gustatory'] > 0.1:
            network.apply_stimulus('gustatory', stimuli['gustatory'], 10)
        if stimuli['thermal'] > 0.1:
            network.apply_stimulus('thermal', stimuli['thermal'], 10)
        if stimuli['visual'] > 0.3:
            network.apply_stimulus('visual', stimuli['visual'], 10)
        if stimuli['mechanosensory'] > 0.1:
            network.apply_stimulus('mechano', stimuli['mechanosensory'], 10)

        # --- PHASE 2: Propagation cérébrale ---
        network.step(DT)

        # --- PHASE 3: Commande motrice ---
        # Récupérer l'activité des DNVNC (commandes locomotrices)
        dn_vnc = network.dn_neurons['VNC']
        if len(dn_vnc) > 0:
            left_act = np.mean([n.output for n in dn_vnc[:len(dn_vnc)//2]])
            right_act = np.mean([n.output for n in dn_vnc[len(dn_vnc)//2:]])

            speed = (left_act + right_act) * 1.5
            turn = (right_act - left_act) * np.pi * 0.5
        else:
            # Comportement aléatoire si pas de commande
            speed = 0.3 + np.random.normal(0, 0.1)
            turn = np.random.normal(0, 0.3)

        # --- PHASE 4: Déplacement dans le monde ---
        world.move_insect_3d(speed, turn, 0.0)

        # --- PHASE 5: Interactions et apprentissage ---
        events = world.check_interactions_3d()

        if events['reward'] > 0:
            learner.apply_reward(events['reward'])
            if verbose:
                print(f"  [t={step*DT:.0f}ms] 🍬 NOURRITURE! Récompense={events['reward']:.2f}")

        if events['punishment'] > 0:
            learner.apply_punishment(events['punishment'])
            if verbose:
                print(f"  [t={step*DT:.0f}ms] 🔥 DANGER! Punition={events['punishment']:.2f}")

        # --- LOG ---
        if step % log_interval == 0 and step > 0:
            state = world.get_state()
            region_act = network.get_region_activity()
            print(f"  [t={step*DT:.0f}ms] Pos=({state['insect_pos'][0]:.1f}, "
                  f"{state['insect_pos'][1]:.1f}, {state['insect_pos'][2]:.1f}) | "
                  f"Food={state['food_consumed']} | "
                  f"Sens={region_act['sensory']:.3f} | "
                  f"MB={region_act['KC']:.3f} | "
                  f"Out={region_act['DN_VNC']:.3f}")

    # 4. Résultats
    print("\n[4/4] Résultats finaux:")
    final_state = world.get_state()

    print(f"\n  Position finale: ({final_state['insect_pos'][0]:.1f}, "
          f"{final_state['insect_pos'][1]:.1f}, {final_state['insect_pos'][2]:.1f})")
    print(f"  Distance parcourue: {final_state['distance_traveled']:.1f} unités")
    print(f"  Nourriture consommée: {final_state['food_consumed']}")
    print(f"  Menaces rencontrées: {final_state['threats_encountered']}")
    print(f"  Interactions totales: {final_state['n_interactions']}")

    # Apprentissage
    learning_summary = learner.get_learning_summary()
    print(f"\n  Apprentissage:")
    print(f"    Signaux DAN: {learning_summary['n_dan_signals']}")
    print(f"    DAN moyen: {learning_summary['mean_dan']:.4f}")

    return world, network, learner


def visualize_trajectory(world, save_path=None):
    """
    Visualise la trajectoire de l'insecte dans le monde 3D.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')

        # Trajectoire
        traj = np.array(world.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
               'c-', alpha=0.6, linewidth=1, label='Trajectoire')

        # Points de départ et arrivée
        ax.scatter(*traj[0], c='green', s=100, marker='o', label='Départ')
        ax.scatter(*traj[-1], c='red', s=100, marker='s', label='Arrivée')

        # Sources d'odeur
        for odor in world.odor_sources:
            ax.scatter(*odor['pos'], c='lime', s=50, marker='*', alpha=0.7)

        # Nourriture
        for food in world.food_sources:
            color = 'gold' if not food['consumed'] else 'gray'
            ax.scatter(*food['pos'], c=color, s=80, marker='o', alpha=0.8)

        # Danger
        for threat in world.threat_zones:
            ax.scatter(*threat['pos'], c='red', s=80, marker='X', alpha=0.7)

        # Obstacles
        for obs in world.obstacles:
            ax.scatter(*obs['pos'], c='white', s=30, marker='^', alpha=0.3)

        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.tick_params(colors='white')

        ax.set_title('Trajectoire de la larve de Drosophila dans le monde 3D\n'
                    f'Distance: {world.distance_traveled:.1f} | '
                    f'Nourriture: {world.food_consumed} | '
                    f'Menaces: {world.threats_encountered}',
                    color='white', fontsize=12)

        ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#1a1a2e')
            print(f"\n✓ Trajectoire sauvegardée: {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("⚠ matplotlib requis pour la visualisation")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Monde 3D Drosophila')
    parser.add_argument('--duration', type=float, default=5000.0,
                       help='Durée de la simulation en ms (défaut: 5000)')
    parser.add_argument('--quiet', action='store_true',
                       help='Mode silencieux (moins de logs)')
    parser.add_argument('--save', type=str, default=None,
                       help='Sauvegarder la trajectoire (ex: trajectoire.png)')

    args = parser.parse_args()

    world, network, learner = run_world_3d_demo(
        duration_ms=args.duration,
        verbose=not args.quiet
    )

    if args.save:
        visualize_trajectory(world, args.save)
    else:
        # Demander si on veut visualiser
        response = input("\nVisualiser la trajectoire? (o/n): ").lower()
        if response in ['o', 'oui', 'y', 'yes']:
            visualize_trajectory(world)
