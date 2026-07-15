"""
world/world.py
Monde virtuel 2D pour l'insecte simulé
Sources d'odeur, nourriture, danger
"""

import numpy as np
from config import WORLD_SIZE, N_ODOR_SOURCES, N_FOOD_SOURCES, N_THREAT_ZONES


class VirtualWorld:
    """
    Monde virtuel 2D où évolue la larve de Drosophila.

    Contient:
        - Sources d'odeur (attractives)
        - Sources de nourriture (récompense)
        - Zones de danger (punition: chaleur)
        - Position de l'insecte

    Le cerveau reçoit les stimuli sensoriels selon la position.
    """

    def __init__(self, size=WORLD_SIZE, seed=42):
        np.random.seed(seed)
        self.size = size
        self.width, self.height = size

        # Position de l'insecte
        self.insect_pos = np.array([self.width / 2, self.height / 2])
        self.insect_orientation = 0.0  # radians
        self.insect_speed = 0.0

        # Générer les entités du monde
        self.odor_sources = self._generate_sources(N_ODOR_SOURCES, 'odor')
        self.food_sources = self._generate_sources(N_FOOD_SOURCES, 'food')
        self.threat_zones = self._generate_sources(N_THREAT_ZONES, 'threat')

        # Historique de trajectoire
        self.trajectory = [self.insect_pos.copy()]

    def _generate_sources(self, n, source_type):
        """Génère des sources aléatoires dans le monde."""
        sources = []
        for _ in range(n):
            pos = np.array([
                np.random.uniform(5, self.width - 5),
                np.random.uniform(5, self.height - 5)
            ])

            if source_type == 'odor':
                sources.append({
                    'pos': pos,
                    'intensity': np.random.uniform(0.5, 1.0),
                    'decay': np.random.uniform(0.05, 0.15),  # Distance de décroissance
                    'type': 'odor'
                })
            elif source_type == 'food':
                sources.append({
                    'pos': pos,
                    'reward_value': np.random.uniform(0.7, 1.0),
                    'radius': np.random.uniform(3, 8),
                    'consumed': False,
                    'type': 'food'
                })
            elif source_type == 'threat':
                sources.append({
                    'pos': pos,
                    'punish_value': np.random.uniform(0.5, 1.0),
                    'radius': np.random.uniform(5, 12),
                    'type': 'threat'
                })

        return sources

    def get_sensory_input(self):
        """
        Calcule les stimuli sensoriels à la position actuelle de l'insecte.

        Returns:
            Dict avec intensités par modalité sensorielle
        """
        stimuli = {
            'olfactory': 0.0,
            'gustatory': 0.0,
            'visual': 0.0,
            'thermal': 0.0,
            'mechano': 0.0,
            'proprioceptive': 0.0
        }

        # Olfaction: somme des odeurs décroissantes avec distance
        for odor in self.odor_sources:
            dist = np.linalg.norm(self.insect_pos - odor['pos'])
            intensity = odor['intensity'] * np.exp(-dist * odor['decay'])
            stimuli['olfactory'] += intensity

        stimuli['olfactory'] = min(stimuli['olfactory'], 1.0)

        # Gustation: nourriture proche
        for food in self.food_sources:
            if not food['consumed']:
                dist = np.linalg.norm(self.insect_pos - food['pos'])
                if dist < food['radius']:
                    stimuli['gustatory'] = max(stimuli['gustatory'], 
                                               food['reward_value'] * (1 - dist/food['radius']))

        # Thermo: zones de danger
        for threat in self.threat_zones:
            dist = np.linalg.norm(self.insect_pos - threat['pos'])
            if dist < threat['radius']:
                stimuli['thermal'] = max(stimuli['thermal'],
                                        threat['punish_value'] * (1 - dist/threat['radius']))

        # Visuel: lumière ambiante + sources lumineuses (simplifié)
        stimuli['visual'] = 0.3  # Lumière de base

        # Mécano: vitesse de déplacement
        stimuli['mechano'] = min(self.insect_speed / 2.0, 1.0)

        # Proprioception: orientation
        stimuli['proprioceptive'] = (np.sin(self.insect_orientation) + 1) / 2

        return stimuli

    def check_interactions(self):
        """
        Vérifie les interactions avec le monde (nourriture consommée, etc.)

        Returns:
            Dict avec événements (reward, punishment)
        """
        events = {'reward': 0.0, 'punishment': 0.0, 'food_eaten': False}

        # Nourriture
        for food in self.food_sources:
            if not food['consumed']:
                dist = np.linalg.norm(self.insect_pos - food['pos'])
                if dist < food['radius'] * 0.3:  # Très proche = consommation
                    food['consumed'] = True
                    events['reward'] = food['reward_value']
                    events['food_eaten'] = True

        # Danger
        for threat in self.threat_zones:
            dist = np.linalg.norm(self.insect_pos - threat['pos'])
            if dist < threat['radius'] * 0.5:
                events['punishment'] = max(events['punishment'], 
                                          threat['punish_value'] * 0.5)

        return events

    def move_insect(self, speed, turn_angle):
        """
        Déplace l'insecte dans le monde.

        Args:
            speed: Vitesse de déplacement
            turn_angle: Angle de rotation (radians)
        """
        self.insect_orientation += turn_angle
        self.insect_speed = speed

        # Nouvelle position
        dx = speed * np.cos(self.insect_orientation)
        dy = speed * np.sin(self.insect_orientation)

        self.insect_pos += np.array([dx, dy])

        # Limites du monde (rebond)
        if self.insect_pos[0] < 0 or self.insect_pos[0] > self.width:
            self.insect_orientation = np.pi - self.insect_orientation
            self.insect_pos[0] = np.clip(self.insect_pos[0], 0, self.width)
        if self.insect_pos[1] < 0 or self.insect_pos[1] > self.height:
            self.insect_orientation = -self.insect_orientation
            self.insect_pos[1] = np.clip(self.insect_pos[1], 0, self.height)

        self.trajectory.append(self.insect_pos.copy())

    def get_brain_outputs(self):
        """
        Convertit les outputs du cerveau en actions dans le monde.

        Returns:
            (speed, turn_angle) pour move_insect
        """
        # Simplification: les DNVNC contrôlent la locomotion
        # Activité des DNVNC → vitesse et direction
        dn_vnc = self.dn_neurons if hasattr(self, 'dn_neurons') else []

        if len(dn_vnc) > 0:
            left_activity = np.mean([n.output for n in dn_vnc[:len(dn_vnc)//2]])
            right_activity = np.mean([n.output for n in dn_vnc[len(dn_vnc)//2:]])

            speed = (left_activity + right_activity) * 2.0
            turn = (right_activity - left_activity) * np.pi
        else:
            speed = 0.5
            turn = 0.0

        return speed, turn

    def render(self, ax=None):
        """Rendu graphique du monde."""
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_facecolor('#2d3436')
        ax.set_aspect('equal')

        # Sources d'odeur (vert, transparent)
        for odor in self.odor_sources:
            circle = plt.Circle(odor['pos'], 3, color='green', alpha=0.3)
            ax.add_patch(circle)
            ax.plot(odor['pos'][0], odor['pos'][1], 'g*', markersize=15)

        # Nourriture (jaune)
        for food in self.food_sources:
            if not food['consumed']:
                color = 'gold' if not food['consumed'] else 'gray'
                circle = plt.Circle(food['pos'], food['radius'], 
                                  color=color, alpha=0.5)
                ax.add_patch(circle)
                ax.plot(food['pos'][0], food['pos'][1], 'yo', markersize=10)

        # Danger (rouge)
        for threat in self.threat_zones:
            circle = plt.Circle(threat['pos'], threat['radius'], 
                              color='red', alpha=0.3)
            ax.add_patch(circle)
            ax.plot(threat['pos'][0], threat['pos'][1], 'rX', markersize=12)

        # Trajectoire
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], 'w-', alpha=0.5, linewidth=0.5)

        # Insecte (triangle orienté)
        dx = 2 * np.cos(self.insect_orientation)
        dy = 2 * np.sin(self.insect_orientation)
        ax.arrow(self.insect_pos[0], self.insect_pos[1], dx, dy,
                head_width=2, head_length=1.5, fc='cyan', ec='cyan')
        ax.plot(self.insect_pos[0], self.insect_pos[1], 'co', markersize=8)

        ax.set_title('Monde virtuel - Drosophila larva', color='white', fontsize=14)
        ax.tick_params(colors='white')

        return ax
