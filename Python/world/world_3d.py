"""
world/world_3d.py
Monde virtuel 3D pour l'insecte
Environnement avec odeurs, nourriture, obstacles, prédateurs
"""

import numpy as np


class VirtualWorld3D:
    """
    Monde virtuel 3D où évolue la larve de Drosophila.

    Entités:
        - Sources d'odeur (attractives ou répulsives)
        - Nourriture (sucre, levure)
        - Zones de danger (chaleur, dessiccation)
        - Obstacles (parois, autres larves)
        - Lumière (phototaxie)

    La larve se déplace dans ce monde et son cerveau reçoit
    les stimuli sensoriels correspondants.
    """

    def __init__(self, size=(50, 50, 20), seed=42):
        np.random.seed(seed)
        self.size = size
        self.width, self.height, self.depth = size

        # Position de la larve
        self.insect_pos = np.array([self.width/2, self.height/2, self.depth/2])
        self.insect_orientation = np.array([1.0, 0.0, 0.0])  # Direction
        self.insect_speed = 0.0

        # Entités du monde
        self.odor_sources = self._generate_odor_sources(8)
        self.food_sources = self._generate_food_sources(5)
        self.threat_zones = self._generate_threat_zones(3)
        self.light_source = np.array([self.width, self.height, self.depth])
        self.obstacles = self._generate_obstacles(10)

        # Historique
        self.trajectory = [self.insect_pos.copy()]
        self.interactions_log = []

        # Métriques
        self.food_consumed = 0
        self.threats_encountered = 0
        self.distance_traveled = 0.0

    def _generate_odor_sources(self, n):
        """Génère des sources d'odeur."""
        sources = []
        for i in range(n):
            sources.append({
                'pos': np.random.uniform([2, 2, 2], [self.width-2, self.height-2, self.depth-2]),
                'intensity': np.random.uniform(0.5, 1.0),
                'decay': np.random.uniform(0.02, 0.1),
                'type': np.random.choice(['attractive', 'aversive', 'neutral']),
                'quality': np.random.choice(['food', 'danger', 'mate', 'home'])
            })
        return sources

    def _generate_food_sources(self, n):
        """Génère des sources de nourriture."""
        sources = []
        for i in range(n):
            sources.append({
                'pos': np.random.uniform([5, 5, 2], [self.width-5, self.height-5, self.depth-2]),
                'reward': np.random.uniform(0.7, 1.0),
                'radius': np.random.uniform(2, 5),
                'nutrient_type': np.random.choice(['sugar', 'yeast', 'protein']),
                'consumed': False,
                'amount': np.random.uniform(0.5, 1.0)
            })
        return sources

    def _generate_threat_zones(self, n):
        """Génère des zones de danger."""
        zones = []
        for i in range(n):
            zones.append({
                'pos': np.random.uniform([3, 3, 1], [self.width-3, self.height-3, self.depth-1]),
                'punish': np.random.uniform(0.5, 1.0),
                'radius': np.random.uniform(3, 8),
                'type': np.random.choice(['heat', 'desiccation', 'toxin', 'predator']),
                'intensity': np.random.uniform(0.3, 1.0)
            })
        return zones

    def _generate_obstacles(self, n):
        """Génère des obstacles."""
        obstacles = []
        for i in range(n):
            obstacles.append({
                'pos': np.random.uniform([0, 0, 0], [self.width, self.height, self.depth]),
                'size': np.random.uniform([1, 1, 1], [5, 5, 3]),
                'type': 'wall'
            })
        return obstacles

    def get_sensory_input_3d(self):
        """
        Calcule les stimuli sensoriels 3D à la position actuelle.

        Returns:
            Dict avec intensités par modalité
        """
        stimuli = {
            'olfactory': {'total': 0.0, 'attractive': 0.0, 'aversive': 0.0},
            'gustatory': 0.0,
            'visual': 0.0,
            'thermal': 0.0,
            'mechanosensory': 0.0,
            'proprioceptive': 0.0
        }

        # Olfaction 3D
        for odor in self.odor_sources:
            dist = np.linalg.norm(self.insect_pos - odor['pos'])
            intensity = odor['intensity'] * np.exp(-dist * odor['decay'])

            stimuli['olfactory']['total'] += intensity
            if odor['type'] == 'attractive':
                stimuli['olfactory']['attractive'] += intensity
            elif odor['type'] == 'aversive':
                stimuli['olfactory']['aversive'] += intensity

        stimuli['olfactory']['total'] = min(stimuli['olfactory']['total'], 1.0)

        # Gustation
        for food in self.food_sources:
            if not food['consumed']:
                dist = np.linalg.norm(self.insect_pos - food['pos'])
                if dist < food['radius']:
                    stimuli['gustatory'] = max(stimuli['gustatory'],
                                               food['reward'] * (1 - dist/food['radius']))

        # Thermo
        for threat in self.threat_zones:
            dist = np.linalg.norm(self.insect_pos - threat['pos'])
            if dist < threat['radius']:
                stimuli['thermal'] = max(stimuli['thermal'],
                                        threat['punish'] * (1 - dist/threat['radius']))

        # Visuel (lumière directionnelle)
        light_dir = self.light_source - self.insect_pos
        light_dist = np.linalg.norm(light_dir)
        if light_dist > 0:
            light_dir_norm = light_dir / light_dist
            # Intensité lumineuse selon l'orientation
            alignment = np.dot(self.insect_orientation, light_dir_norm)
            stimuli['visual'] = 0.3 + 0.7 * max(0, alignment) * np.exp(-light_dist * 0.01)

        # Mécano: vitesse
        stimuli['mechanosensory'] = min(self.insect_speed / 2.0, 1.0)

        # Proprioception: orientation 3D
        stimuli['proprioceptive'] = {
            'heading': np.arctan2(self.insect_orientation[1], self.insect_orientation[0]),
            'pitch': np.arcsin(self.insect_orientation[2]),
            'speed': self.insect_speed
        }

        return stimuli

    def move_insect_3d(self, speed, turn_yaw, turn_pitch):
        """
        Déplace l'insecte dans le monde 3D.

        Args:
            speed: Vitesse linéaire
            turn_yaw: Rotation horizontale (radians)
            turn_pitch: Rotation verticale (radians)
        """
        # Rotation
        # Yaw (autour de Z)
        cos_y, sin_y = np.cos(turn_yaw), np.sin(turn_yaw)
        new_x = self.insect_orientation[0] * cos_y - self.insect_orientation[1] * sin_y
        new_y = self.insect_orientation[0] * sin_y + self.insect_orientation[1] * cos_y
        self.insect_orientation[0] = new_x
        self.insect_orientation[1] = new_y

        # Pitch (autour de Y)
        cos_p, sin_p = np.cos(turn_pitch), np.sin(turn_pitch)
        new_x = self.insect_orientation[0] * cos_p + self.insect_orientation[2] * sin_p
        new_z = -self.insect_orientation[0] * sin_p + self.insect_orientation[2] * cos_p
        self.insect_orientation[0] = new_x
        self.insect_orientation[2] = new_z

        # Normaliser
        norm = np.linalg.norm(self.insect_orientation)
        if norm > 0:
            self.insect_orientation /= norm

        # Déplacement
        self.insect_speed = speed
        movement = speed * self.insect_orientation
        new_pos = self.insect_pos + movement

        # Collision avec obstacles
        for obs in self.obstacles:
            obs_min = obs['pos'] - obs['size']/2
            obs_max = obs['pos'] + obs['size']/2
            if np.all(new_pos > obs_min) and np.all(new_pos < obs_max):
                # Rebond
                self.insect_orientation *= -0.5
                new_pos = self.insect_pos + speed * self.insect_orientation

        # Limites du monde
        for i in range(3):
            if new_pos[i] < 0 or new_pos[i] > self.size[i]:
                self.insect_orientation[i] *= -1
                new_pos[i] = np.clip(new_pos[i], 0, self.size[i])

        # Mettre à jour
        self.distance_traveled += np.linalg.norm(new_pos - self.insect_pos)
        self.insect_pos = new_pos
        self.trajectory.append(self.insect_pos.copy())

    def check_interactions_3d(self):
        """Vérifie les interactions avec le monde 3D."""
        events = {
            'reward': 0.0,
            'punishment': 0.0,
            'food_eaten': False,
            'threat_encountered': False,
            'odor_detected': None
        }

        # Nourriture
        for food in self.food_sources:
            if not food['consumed']:
                dist = np.linalg.norm(self.insect_pos - food['pos'])
                if dist < food['radius'] * 0.3:
                    food['consumed'] = True
                    food['amount'] -= 0.2
                    events['reward'] = food['reward']
                    events['food_eaten'] = True
                    self.food_consumed += 1
                    self.interactions_log.append({
                        'type': 'food', 'pos': self.insect_pos.copy(),
                        'reward': food['reward']
                    })

        # Danger
        for threat in self.threat_zones:
            dist = np.linalg.norm(self.insect_pos - threat['pos'])
            if dist < threat['radius'] * 0.5:
                events['punishment'] = max(events['punishment'], threat['punish'] * 0.5)
                events['threat_encountered'] = True
                self.threats_encountered += 1
                self.interactions_log.append({
                    'type': 'threat', 'pos': self.insect_pos.copy(),
                    'punish': threat['punish']
                })

        # Odeur proche
        for odor in self.odor_sources:
            dist = np.linalg.norm(self.insect_pos - odor['pos'])
            if dist < 5.0:
                events['odor_detected'] = odor['type']

        return events

    def get_state(self):
        """Retourne l'état complet du monde."""
        return {
            'insect_pos': self.insect_pos.copy(),
            'insect_orientation': self.insect_orientation.copy(),
            'speed': self.insect_speed,
            'food_remaining': sum(1 for f in self.food_sources if not f['consumed']),
            'food_consumed': self.food_consumed,
            'threats_encountered': self.threats_encountered,
            'distance_traveled': self.distance_traveled,
            'n_interactions': len(self.interactions_log)
        }
