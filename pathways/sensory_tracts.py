"""
pathways/sensory_tracts.py
Faisceaux de neurones sensoriels regroupés
Organisation des inputs en tracts fonctionnels
"""

import numpy as np


class SensoryTracts:
    """
    Organisation des faisceaux sensoriels.

    Architecture:
        Capteurs périphériques
          ↓
        Tracts sensoriels (regroupés par modalité)
          ↓
        Neuropiles de traitement (AL, LH, etc.)

    Modalités:
        - Olfaction: ORN → AL (176 neurones)
        - Gustation: GRN → SEZ/suboesophageal (42 neurones)
        - Vision: PR → centres visuels (29 neurones)
        - Thermo: thermoR → centres thermiques (8 neurones)
        - Mécano: mécanoR → VNC → ANs (10 neurones)
        - Proprioception: proprioR → VNC → ANs (12 neurones)
    """

    def __init__(self, network):
        self.network = network

        self.tracts = {
            'olfactory': {'neurons': [], 'target': 'AL', 'color': 'green'},
            'gustatory': {'neurons': [], 'target': 'SEZ', 'color': 'yellow'},
            'visual': {'neurons': [], 'target': 'VIS', 'color': 'blue'},
            'thermal': {'neurons': [], 'target': 'THERMO', 'color': 'red'},
            'mechanosensory': {'neurons': [], 'target': 'VNC', 'color': 'purple'},
            'proprioceptive': {'neurons': [], 'target': 'VNC', 'color': 'orange'}
        }

        self._organize_tracts()

    def _organize_tracts(self):
        """Organise les neurones sensoriels en faisceaux."""
        # Olfaction
        self.tracts['olfactory']['neurons'] = self.network.sensory_neurons['ORN']

        # Gustation
        self.tracts['gustatory']['neurons'] = self.network.sensory_neurons['GRN']

        # Vision
        self.tracts['visual']['neurons'] = self.network.sensory_neurons['PR']

        # Thermo
        self.tracts['thermal']['neurons'] = self.network.sensory_neurons['thermo']

        # Mécano + proprio (regroupés avec ANs)
        self.tracts['mechanosensory']['neurons'] = (
            self.network.sensory_neurons['mechano'] + 
            [n for n in self.network.sensory_neurons.get('mechano', []) 
             if n.type == 'AN']
        )
        self.tracts['proprioceptive']['neurons'] = self.network.sensory_neurons['proprio']

        # Statistiques
        for name, tract in self.tracts.items():
            print(f"  Tract {name}: {len(tract['neurons'])} neurones → {tract['target']}")

    def get_tract_activity(self, tract_name):
        """Retourne l'activité moyenne d'un faisceau."""
        neurons = self.tracts.get(tract_name, {}).get('neurons', [])
        if not neurons:
            return 0.0
        return np.mean([n.output for n in neurons])

    def get_all_activities(self):
        """Retourne l'activité de tous les faisceaux."""
        return {name: self.get_tract_activity(name) 
                for name in self.tracts.keys()}

    def stimulate_tract(self, tract_name, intensity=0.8, pattern=None):
        """
        Stimule un faisceau sensoriel complet.

        Args:
            tract_name: Nom du faisceau
            intensity: Intensité globale
            pattern: Motif spatial (None = uniforme)
        """
        neurons = self.tracts.get(tract_name, {}).get('neurons', [])

        for i, neuron in enumerate(neurons):
            if pattern and i < len(pattern):
                neuron.V = intensity * pattern[i]
            else:
                neuron.V = intensity
            neuron.output = neuron.sigmoid(neuron.V)

    def get_tract_statistics(self):
        """Statistiques des faisceaux."""
        stats = {}
        for name, tract in self.tracts.items():
            neurons = tract['neurons']
            if neurons:
                stats[name] = {
                    'n_neurons': len(neurons),
                    'mean_activity': np.mean([n.output for n in neurons]),
                    'max_activity': max([n.output for n in neurons]),
                    'active_fraction': sum(1 for n in neurons if n.output > 0.3) / len(neurons)
                }
        return stats
