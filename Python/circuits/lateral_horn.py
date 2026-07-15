"""
circuits/lateral_horn.py
Circuit de la Corne Latérale (LH) - Valeurs innées
Intègre les informations sensorielles pour les réponses instinctives
"""

import numpy as np


class LateralHornCircuit:
    """
    Corne Latérale - Centre des valeurs innées (non apprises).

    Architecture:
        PN (toutes modalités)
          ↓
        LHN (Lateral Horn Neurons) - ~50 neurones
          ↓
        CN (Convergence Neurons) - intègrent avec MB

    Fonction:
        - Réponses instinctives à l'odeur (ex: répulsion CO₂)
        - Intégration multimodale (odeur + lumière)
        - Pas de plasticité (contrairement au MB)
    """

    def __init__(self, network):
        self.network = network
        self.lhn_neurons = [n for n in network.interneurons if n.type == 'LHN']

        # Organiser par fonction
        self.innate_valence = {
            'attractive': [],  # LHN pour stimuli attractifs
            'aversive': [],    # LHN pour stimuli répulsifs
            'neutral': []      # LHN neutres
        }
        self._organize_lhn()

    def _organize_lhn(self):
        """Organise les LHN par valence innée."""
        for i, lhn in enumerate(self.lhn_neurons):
            if i < len(self.lhn_neurons) // 3:
                self.innate_valence['attractive'].append(lhn)
                lhn.compartment = 'LH_attractive'
            elif i < 2 * len(self.lhn_neurons) // 3:
                self.innate_valence['aversive'].append(lhn)
                lhn.compartment = 'LH_aversive'
            else:
                self.innate_valence['neutral'].append(lhn)
                lhn.compartment = 'LH_neutral'

    def get_innate_response(self, stimulus_type):
        """
        Calcule la réponse innée à un type de stimulus.

        Returns:
            Valeur positive (attractif) ou négative (répulsif)
        """
        attractive = np.mean([n.output for n in self.innate_valence['attractive']])
        aversive = np.mean([n.output for n in self.innate_valence['aversive']])
        return attractive - aversive

    def get_activity(self):
        """Retourne l'activité par catégorie de valence."""
        return {
            'attractive': np.mean([n.output for n in self.innate_valence['attractive']]),
            'aversive': np.mean([n.output for n in self.innate_valence['aversive']]),
            'neutral': np.mean([n.output for n in self.innate_valence['neutral']])
        }
