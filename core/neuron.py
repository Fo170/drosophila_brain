"""
core/neuron.py
Classe Neurone : Sigmoïde + Fuite Temporelle
Modèle hybride : potentiel de membrane avec fuite + sigmoïde pour sortie continue
"""

import numpy as np
from config import (
    TAU_MEMBRANE, V_REST, V_THRESHOLD, TAU_REFRACTORY,
    SIGMOID_STEEPNESS, SIGMOID_OFFSET, DT
)


class Neuron:
    """
    Neurone avec dynamique temporelle réaliste.

    Architecture:
        entrées ──[poids synaptiques]──→ Σ ──→ V_membrane ──→ σ(V) ──→ sortie
                                                     ↑
                                              fuite : dV/dt = -(V-Vrest)/τ

    La fuite temporelle crée une mémoire courte du potentiel, permettant
    les oscillations et la dynamique récurrente.
    """

    def __init__(self, neuron_id, neuron_type, hemisphere='left', compartment=None):
        """
        Args:
            neuron_id: Identifiant unique (0-3015)
            neuron_type: Type cellulaire (ex: 'ORN', 'KC', 'MBON', 'DAN', etc.)
            hemisphere: 'left' ou 'right'
            compartment: Sous-compartiment (ex: glomérule, zone MB, etc.)
        """
        self.id = neuron_id
        self.type = neuron_type
        self.hemisphere = hemisphere
        self.compartment = compartment

        # État dynamique
        self.V = V_REST              # Potentiel de membrane (fuite)
        self.output = 0.0            # Sortie sigmoïde [0,1]
        self.refractory_timer = 0.0  # Timer période réfractaire
        self.spike_history = []      # Historique des spikes (pour STDP)

        # Connexions
        self.presynaptic = []        # Liste de (neurone_pré, poids, type_synapse)
        self.postsynaptic = []       # Liste de (neurone_post, poids, type_synapse)

        # Métadonnées
        self.total_input_current = 0.0
        self.is_active = False

    def sigmoid(self, x):
        """Fonction d'activation sigmoïde standard."""
        return 1.0 / (1.0 + np.exp(-SIGMOID_STEEPNESS * (x - V_THRESHOLD + SIGMOID_OFFSET)))

    def sigmoid_derivative(self, x):
        """Dérivée de la sigmoïde pour rétropropagation."""
        s = self.sigmoid(x)
        return s * (1.0 - s) * SIGMOID_STEEPNESS

    def compute_input_current(self):
        """
        Calcule le courant synaptique total entrant.
        Somme pondérée des sorties des neurones présynaptiques.
        """
        I_syn = 0.0
        for pre_neuron, weight, syn_type in self.presynaptic:
            # Chaque type de synapse modulate l'efficacité
            if syn_type == 'a_d':
                # Axo-dendritique : transmission standard
                I_syn += weight * pre_neuron.output
            elif syn_type == 'a_a':
                # Axo-axonique : modulation de l'axon (gain)
                I_syn += weight * pre_neuron.output * 0.7
            elif syn_type == 'd_d':
                # Dendro-dendritique : interaction locale
                I_syn += weight * pre_neuron.output * 0.5
            elif syn_type == 'd_a':
                # Dendro-axonique : feedback vers l'axon
                I_syn += weight * pre_neuron.output * 0.4

        self.total_input_current = I_syn
        return I_syn

    def update(self, dt=DT):
        """
        Met à jour l'état du neurone pour un pas de temps dt.

        Équation : dV/dt = -(V - V_rest)/τ + I_syn
        """
        if self.refractory_timer > 0:
            self.refractory_timer -= dt
            self.V = V_REST * 0.5  # Hyperpolarisation partielle
            self.output = self.sigmoid(self.V)
            return self.output

        # Courant synaptique entrant
        I_syn = self.compute_input_current()

        # Équation de fuite (intégration temporelle)
        dV = (-(self.V - V_REST) / TAU_MEMBRANE + I_syn) * dt
        self.V += dV

        # Limites physiques
        self.V = np.clip(self.V, -2.0, 2.0)

        # Activation sigmoïde
        self.output = self.sigmoid(self.V)

        # Détection de "spike" (pour STDP et historique)
        if self.V > V_THRESHOLD + 0.3 and self.refractory_timer <= 0:
            self.spike_history.append(0)  # Timestamp relatif géré par le réseau
            self.refractory_timer = TAU_REFRACTORY
            self.is_active = True
        else:
            self.is_active = False

        return self.output

    def add_presynaptic(self, pre_neuron, weight, syn_type='a_d'):
        """Ajoute une connexion entrante."""
        self.presynaptic.append((pre_neuron, weight, syn_type))

    def add_postsynaptic(self, post_neuron, weight, syn_type='a_d'):
        """Ajoute une connexion sortante."""
        self.postsynaptic.append((post_neuron, weight, syn_type))

    def reset(self):
        """Réinitialise l'état du neurone."""
        self.V = V_REST
        self.output = 0.0
        self.refractory_timer = 0.0
        self.spike_history = []
        self.is_active = False

    def get_info(self):
        """Retourne les informations du neurone."""
        return {
            'id': self.id,
            'type': self.type,
            'hemisphere': self.hemisphere,
            'compartment': self.compartment,
            'V': round(self.V, 4),
            'output': round(self.output, 4),
            'n_presynaptic': len(self.presynaptic),
            'n_postsynaptic': len(self.postsynaptic),
            'is_active': self.is_active
        }

    def __repr__(self):
        return f"Neuron(id={self.id}, type={self.type}, V={self.V:.3f}, out={self.output:.3f})"
