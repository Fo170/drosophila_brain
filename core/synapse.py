"""
core/synapse.py
Gestion des synapses avec types a-d, a-a, d-d, d-a
Poids basés sur les statistiques du connectome réel
"""

import numpy as np
from config import (
    WEAK_SYNAPSE_THRESHOLD, STRONG_SYNAPSE_THRESHOLD, MAX_SYNAPSE_STRENGTH,
    WEIGHT_SCALE, SYNAPSE_TYPES
)


class Synapse:
    """
    Synapse biologiquement réaliste avec plasticité.

    Types:
        a_d (66.6%) : axo-dendritique - transmission feedforward standard
        a_a (25.8%) : axo-axonique - modulation présynaptique
        d_d (5.8%)  : dendro-dendritique - interaction locale
        d_a (1.8%)  : dendro-axonique - feedback vers l'axon
    """

    def __init__(self, pre_id, post_id, syn_type='a_d', n_synapses=1, 
                 plastic=True, learning_rate=0.01):
        """
        Args:
            pre_id: ID neurone présynaptique
            post_id: ID neurone postsynaptique
            syn_type: Type de connexion ('a_d', 'a_a', 'd_d', 'd_a')
            n_synapses: Nombre de synapses physiques (1-50)
            plastic: Si True, la synapse est modifiable par l'apprentissage
            learning_rate: Taux de plasticité
        """
        self.pre_id = pre_id
        self.post_id = post_id
        self.syn_type = syn_type
        self.n_synapses = n_synapses
        self.plastic = plastic
        self.learning_rate = learning_rate

        # Calcul du poids initial basé sur le nombre de synapses physiques
        self.weight = self._compute_initial_weight(n_synapses, syn_type)
        self.initial_weight = self.weight  # Pour traçage

        # Historique pour STDP
        self.pre_trace = 0.0   # Trace d'activité pré
        self.post_trace = 0.0  # Trace d'activité post
        self.trace_tau = 20.0  # Constante de temps STDP (ms)

        # Métadonnées
        self.is_strong = n_synapses >= STRONG_SYNAPSE_THRESHOLD
        self.is_weak = n_synapses <= WEAK_SYNAPSE_THRESHOLD

    def _compute_initial_weight(self, n_synapses, syn_type):
        """
        Calcule le poids initial à partir du nombre de synapses physiques.
        Basé sur les statistiques de Winding et al. 2023.
        """
        # Normalisation logarithmique du nombre de synapses
        normalized = np.log1p(n_synapses) / np.log1p(MAX_SYNAPSE_STRENGTH)

        # Échelle par type de synapse
        type_scale = SYNAPSE_TYPES[syn_type]['weight_scale']

        # Poids final
        weight = normalized * type_scale * WEIGHT_SCALE

        # Variabilité biologique (±20%)
        weight *= np.random.uniform(0.8, 1.2)

        return weight

    def update_traces(self, pre_active, post_active, dt):
        """
        Met à jour les traces d'activité pour STDP.
        Les traces décroissent exponentiellement (fuite).
        """
        # Fuite exponentielle
        self.pre_trace *= np.exp(-dt / self.trace_tau)
        self.post_trace *= np.exp(-dt / self.trace_tau)

        # Incrément si activité
        if pre_active:
            self.pre_trace += 1.0
        if post_active:
            self.post_trace += 1.0

    def stdp_update(self, dan_signal=0.0, dt=0.1):
        """
        Mise à jour du poids par STDP modulé par DANs.

        Règle de trois facteurs:
            Δw = η × DAN(t) × (pre_trace × post_trace)

        Args:
            dan_signal: Signal dopaminergique (positif = récompense, négatif = punition)
            dt: Pas de temps
        """
        if not self.plastic:
            return 0.0

        # Corrélation pré-post (Hebb)
        correlation = self.pre_trace * self.post_trace

        # Modulation par DAN (troisième facteur)
        # DAN > 0 → potentiation, DAN < 0 → dépression
        if dan_signal > 0:
            # Récompense : renforcer les synapses corrélées
            delta_w = self.learning_rate * dan_signal * correlation
        elif dan_signal < 0:
            # Punition : affaiblir les synapses corrélées
            delta_w = self.learning_rate * dan_signal * correlation * 0.5
        else:
            # Pas de signal DAN : décroissance lente (oubli)
            delta_w = -0.001 * (self.weight - self.initial_weight) * dt

        # Application avec limites
        self.weight += delta_w
        self.weight = np.clip(self.weight, 0.0, 5.0 * WEIGHT_SCALE)

        return delta_w

    def get_strength_category(self):
        """Retourne la catégorie de force de la synapse."""
        if self.n_synapses >= STRONG_SYNAPSE_THRESHOLD:
            return 'strong'
        elif self.n_synapses <= WEAK_SYNAPSE_THRESHOLD:
            return 'weak'
        return 'medium'

    def get_info(self):
        """Informations sur la synapse."""
        return {
            'pre_id': self.pre_id,
            'post_id': self.post_id,
            'type': self.syn_type,
            'n_synapses': self.n_synapses,
            'weight': round(self.weight, 6),
            'plastic': self.plastic,
            'strength': self.get_strength_category(),
            'is_strong': self.is_strong
        }

    def __repr__(self):
        return (f"Synapse({self.pre_id}→{self.post_id}, "
                f"type={self.syn_type}, w={self.weight:.4f}, "
                f"n_syn={self.n_synapses})")
