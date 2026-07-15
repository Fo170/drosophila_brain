"""
learning/reinforcement_learning.py
Apprentissage par renforcement modulé par DANs (Dopaminergic Neurons)
Règle de Hebb à trois facteurs: pré × post × DAN
"""

import numpy as np
from config import LEARNING_RATE, DAN_REWARD_SCALE, DAN_PUNISH_SCALE, STDP_WINDOW, LTD_FACTOR


class DANModulatedLearning:
    """
    Apprentissage inspiré du Mushroom Body de Drosophila.

    Architecture d'apprentissage:
        Stimulus (odeur) → KC → MBON → Action (approche/fuite)
                              ↓
                        Récompense/Punition → DANs → Modulation KC→MBON

    Règle:
        Δw_KC→MBON = η × signal_DAN × (activité_KC × activité_MBON)

        signal_DAN > 0 (récompense) → potentiation (renforcement)
        signal_DAN < 0 (punition) → dépression (affaiblissement)
    """

    def __init__(self, network, learning_rate=LEARNING_RATE):
        self.network = network
        self.lr = learning_rate
        self.dan_history = []  # Historique des signaux DAN

        # Seuils de récompense/punition
        self.reward_threshold = 0.6
        self.punish_threshold = 0.3

    def compute_dan_signal(self, reward=0.0, punishment=0.0):
        """
        Calcule le signal dopaminergique global.

        Args:
            reward: Valeur de récompense (0-1), ex: sucre trouvé
            punishment: Valeur de punition (0-1), ex: chaleur, choc

        Returns:
            Signal DAN net (-1 à +1)
        """
        dan_signal = 0.0

        if reward > self.reward_threshold:
            # Récompense positive → DANs activés fortement
            dan_signal = DAN_REWARD_SCALE * reward
        elif punishment > self.punish_threshold:
            # Punition → DANs activés négativement (ou sous-ensemble différent)
            dan_signal = DAN_PUNISH_SCALE * punishment
        else:
            # Pas de signal → décroissance lente (oubli)
            dan_signal = -0.001

        self.dan_history.append(dan_signal)
        if len(self.dan_history) > 1000:
            self.dan_history.pop(0)

        return dan_signal

    def update_kc_mbon_weights(self, dan_signal):
        """
        Met à jour les poids KC→MBON selon la règle de trois facteurs.

        Δw = η × DAN(t) × trace_KC × trace_MBON

        CORRIGÉ: Met aussi à jour les poids dans les listes
        presynaptic/postsynaptic des neurones.
        """
        kcs = self.network.kc_neurons
        mbons = self.network.mbon_neurons

        n_updates = 0
        total_delta = 0.0

        for kc in kcs:
            for syn_data in kc.postsynaptic:
                post_neuron, _, syn_type = syn_data

                if post_neuron.type == 'MBON':
                    # Récupérer la synapse
                    syn_key = (kc.id, post_neuron.id)
                    if syn_key in self.network.synapse_matrix:
                        syn = self.network.synapse_matrix[syn_key]

                        if syn.plastic:
                            # Traces d'activité (fenêtre temporelle)
                            pre_trace = syn.pre_trace
                            post_trace = syn.post_trace

                            # Règle de trois facteurs
                            correlation = pre_trace * post_trace
                            delta_w = self.lr * dan_signal * correlation

                            # Appliquer le changement
                            old_weight = syn.weight
                            syn.weight += delta_w
                            syn.weight = np.clip(syn.weight, 0.0, 2.0)

                            # CORRECTION: Mettre à jour les poids dans les neurones aussi
                            self._update_neuron_weights(kc, post_neuron, syn.weight)

                            n_updates += 1
                            total_delta += abs(syn.weight - old_weight)

        return n_updates, total_delta

    def _update_neuron_weights(self, pre_neuron, post_neuron, new_weight):
        """
        Met à jour les poids dans les listes presynaptic/postsynaptic.

        CORRECTION CRITIQUE: Les poids dans les tuples du neurone
        doivent être synchronisés avec syn.weight.
        """
        # Mettre à jour dans pre_neuron.postsynaptic
        for i, (post, weight, syn_type) in enumerate(pre_neuron.postsynaptic):
            if post.id == post_neuron.id:
                pre_neuron.postsynaptic[i] = (post, new_weight, syn_type)
                break

        # Mettre à jour dans post_neuron.presynaptic
        for i, (pre, weight, syn_type) in enumerate(post_neuron.presynaptic):
            if pre.id == pre_neuron.id:
                post_neuron.presynaptic[i] = (pre, new_weight, syn_type)
                break

    def apply_reward(self, reward_value=1.0):
        """
        Applique une récompense et met à jour les poids.

        Args:
            reward_value: Intensité de la récompense (0-1)

        Returns:
            Dict avec statistiques de mise à jour
        """
        dan_signal = self.compute_dan_signal(reward=reward_value)
        n_updates, total_delta = self.update_kc_mbon_weights(dan_signal)

        # Activation des DANs pour visualisation
        for dan in self.network.mbin_neurons:
            dan.V = reward_value * 0.8
            dan.output = dan.sigmoid(dan.V)

        return {
            'dan_signal': dan_signal,
            'n_updates': n_updates,
            'total_delta': total_delta,
            'mean_delta': total_delta / n_updates if n_updates > 0 else 0,
            'type': 'reward'
        }

    def apply_punishment(self, punishment_value=1.0):
        """
        Applique une punition et met à jour les poids.

        Args:
            punishment_value: Intensité de la punition (0-1)

        Returns:
            Dict avec statistiques de mise à jour
        """
        dan_signal = self.compute_dan_signal(punishment=punishment_value)
        n_updates, total_delta = self.update_kc_mbon_weights(dan_signal)

        # Activation des DANs (négative)
        for dan in self.network.mbin_neurons:
            dan.V = -punishment_value * 0.5
            dan.output = dan.sigmoid(dan.V)

        return {
            'dan_signal': dan_signal,
            'n_updates': n_updates,
            'total_delta': total_delta,
            'mean_delta': total_delta / n_updates if n_updates > 0 else 0,
            'type': 'punishment'
        }

    def get_association_strength(self, kc_compartment, mbon_type):
        """
        Retourne la force de l'association entre un compartiment KC et un MBON.
        Utile pour suivre l'apprentissage au fil du temps.
        """
        total_weight = 0.0
        n_synapses = 0

        for kc in self.network.kc_neurons:
            if kc.compartment == kc_compartment:
                for post, weight, _ in kc.postsynaptic:
                    if post.type == 'MBON' and post.compartment == mbon_type:
                        total_weight += weight
                        n_synapses += 1

        return total_weight / n_synapses if n_synapses > 0 else 0.0

    def get_learning_summary(self):
        """Résumé de l'état d'apprentissage."""
        return {
            'n_dan_signals': len(self.dan_history),
            'mean_dan': np.mean(self.dan_history) if self.dan_history else 0,
            'max_dan': max(self.dan_history) if self.dan_history else 0,
            'min_dan': min(self.dan_history) if self.dan_history else 0
        }
