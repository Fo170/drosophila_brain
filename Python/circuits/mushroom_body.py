"""
circuits/mushroom_body.py
Circuit du Mushroom Body (MB) - Centre d'apprentissage et mémoire
KC (Kenyon Cells) → MBON → DAN (modulation)
Architecture: 176 KC → 48 MBON → 30 DAN avec feedback
"""

import numpy as np
from config import N_KC_COMPARTMENTS, N_MBON_TYPES, N_DAN_CLUSTERS
from core.neuron import Neuron


class MushroomBodyCircuit:
    """
    Mushroom Body - Centre d'apprentissage associatif.

    Architecture biologique:
        PN (olfaction/gustation)
          ↓
        KC (Kenyon Cells) - 176 neurones, représentation éparses
          ↓ (sparsité ~7%)
        MBON (Mushroom Body Output Neurons) - 15 types, valence apprise
          ↓
        CN (Convergence Neurons) - intègrent MB + LH
          ↓
        DAN (Dopaminergic Neurons) - signal récompense/punition
          ↓ (feedback)
        KC (modulation de la plasticité)

    Apprentissage:
        - Avant: KC→MBON faible
        - Odeur + Sucre → DAN+ → renforcement KC→MBON
        - Odeur + Choc → DAN- → affaiblissement KC→MBON
    """

    def __init__(self, network):
        self.network = network

        # Compartiments KC
        self.kc_compartments = {}  # comp_id -> liste KC
        self.mbon_by_compartment = {}  # comp_id -> liste MBON
        self.dan_by_cluster = {}  # cluster_id -> liste DAN

        self._organize_compartments()
        self._create_sparse_connectivity()
        self._create_feedback_loops()

    def _organize_compartments(self):
        """Organise les KC en compartiments fonctionnels."""
        kc_neurons = self.network.kc_neurons
        mbon_neurons = self.network.mbon_neurons
        dan_neurons = self.network.mbin_neurons

        # KC en compartiments
        for i, kc in enumerate(kc_neurons):
            comp_id = i % N_KC_COMPARTMENTS
            if comp_id not in self.kc_compartments:
                self.kc_compartments[comp_id] = []
            self.kc_compartments[comp_id].append(kc)
            kc.compartment = f'MB_comp_{comp_id}'

        # MBON par compartiment
        for i, mbon in enumerate(mbon_neurons):
            comp_id = i % N_MBON_TYPES
            if comp_id not in self.mbon_by_compartment:
                self.mbon_by_compartment[comp_id] = []
            self.mbon_by_compartment[comp_id].append(mbon)
            mbon.compartment = f'MBON_type_{comp_id}'

        # DAN en clusters
        for i, dan in enumerate(dan_neurons):
            cluster_id = i % N_DAN_CLUSTERS
            if cluster_id not in self.dan_by_cluster:
                self.dan_by_cluster[cluster_id] = []
            self.dan_by_cluster[cluster_id].append(dan)
            dan.compartment = f'DAN_clust_{cluster_id}'

        print(f"  MB: {N_KC_COMPARTMENTS} compartiments KC, "
              f"{N_MBON_TYPES} types MBON, {N_DAN_CLUSTERS} clusters DAN")

    def _create_sparse_connectivity(self):
        """
        Crée la connectivité épars caractéristique du MB.
        Chaque KC se connecte à peu de MBON (~7% de connexions).
        """
        for comp_id, kcs in self.kc_compartments.items():
            for kc in kcs:
                # Chaque KC se connecte à ~15% des MBON (sparsité)
                for mbon_type, mbons in self.mbon_by_compartment.items():
                    if np.random.random() < 0.15:  # Sparsité
                        for mbon in mbons:
                            weight = np.random.uniform(0.05, 0.2)
                            kc.add_postsynaptic(mbon, weight, 'a_d')
                            mbon.add_presynaptic(kc, weight, 'a_d')

                            # Créer la synapse dans le réseau
                            syn_key = (kc.id, mbon.id)
                            if syn_key not in self.network.synapse_matrix:
                                from core.synapse import Synapse
                                syn = Synapse(kc.id, mbon.id, 'a_d', 
                                            n_synapses=np.random.randint(1, 3),
                                            plastic=True)
                                syn.weight = weight
                                self.network.synapses.append(syn)
                                self.network.synapse_matrix[syn_key] = syn

    def _create_feedback_loops(self):
        """Crée les boucles de feedback MBON → DAN → KC."""
        for mbon_type, mbons in self.mbon_by_compartment.items():
            for mbon in mbons:
                # MBON → DAN (feedback positif ou négatif)
                for cluster_id, dans in self.dan_by_cluster.items():
                    for dan in dans:
                        # Connexion a-a (modulation axonique)
                        weight = np.random.uniform(0.1, 0.4)
                        mbon.add_postsynaptic(dan, weight, 'a_a')
                        dan.add_presynaptic(mbon, weight, 'a_a')

                # MBON → CN (convergence avec LH)
                for cn in self.network.cn_neurons:
                    if np.random.random() < 0.3:
                        weight = np.random.uniform(0.2, 0.6)
                        mbon.add_postsynaptic(cn, weight, 'a_d')
                        cn.add_presynaptic(mbon, weight, 'a_d')

    def activate_kc_pattern(self, pattern, intensity=0.8):
        """
        Active un motif spécifique de KC (représentation d'odeur).

        Args:
            pattern: Liste binaire ou valeurs [0-1] de taille N_KC_COMPARTMENTS
            intensity: Intensité globale
        """
        if len(pattern) != N_KC_COMPARTMENTS:
            pattern = [0.5] * N_KC_COMPARTMENTS

        for comp_id, activation in enumerate(pattern):
            comp_intensity = intensity * activation

            # Activer ~15% des KC du compartiment (sparsité)
            kcs = self.kc_compartments.get(comp_id, [])
            n_active = max(1, int(len(kcs) * 0.15))
            active_kcs = np.random.choice(kcs, size=min(n_active, len(kcs)), replace=False)

            for kc in active_kcs:
                noise = np.random.normal(0, 0.1)
                kc.V = comp_intensity + noise
                kc.output = kc.sigmoid(kc.V)

    def get_kc_sparsity(self):
        """Calcule la sparsité de l'activité KC (doit être ~7-15%)."""
        active = sum(1 for kc in self.network.kc_neurons if kc.output > 0.3)
        return active / len(self.network.kc_neurons) if self.network.kc_neurons else 0

    def get_mbon_valence(self):
        """
        Calcule la valence apprise (approche vs fuite).
        Retourne une valeur positive (approche) ou négative (fuite).
        """
        # MBONs positifs (approche) vs négatifs (fuite)
        # Simplification: types 0-7 = approche, 8-14 = fuite
        approach = np.mean([mbon.output for mbon in self.network.mbon_neurons[:24]])
        avoidance = np.mean([mbon.output for mbon in self.network.mbon_neurons[24:]])
        return approach - avoidance

    def get_compartment_activity(self):
        """Retourne l'activité par compartiment KC."""
        activity = {}
        for comp_id, kcs in self.kc_compartments.items():
            act = np.mean([kc.output for kc in kcs]) if kcs else 0
            activity[f'KC_comp_{comp_id}'] = act
        return activity

    def get_dan_activity(self):
        """Retourne l'activité par cluster DAN."""
        activity = {}
        for cluster_id, dans in self.dan_by_cluster.items():
            act = np.mean([dan.output for dan in dans]) if dans else 0
            activity[f'DAN_clust_{cluster_id}'] = act
        return activity
