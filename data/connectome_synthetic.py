"""
data/connectome_synthetic.py
Matrice de connectivité synthétique mais réaliste basée sur Winding et al. 2023
Génère une matrice 3016×3016 avec statistiques du connectome réel
"""

import numpy as np
from config import (
    N_NEURONS, N_SYNAPSES, SYNAPSE_TYPES,
    WEAK_SYNAPSE_THRESHOLD, STRONG_SYNAPSE_THRESHOLD, MAX_SYNAPSE_STRENGTH,
    P_CONNECT_SENSORY_TO_PN, P_CONNECT_PN_TO_MB, P_CONNECT_PN_TO_LH,
    P_CONNECT_KC_TO_MBON, P_CONNECT_MBON_TO_CN, P_CONNECT_CN_TO_OUTPUT,
    P_CONNECT_MBON_TO_MBIN, P_CONNECT_DAN_TO_KC, P_CONNECT_INTERHEMISPHERIC,
    P_CONNECT_DN_TO_VNC, P_CONNECT_DN_FEEDBACK
)


class SyntheticConnectome:
    """
    Génère une matrice de connectivité 3016×3016 réaliste.

    Basée sur les statistiques publiées:
    - 548000 synapses au total
    - 66.6% a-d, 25.8% a-a, 5.8% d-d, 1.8% d-a
    - Distribution de force: 60% faibles (1-2), 17.9% fortes (≥5)
    - 41% des neurones en boucles récurrentes
    - Densité ~1%
    """

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.n_neurons = N_NEURONS
        self.n_synapses_target = N_SYNAPSES

        # Matrices de connectivité
        self.adjacency = np.zeros((N_NEURONS, N_NEURONS), dtype=np.int32)
        self.weights = np.zeros((N_NEURONS, N_NEURONS), dtype=np.float32)
        self.syn_types = np.full((N_NEURONS, N_NEURONS), '', dtype=object)

        # Métadonnées par neurone
        self.neuron_types = np.full(N_NEURONS, 'IN', dtype=object)
        self.neuron_regions = np.full(N_NEURONS, 'unknown', dtype=object)
        self.neuron_hemispheres = np.full(N_NEURONS, 'left', dtype=object)

        self._generate()

    def _generate(self):
        """Génère la matrice complète."""
        print("Génération de la matrice de connectivité synthétique...")

        # 1. Assigner les types de neurones
        self._assign_neuron_types()

        # 2. Générer les connexions par région
        self._generate_sensory_to_pn()
        self._generate_pn_to_mb()
        self._generate_pn_to_lh()
        self._generate_kc_to_mbon()
        self._generate_mbon_to_cn()
        self._generate_lhn_to_cn()
        self._generate_cn_to_output()
        self._generate_recurrent_loops()
        self._generate_interhemispheric()
        self._generate_feedback()
        self._generate_background_connectivity()

        # 3. Normaliser pour atteindre ~548000 synapses
        self._normalize_synapses()

        print(f"✓ Matrice générée: {np.count_nonzero(self.adjacency)} connexions, "
              f"{self.adjacency.sum()} synapses")

    def _assign_neuron_types(self):
        """Assigne les types de neurones selon les effectifs du connectome."""
        idx = 0

        # Inputs sensoriels (477)
        n_orn = 176
        n_grn = 42
        n_pr = 29
        n_thermo = 8
        n_mechano = 10
        n_proprio = 12
        n_an = 200  # Ascending neurons

        for i in range(n_orn):
            self.neuron_types[idx] = 'ORN'
            self.neuron_regions[idx] = 'AL'
            idx += 1
        for i in range(n_grn):
            self.neuron_types[idx] = 'GRN'
            self.neuron_regions[idx] = 'gustatory'
            idx += 1
        for i in range(n_pr):
            self.neuron_types[idx] = 'PR'
            self.neuron_regions[idx] = 'visual'
            idx += 1
        for i in range(n_thermo):
            self.neuron_types[idx] = 'thermo'
            self.neuron_regions[idx] = 'somato'
            idx += 1
        for i in range(n_mechano):
            self.neuron_types[idx] = 'mechano'
            self.neuron_regions[idx] = 'somato'
            idx += 1
        for i in range(n_proprio):
            self.neuron_types[idx] = 'proprio'
            self.neuron_regions[idx] = 'somato'
            idx += 1
        for i in range(n_an):
            self.neuron_types[idx] = 'AN'
            self.neuron_regions[idx] = 'VNC'
            idx += 1

        # Projection neurons (210)
        for i in range(210):
            self.neuron_types[idx] = 'PN'
            self.neuron_regions[idx] = 'AL'
            idx += 1

        # Mushroom Body
        for i in range(176):  # KC
            self.neuron_types[idx] = 'KC'
            self.neuron_regions[idx] = 'MB'
            idx += 1
        for i in range(48):  # MBON
            self.neuron_types[idx] = 'MBON'
            self.neuron_regions[idx] = 'MB'
            idx += 1
        for i in range(30):  # DAN/MBIN
            self.neuron_types[idx] = 'DAN'
            self.neuron_regions[idx] = 'MB'
            idx += 1

        # Corne latérale
        for i in range(50):
            self.neuron_types[idx] = 'LHN'
            self.neuron_regions[idx] = 'LH'
            idx += 1

        # Convergence neurons
        for i in range(108):
            self.neuron_types[idx] = 'CN'
            self.neuron_regions[idx] = 'convergence'
            idx += 1

        # Outputs
        for i in range(180):
            self.neuron_types[idx] = 'DNVNC'
            self.neuron_regions[idx] = 'output'
            idx += 1
        for i in range(54):
            self.neuron_types[idx] = 'DNSEZ'
            self.neuron_regions[idx] = 'output'
            idx += 1
        for i in range(184):
            self.neuron_types[idx] = 'RGN'
            self.neuron_regions[idx] = 'output'
            idx += 1

        # Interneurones restants
        while idx < N_NEURONS:
            self.neuron_types[idx] = 'IN'
            self.neuron_regions[idx] = 'interneuron'
            idx += 1

        # Hémisphères (alternance approximative)
        for i in range(N_NEURONS):
            self.neuron_hemispheres[i] = 'left' if i % 2 == 0 else 'right'

    def _add_connections(self, pre_indices, post_indices, prob, syn_type='a_d',
                        n_syn_range=(1, 10), weight_scale=1.0):
        """Ajoute des connexions entre deux populations."""
        for pre in pre_indices:
            for post in post_indices:
                if np.random.random() < prob and pre != post:
                    n_syn = np.random.randint(*n_syn_range)

                    # Si connexion existe déjà, cumuler
                    if self.adjacency[pre, post] > 0:
                        self.adjacency[pre, post] += n_syn
                    else:
                        self.adjacency[pre, post] = n_syn
                        self.syn_types[pre, post] = syn_type

                    # Poids basé sur nombre de synapses et type
                    w = np.log1p(n_syn) / np.log1p(MAX_SYNAPSE_STRENGTH)
                    w *= SYNAPSE_TYPES[syn_type]['weight_scale'] * weight_scale
                    w *= np.random.uniform(0.8, 1.2)
                    self.weights[pre, post] = w

    def _get_indices(self, types):
        """Retourne les indices des neurones d'un ou plusieurs types."""
        if isinstance(types, str):
            types = [types]
        return [i for i, t in enumerate(self.neuron_types) if t in types]

    def _generate_sensory_to_pn(self):
        """ORN/GRN/PR → PN (lobe antennaire et autres)"""
        sensory = self._get_indices(['ORN', 'GRN', 'PR', 'thermo', 'mechano', 'proprio', 'AN'])
        pns = self._get_indices('PN')
        self._add_connections(sensory, pns, P_CONNECT_SENSORY_TO_PN, 'a_d', (1, 8))

    def _generate_pn_to_mb(self):
        """PN → KC (Mushroom Body)"""
        pns = self._get_indices('PN')
        kcs = self._get_indices('KC')
        self._add_connections(pns, kcs, P_CONNECT_PN_TO_MB, 'a_d', (1, 5))

    def _generate_pn_to_lh(self):
        """PN → LHN (Corne Latérale)"""
        pns = self._get_indices('PN')
        lhns = self._get_indices('LHN')
        self._add_connections(pns, lhns, P_CONNECT_PN_TO_LH, 'a_d', (1, 6))

    def _generate_kc_to_mbon(self):
        """KC → MBON (sparsité caractéristique)"""
        kcs = self._get_indices('KC')
        mbons = self._get_indices('MBON')
        self._add_connections(kcs, mbons, P_CONNECT_KC_TO_MBON, 'a_d', (1, 3))

    def _generate_mbon_to_cn(self):
        """MBON → CN (valeurs apprises)"""
        mbons = self._get_indices('MBON')
        cns = self._get_indices('CN')
        self._add_connections(mbons, cns, P_CONNECT_MBON_TO_CN, 'a_d', (1, 8))

    def _generate_lhn_to_cn(self):
        """LHN → CN (valeurs innées)"""
        lhns = self._get_indices('LHN')
        cns = self._get_indices('CN')
        self._add_connections(lhns, cns, P_CONNECT_MBON_TO_CN * 0.8, 'a_d', (1, 6))

    def _generate_cn_to_output(self):
        """CN → DNs + RGN"""
        cns = self._get_indices('CN')
        outputs = self._get_indices(['DNVNC', 'DNSEZ', 'RGN'])
        self._add_connections(cns, outputs, P_CONNECT_CN_TO_OUTPUT, 'a_d', (1, 10))

    def _generate_recurrent_loops(self):
        """Boucles récurrentes: MBON → DAN, DAN → KC"""
        mbons = self._get_indices('MBON')
        dans = self._get_indices('DAN')
        kcs = self._get_indices('KC')

        # MBON → DAN (feedback)
        self._add_connections(mbons, dans, P_CONNECT_MBON_TO_MBIN, 'a_a', (1, 5))
        # DAN → KC (modulation)
        self._add_connections(dans, kcs, P_CONNECT_DAN_TO_KC, 'd_a', (1, 4))

    def _generate_interhemispheric(self):
        """Connexions contralatérales"""
        left = [i for i, h in enumerate(self.neuron_hemispheres) if h == 'left']
        right = [i for i, h in enumerate(self.neuron_hemispheres) if h == 'right']

        # Connexions croisées
        self._add_connections(left, right, P_CONNECT_INTERHEMISPHERIC * 0.3, 'a_d', (1, 4))
        self._add_connections(right, left, P_CONNECT_INTERHEMISPHERIC * 0.3, 'a_d', (1, 4))

    def _generate_feedback(self):
        """DN → interneurons (efference copy)"""
        dns = self._get_indices(['DNVNC', 'DNSEZ'])
        interns = self._get_indices(['IN', 'LHN', 'PN'])
        self._add_connections(dns, interns[:min(len(interns), 300)], 
                             P_CONNECT_DN_FEEDBACK, 'd_a', (1, 3))

    def _generate_background_connectivity(self):
        """Connexions aléatoires de fond pour atteindre la densité réaliste"""
        n_connections = np.count_nonzero(self.adjacency)
        target_connections = int(N_NEURONS * N_NEURONS * 0.01)  # ~1% densité

        remaining = target_connections - n_connections
        if remaining > 0:
            # Ajouter des connexions aléatoires entre interneurones
            interns = self._get_indices(['IN', 'PN', 'LHN'])
            for _ in range(remaining):
                pre = np.random.choice(interns)
                post = np.random.choice(interns)
                if pre != post and self.adjacency[pre, post] == 0:
                    n_syn = np.random.randint(1, 4)
                    self.adjacency[pre, post] = n_syn
                    self.syn_types[pre, post] = np.random.choice(['a_d', 'a_a', 'd_d', 'd_a'],
                                                                  p=[0.666, 0.258, 0.058, 0.018])
                    w = np.log1p(n_syn) / np.log1p(MAX_SYNAPSE_STRENGTH)
                    w *= SYNAPSE_TYPES[self.syn_types[pre, post]]['weight_scale'] * 0.05
                    self.weights[pre, post] = w

    def _normalize_synapses(self):
        """Normalise pour approcher 548000 synapses."""
        current = self.adjacency.sum()
        if current > 0:
            scale = N_SYNAPSES / current
            self.adjacency = (self.adjacency * scale).astype(np.int32)
            self.adjacency = np.clip(self.adjacency, 0, MAX_SYNAPSE_STRENGTH)

    def get_stats(self):
        """Statistiques de la matrice générée."""
        n_conn = np.count_nonzero(self.adjacency)
        n_syn = self.adjacency.sum()

        type_counts = {}
        for i in range(N_NEURONS):
            for j in range(N_NEURONS):
                if self.adjacency[i, j] > 0:
                    t = self.syn_types[i, j]
                    type_counts[t] = type_counts.get(t, 0) + 1

        strong = np.sum(self.adjacency >= STRONG_SYNAPSE_THRESHOLD)
        weak = np.sum((self.adjacency > 0) & (self.adjacency <= WEAK_SYNAPSE_THRESHOLD))

        return {
            'n_neurons': N_NEURONS,
            'n_connections': n_conn,
            'n_synapses': n_syn,
            'density': n_conn / (N_NEURONS * N_NEURONS),
            'synapse_types': type_counts,
            'strong_synapses': int(strong),
            'weak_synapses': int(weak),
            'mean_synapses_per_connection': n_syn / n_conn if n_conn > 0 else 0
        }

    def export_to_network(self, network):
        """Exporte la matrice vers un objet BrainNetwork."""
        for i in range(N_NEURONS):
            for j in range(N_NEURONS):
                if self.adjacency[i, j] > 0 and i in network.neurons and j in network.neurons:
                    pre = network.neurons[i]
                    post = network.neurons[j]
                    syn_type = self.syn_types[i, j] if self.syn_types[i, j] else 'a_d'
                    weight = self.weights[i, j]

                    pre.add_postsynaptic(post, weight, syn_type)
                    post.add_presynaptic(pre, weight, syn_type)

                    syn = Synapse(i, j, syn_type, self.adjacency[i, j])
                    syn.weight = weight
                    network.synapses.append(syn)
                    network.synapse_matrix[(i, j)] = syn


# Import ici pour éviter circular import
from core.synapse import Synapse
