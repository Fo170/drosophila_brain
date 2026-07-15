"""
core/network.py
Gestion du réseau complet de 3016 neurones
Propagation dynamique/récurrente avec boucles de feedback
"""

import numpy as np
from collections import defaultdict, deque
from config import (
    N_NEURONS, DT, T_MAX, N_STEPS,
    P_CONNECT_SENSORY_TO_PN, P_CONNECT_PN_TO_MB, P_CONNECT_PN_TO_LH,
    P_CONNECT_KC_TO_MBON, P_CONNECT_MBON_TO_CN, P_CONNECT_CN_TO_OUTPUT,
    P_CONNECT_MBON_TO_MBIN, P_CONNECT_DAN_TO_KC, P_CONNECT_INTERHEMISPHERIC,
    P_CONNECT_DN_TO_VNC, P_CONNECT_DN_FEEDBACK,
    N_ORN_GLOMERULI, N_GRN_ZONES, N_VISUAL_COLUMNS, N_SOMATO_TRACTS,
    N_KC_COMPARTMENTS, N_MBON_TYPES, N_DAN_CLUSTERS
)
from core.neuron import Neuron
from core.synapse import Synapse


class BrainNetwork:
    """
    Réseau cérébral complet avec 3016 neurones.

    Architecture:
        Inputs → PNs → [MB + LH] → CNs → Outputs (DNs)
                  ↓     ↓
                DANs ← MBONs (feedback)

    La propagation est dynamique et récurrente:
        - Feedforward : sensory → interneurons → output
        - Recurrent : MBON → MBIN/DAN → KC, DN → interneurons
        - Interhémisphérique : connexions contralatérales
    """

    def __init__(self, seed=42):
        np.random.seed(seed)

        self.neurons = {}          # id -> Neuron
        self.synapses = []         # Liste de toutes les synapses
        self.synapse_matrix = {}   # (pre, post) -> Synapse

        # Regroupements fonctionnels
        self.sensory_neurons = {'ORN': [], 'GRN': [], 'PR': [], 
                                'thermo': [], 'mechano': [], 'proprio': []}
        self.pn_neurons = []       # Projection neurons
        self.kc_neurons = []       # Kenyon cells
        self.mbon_neurons = []     # MB output neurons
        self.mbin_neurons = []     # MB input/modulatory (DANs)
        self.cn_neurons = []       # Convergence neurons
        self.dn_neurons = {'VNC': [], 'SEZ': []}  # Descending neurons
        self.rgn_neurons = []      # Ring gland neurons
        self.interneurons = []     # Autres interneurons

        # État global
        self.time = 0.0
        self.current_step = 0
        self.history = deque(maxlen=1000)  # Historique d'activité

        # Stimuli actifs
        self.active_stimuli = {}

        self._build_network()

    def _build_network(self):
        """Construit le réseau complet avec regroupement intelligent."""
        print("Construction du réseau cérébral (3016 neurones)...")

        # === 1. CRÉATION DES NEURONES ===
        neuron_id = 0

        # --- INPUTS SENSORIELS (477 neurones) ---
        # ORN regroupés en glomérules virtuels
        for g in range(N_ORN_GLOMERULI):
            n_orn = 176 // N_ORN_GLOMERULI + (1 if g < 176 % N_ORN_GLOMERULI else 0)
            for i in range(n_orn):
                n = Neuron(neuron_id, 'ORN', 'left', f'AL_glom_{g}')
                self.neurons[neuron_id] = n
                self.sensory_neurons['ORN'].append(n)
                neuron_id += 1

        # GRN regroupés en zones
        for z in range(N_GRN_ZONES):
            n_grn = 42 // N_GRN_ZONES + (1 if z < 42 % N_GRN_ZONES else 0)
            for i in range(n_grn):
                n = Neuron(neuron_id, 'GRN', 'left', f'GRN_zone_{z}')
                self.neurons[neuron_id] = n
                self.sensory_neurons['GRN'].append(n)
                neuron_id += 1

        # Photorécepteurs
        for c in range(N_VISUAL_COLUMNS):
            n_pr = 29 // N_VISUAL_COLUMNS + (1 if c < 29 % N_VISUAL_COLUMNS else 0)
            for i in range(n_pr):
                n = Neuron(neuron_id, 'PR', 'left', f'VIS_col_{c}')
                self.neurons[neuron_id] = n
                self.sensory_neurons['PR'].append(n)
                neuron_id += 1

        # Thermo, mécano, proprio
        for i in range(8):
            n = Neuron(neuron_id, 'thermo', 'left', 'thermo')
            self.neurons[neuron_id] = n
            self.sensory_neurons['thermo'].append(n)
            neuron_id += 1
        for i in range(10):
            n = Neuron(neuron_id, 'mechano', 'left', 'mechano')
            self.neurons[neuron_id] = n
            self.sensory_neurons['mechano'].append(n)
            neuron_id += 1
        for i in range(12):
            n = Neuron(neuron_id, 'proprio', 'left', 'proprio')
            self.neurons[neuron_id] = n
            self.sensory_neurons['proprio'].append(n)
            neuron_id += 1

        # Neurones ascendants (ANs) du VNC ~200
        for i in range(200):
            n = Neuron(neuron_id, 'AN', 'left', f'VNC_A1_{i//50}')
            self.neurons[neuron_id] = n
            self.sensory_neurons['mechano'].append(n)  # Regroupés avec mécano
            neuron_id += 1

        # --- PROJECTION NEURONS (PNs) ---
        for i in range(210):
            n = Neuron(neuron_id, 'PN', 'left', f'PN_{i//30}')
            self.neurons[neuron_id] = n
            self.pn_neurons.append(n)
            neuron_id += 1

        # --- MUSHROOM BODY ---
        # KC regroupés en compartiments
        for comp in range(N_KC_COMPARTMENTS):
            n_kc = 176 // N_KC_COMPARTMENTS + (1 if comp < 176 % N_KC_COMPARTMENTS else 0)
            for i in range(n_kc):
                n = Neuron(neuron_id, 'KC', 'left', f'MB_comp_{comp}')
                self.neurons[neuron_id] = n
                self.kc_neurons.append(n)
                neuron_id += 1

        # MBON
        for t in range(N_MBON_TYPES):
            n_mbon = 48 // N_MBON_TYPES + (1 if t < 48 % N_MBON_TYPES else 0)
            for i in range(n_mbon):
                n = Neuron(neuron_id, 'MBON', 'left', f'MBON_type_{t}')
                self.neurons[neuron_id] = n
                self.mbon_neurons.append(n)
                neuron_id += 1

        # MBIN/DAN
        for c in range(N_DAN_CLUSTERS):
            n_dan = 30 // N_DAN_CLUSTERS + (1 if c < 30 % N_DAN_CLUSTERS else 0)
            for i in range(n_dan):
                n = Neuron(neuron_id, 'DAN', 'left', f'DAN_clust_{c}')
                self.neurons[neuron_id] = n
                self.mbin_neurons.append(n)
                neuron_id += 1

        # --- CORNE LATÉRALE (LH) ---
        for i in range(50):
            n = Neuron(neuron_id, 'LHN', 'left', f'LH_{i//10}')
            self.neurons[neuron_id] = n
            self.interneurons.append(n)
            neuron_id += 1

        # --- CONVERGENCE NEURONS (CN) ---
        for i in range(108):
            n = Neuron(neuron_id, 'CN', 'left', f'CN_{i//20}')
            self.neurons[neuron_id] = n
            self.cn_neurons.append(n)
            neuron_id += 1

        # --- OUTPUTS : DNs ---
        for i in range(180):
            n = Neuron(neuron_id, 'DNVNC', 'left', f'DNVNC_{i//30}')
            self.neurons[neuron_id] = n
            self.dn_neurons['VNC'].append(n)
            neuron_id += 1
        for i in range(54):
            n = Neuron(neuron_id, 'DNSEZ', 'left', f'DNSEZ_{i//10}')
            self.neurons[neuron_id] = n
            self.dn_neurons['SEZ'].append(n)
            neuron_id += 1

        # RGN
        for i in range(184):
            n = Neuron(neuron_id, 'RGN', 'left', f'RGN_{i//30}')
            self.neurons[neuron_id] = n
            self.rgn_neurons.append(n)
            neuron_id += 1

        # --- INTERNEURONES RESTANTS ---
        remaining = N_NEURONS - neuron_id
        for i in range(remaining):
            n = Neuron(neuron_id, 'IN', 'left', f'IN_{i//50}')
            self.neurons[neuron_id] = n
            self.interneurons.append(n)
            neuron_id += 1

        # === 2. CRÉATION DES SYNAPSES ===
        self._create_synapses()

        print(f"✓ Réseau construit: {len(self.neurons)} neurones, {len(self.synapses)} synapses")

    def _create_synapses(self):
        """Crée les connexions synaptiques selon l'architecture du connectome."""

        def add_synapse(pre, post, syn_type, prob, n_syn_range=(1, 10)):
            if np.random.random() < prob:
                n_syn = np.random.randint(*n_syn_range)
                syn = Synapse(pre.id, post.id, syn_type, n_syn)
                pre.add_postsynaptic(post, syn.weight, syn_type)
                post.add_presynaptic(pre, syn.weight, syn_type)
                self.synapses.append(syn)
                self.synapse_matrix[(pre.id, post.id)] = syn

        # --- SENSORY → PN ---
        for sn_type, sns in self.sensory_neurons.items():
            for sn in sns:
                for pn in self.pn_neurons:
                    add_synapse(sn, pn, 'a_d', P_CONNECT_SENSORY_TO_PN, (1, 8))

        # --- PN → KC (Mushroom Body) ---
        for pn in self.pn_neurons:
            for kc in self.kc_neurons:
                add_synapse(pn, kc, 'a_d', P_CONNECT_PN_TO_MB, (1, 5))

        # --- PN → LHN (Corne Latérale) ---
        for pn in self.pn_neurons:
            for lhn in [n for n in self.interneurons if n.type == 'LHN']:
                add_synapse(pn, lhn, 'a_d', P_CONNECT_PN_TO_LH, (1, 6))

        # --- KC → MBON (sparsité caractéristique du MB) ---
        for kc in self.kc_neurons:
            for mbon in self.mbon_neurons:
                add_synapse(kc, mbon, 'a_d', P_CONNECT_KC_TO_MBON, (1, 3))

        # --- MBON → CN (intégration valeurs apprises) ---
        for mbon in self.mbon_neurons:
            for cn in self.cn_neurons:
                add_synapse(mbon, cn, 'a_d', P_CONNECT_MBON_TO_CN, (1, 8))

        # --- LHN → CN (intégration valeurs innées) ---
        for lhn in [n for n in self.interneurons if n.type == 'LHN']:
            for cn in self.cn_neurons:
                add_synapse(lhn, cn, 'a_d', P_CONNECT_MBON_TO_CN * 0.8, (1, 6))

        # --- CN → OUTPUTS ---
        for cn in self.cn_neurons:
            for dn in self.dn_neurons['VNC'] + self.dn_neurons['SEZ']:
                add_synapse(cn, dn, 'a_d', P_CONNECT_CN_TO_OUTPUT, (1, 10))
            for rgn in self.rgn_neurons:
                add_synapse(cn, rgn, 'a_d', P_CONNECT_CN_TO_OUTPUT * 0.5, (1, 5))

        # --- BOUCLES RÉCURRENTES ---
        # MBON → MBIN/DAN (feedback)
        for mbon in self.mbon_neurons:
            for dan in self.mbin_neurons:
                add_synapse(mbon, dan, 'a_a', P_CONNECT_MBON_TO_MBIN, (1, 5))

        # DAN → KC (modulation)
        for dan in self.mbin_neurons:
            for kc in self.kc_neurons:
                add_synapse(dan, kc, 'd_a', P_CONNECT_DAN_TO_KC, (1, 4))

        # DN → interneurons (feedback/efference copy)
        for dn in self.dn_neurons['VNC'] + self.dn_neurons['SEZ']:
            for intern in self.interneurons[:100]:
                add_synapse(dn, intern, 'd_a', P_CONNECT_DN_FEEDBACK, (1, 3))

        # --- CONNEXIONS INTERHÉMISPHÉRIQUES ---
        all_neurons = list(self.neurons.values())
        for i, n1 in enumerate(all_neurons):
            for n2 in all_neurons[i+1:]:
                if n1.hemisphere != n2.hemisphere:
                    add_synapse(n1, n2, 'a_d', P_CONNECT_INTERHEMISPHERIC * 0.3, (1, 4))

    def apply_stimulus(self, stimulus_type, intensity=0.8, duration=50.0, target_ids=None):
        """
        Applique un stimulus sensoriel.

        Args:
            stimulus_type: 'olfactory', 'gustatory', 'visual', 'thermal', 
                          'mechano', 'proprioceptive'
            intensity: 0-1
            duration: durée en ms
            target_ids: IDs spécifiques (None = tous du type)
        """
        stimulus_key = f"{stimulus_type}_{self.time}"
        self.active_stimuli[stimulus_key] = {
            'type': stimulus_type,
            'intensity': intensity,
            'duration': duration,
            'start_time': self.time
        }

        # Mapping stimulus → neurones
        type_map = {
            'olfactory': 'ORN',
            'gustatory': 'GRN',
            'visual': 'PR',
            'thermal': 'thermo',
            'mechano': 'mechano',
            'proprioceptive': 'proprio'
        }

        target_type = type_map.get(stimulus_type, 'ORN')
        targets = self.sensory_neurons.get(target_type, [])

        if target_ids:
            targets = [self.neurons[i] for i in target_ids if i in self.neurons]

        # Activation avec bruit biologique
        for neuron in targets:
            noise = np.random.normal(0, 0.1)
            neuron.V = intensity + noise
            neuron.output = neuron.sigmoid(neuron.V)

    def step(self, dt=DT):
        """
        Exécute un pas de simulation.

        Ordre de mise à jour (important pour la dynamique):
            1. Inputs sensoriels (stimuli actifs)
            2. Interneurons (PN, LHN, KC)
            3. MB outputs (MBON)
            4. Modulatory (DAN)
            5. Convergence (CN)
            6. Outputs (DN, RGN)
        """
        # Mise à jour des stimuli actifs
        expired = []
        for key, stim in self.active_stimuli.items():
            if self.time - stim['start_time'] > stim['duration']:
                expired.append(key)
        for key in expired:
            del self.active_stimuli[key]

        # Ordre de mise à jour par type
        update_order = [
            self.sensory_neurons['ORN'] + self.sensory_neurons['GRN'] + 
            self.sensory_neurons['PR'] + self.sensory_neurons['thermo'] + 
            self.sensory_neurons['mechano'] + self.sensory_neurons['proprio'],
            self.pn_neurons,
            [n for n in self.interneurons if n.type == 'LHN'],
            self.kc_neurons,
            self.mbon_neurons,
            self.mbin_neurons,
            self.cn_neurons,
            self.dn_neurons['VNC'] + self.dn_neurons['SEZ'],
            self.rgn_neurons,
            [n for n in self.interneurons if n.type == 'IN']
        ]

        # Mise à jour de tous les neurones
        for group in update_order:
            for neuron in group:
                neuron.update(dt)

        # Mise à jour des traces STDP
        for syn in self.synapses:
            if syn.plastic:
                pre_n = self.neurons[syn.pre_id]
                post_n = self.neurons[syn.post_id]
                syn.update_traces(pre_n.is_active, post_n.is_active, dt)

        # Historique
        self._record_state()

        self.time += dt
        self.current_step += 1

        return self.get_state()

    def _record_state(self):
        """Enregistre l'état actuel pour visualisation."""
        state = {
            'time': self.time,
            'neurons': {nid: n.output for nid, n in self.neurons.items()},
            'active_count': sum(1 for n in self.neurons.values() if n.is_active),
            'mean_activity': np.mean([n.output for n in self.neurons.values()]),
            'mb_activity': np.mean([n.output for n in self.kc_neurons]) if self.kc_neurons else 0,
            'output_activity': np.mean([n.output for n in self.dn_neurons['VNC']]) if self.dn_neurons['VNC'] else 0
        }
        self.history.append(state)

    def get_state(self):
        """Retourne l'état complet du réseau."""
        return {
            'time': self.time,
            'step': self.current_step,
            'n_neurons': len(self.neurons),
            'n_synapses': len(self.synapses),
            'mean_activity': np.mean([n.output for n in self.neurons.values()]),
            'active_neurons': sum(1 for n in self.neurons.values() if n.output > 0.5),
            'stimuli': list(self.active_stimuli.keys()),
            'history_size': len(self.history)
        }

    def get_region_activity(self):
        """Retourne l'activité par région cérébrale."""
        return {
            'sensory': np.mean([n.output for sns in self.sensory_neurons.values() for n in sns]),
            'PN': np.mean([n.output for n in self.pn_neurons]) if self.pn_neurons else 0,
            'KC': np.mean([n.output for n in self.kc_neurons]) if self.kc_neurons else 0,
            'MBON': np.mean([n.output for n in self.mbon_neurons]) if self.mbon_neurons else 0,
            'DAN': np.mean([n.output for n in self.mbin_neurons]) if self.mbin_neurons else 0,
            'CN': np.mean([n.output for n in self.cn_neurons]) if self.cn_neurons else 0,
            'DN_VNC': np.mean([n.output for n in self.dn_neurons['VNC']]) if self.dn_neurons['VNC'] else 0,
            'DN_SEZ': np.mean([n.output for n in self.dn_neurons['SEZ']]) if self.dn_neurons['SEZ'] else 0,
            'RGN': np.mean([n.output for n in self.rgn_neurons]) if self.rgn_neurons else 0
        }

    def reset(self):
        """Réinitialise le réseau."""
        for neuron in self.neurons.values():
            neuron.reset()
        self.time = 0.0
        self.current_step = 0
        self.history.clear()
        self.active_stimuli.clear()

    def run(self, duration_ms, stimulus=None):
        """
        Exécute une simulation complète.

        Args:
            duration_ms: Durée en millisecondes
            stimulus: Dict {'type': ..., 'intensity': ..., 'time': ...} ou None
        """
        n_steps = int(duration_ms / DT)

        if stimulus:
            self.apply_stimulus(
                stimulus['type'], 
                stimulus.get('intensity', 0.8),
                stimulus.get('duration', 50.0)
            )

        for _ in range(n_steps):
            self.step(DT)

        return self.get_state()
