"""
utils/io.py
Sauvegarde et chargement des états du réseau
"""

import json
import pickle
import numpy as np
import os


class NetworkIO:
    """
    Gestion de la persistance du réseau.

    Formats:
        - JSON: métadonnées, configuration
        - Pickle: état complet du réseau
        - NPZ: matrices de connectivité
    """

    def __init__(self, network):
        self.network = network

    def save_state(self, filepath):
        """Sauvegarde l'état complet du réseau."""
        state = {
            'neurons': {},
            'synapses': [],
            'time': self.network.time,
            'step': self.network.step
        }

        # Sauvegarder les neurones
        for nid, neuron in self.network.neurons.items():
            state['neurons'][nid] = {
                'id': neuron.id,
                'type': neuron.type,
                'hemisphere': neuron.hemisphere,
                'compartment': neuron.compartment,
                'V': neuron.V,
                'output': neuron.output,
                'presynaptic': [(pre.id, w, t) for pre, w, t in neuron.presynaptic],
                'postsynaptic': [(post.id, w, t) for post, w, t in neuron.postsynaptic]
            }

        # Sauvegarder les synapses
        for syn in self.network.synapses:
            state['synapses'].append({
                'pre_id': syn.pre_id,
                'post_id': syn.post_id,
                'type': syn.syn_type,
                'n_synapses': syn.n_synapses,
                'weight': syn.weight,
                'plastic': syn.plastic
            })

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"✓ État sauvegardé: {filepath}")

    def load_state(self, filepath):
        """Charge l'état du réseau."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Restaurer les neurones
        for nid, n_data in state['neurons'].items():
            if nid in self.network.neurons:
                neuron = self.network.neurons[nid]
                neuron.V = n_data['V']
                neuron.output = n_data['output']

        # Restaurer les synapses
        for syn_data in state['synapses']:
            key = (syn_data['pre_id'], syn_data['post_id'])
            if key in self.network.synapse_matrix:
                syn = self.network.synapse_matrix[key]
                syn.weight = syn_data['weight']

        self.network.time = state['time']
        self.network.step = state['step']

        print(f"✓ État chargé: {filepath}")

    def export_connectivity_matrix(self, filepath):
        """Exporte la matrice de connectivité au format NPZ."""
        n = len(self.network.neurons)
        adjacency = np.zeros((n, n), dtype=np.int32)
        weights = np.zeros((n, n), dtype=np.float32)

        for syn in self.network.synapses:
            adjacency[syn.pre_id, syn.post_id] = syn.n_synapses
            weights[syn.pre_id, syn.post_id] = syn.weight

        np.savez(filepath, adjacency=adjacency, weights=weights)
        print(f"✓ Matrice exportée: {filepath}")

    def save_activity_log(self, filepath):
        """Sauvegarde l'historique d'activité."""
        if not self.network.history:
            print("⚠ Pas d'historique à sauvegarder")
            return

        with open(filepath, 'w') as f:
            f.write("time,mean_activity,active_neurons,sensory,PN,KC,MBON,DAN,CN,DN_VNC\n")
            for state in self.network.history:
                f.write(f"{state['time']},{state['mean_activity']},"
                       f"{state['active_count']},{state.get('sensory', 0)},"
                       f"{state.get('PN', 0)},{state.get('KC', 0)},"
                       f"{state.get('MBON', 0)},{state.get('DAN', 0)},"
                       f"{state.get('CN', 0)},{state.get('DN_VNC', 0)}\n")

        print(f"✓ Log d'activité sauvegardé: {filepath}")
