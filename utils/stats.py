"""
utils/stats.py
Statistiques du connectome et du réseau
Analyse de la topologie, des hubs, de la récurrence
"""

import numpy as np
from collections import Counter


class ConnectomeStats:
    """
    Analyse statistique du connectome.

    Métriques:
        - Distribution des degrés (in/out)
        - Identification des hubs
        - Coefficient de clustering
        - Longueur des chemins
        - Modularité
    """

    def __init__(self, network):
        self.network = network

    def degree_distribution(self):
        """Distribution des degrés entrants et sortants."""
        in_degrees = [len(n.presynaptic) for n in self.network.neurons.values()]
        out_degrees = [len(n.postsynaptic) for n in self.network.neurons.values()]

        return {
            'in_mean': np.mean(in_degrees),
            'in_max': max(in_degrees),
            'out_mean': np.mean(out_degrees),
            'out_max': max(out_degrees),
            'in_distribution': Counter(in_degrees),
            'out_distribution': Counter(out_degrees)
        }

    def identify_hubs(self, threshold_percentile=95):
        """Identifie les hubs du réseau."""
        in_degrees = {nid: len(n.presynaptic) for nid, n in self.network.neurons.items()}
        out_degrees = {nid: len(n.postsynaptic) for nid, n in self.network.neurons.items()}

        in_threshold = np.percentile(list(in_degrees.values()), threshold_percentile)
        out_threshold = np.percentile(list(out_degrees.values()), threshold_percentile)

        in_hubs = [nid for nid, deg in in_degrees.items() if deg >= in_threshold]
        out_hubs = [nid for nid, deg in out_degrees.items() if deg >= out_threshold]
        in_out_hubs = list(set(in_hubs) & set(out_hubs))

        return {
            'in_hubs': in_hubs,
            'out_hubs': out_hubs,
            'in_out_hubs': in_out_hubs,
            'n_in_hubs': len(in_hubs),
            'n_out_hubs': len(out_hubs),
            'n_in_out_hubs': len(in_out_hubs)
        }

    def get_hub_neurons(self):
        """Retourne les neurones hubs avec leurs types."""
        hubs = self.identify_hubs()
        hub_info = []

        for nid in hubs['in_out_hubs']:
            n = self.network.neurons[nid]
            hub_info.append({
                'id': nid,
                'type': n.type,
                'in_degree': len(n.presynaptic),
                'out_degree': len(n.postsynaptic),
                'compartment': n.compartment
            })

        return hub_info

    def clustering_coefficient(self):
        """Coefficient de clustering moyen."""
        coeffs = []
        for n in self.network.neurons.values():
            neighbors = set()
            for pre, _, _ in n.presynaptic:
                neighbors.add(pre.id)
            for post, _, _ in n.postsynaptic:
                neighbors.add(post.id)

            if len(neighbors) < 2:
                continue

            # Compter les connexions entre voisins
            edges = 0
            neighbors = list(neighbors)
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if (n1, n2) in self.network.synapse_matrix or                        (n2, n1) in self.network.synapse_matrix:
                        edges += 1

            possible = len(neighbors) * (len(neighbors) - 1) / 2
            if possible > 0:
                coeffs.append(edges / possible)

        return np.mean(coeffs) if coeffs else 0

    def network_summary(self):
        """Résumé complet des statistiques."""
        deg = self.degree_distribution()
        hubs = self.identify_hubs()
        cluster = self.clustering_coefficient()

        return {
            'n_neurons': len(self.network.neurons),
            'n_synapses': len(self.network.synapses),
            'mean_in_degree': deg['in_mean'],
            'mean_out_degree': deg['out_mean'],
            'max_in_degree': deg['in_max'],
            'max_out_degree': deg['out_max'],
            'n_in_out_hubs': hubs['n_in_out_hubs'],
            'clustering_coeff': cluster,
            'density': len(self.network.synapses) / (len(self.network.neurons) ** 2)
        }
