"""
circuits/interhemispheric.py
Communication interhémisphérique
Neurones contralatéraux et bilatéraux
"""

import numpy as np


class InterhemisphericCircuit:
    """
    Communication entre les deux hémisphères cérébraux.

    Statistiques du connectome:
        - 37% des neurones ont des branches contralatérales
        - 88% des hubs in-out ont des axones contra- ou bilatéraux
        - Connexions réciproques entre homologues gauche/droite

    Fonction:
        - Intégration sensorielle bilatérale
        - Coordination motrice symétrique
        - Transfert d'information entre hémisphères
    """

    def __init__(self, network):
        self.network = network
        self.contralateral_pairs = []  # Paires d'homologues
        self.commissural_neurons = []  # Neurones traversant la ligne médiane

        self._identify_pairs()
        self._create_commissural_connections()

    def _identify_pairs(self):
        """Identifie les paires homologues gauche/droite."""
        # Regrouper par type et compartiment
        by_type_comp = {}
        for nid, neuron in self.network.neurons.items():
            key = (neuron.type, neuron.compartment)
            if key not in by_type_comp:
                by_type_comp[key] = {'left': [], 'right': []}
            by_type_comp[key][neuron.hemisphere].append(neuron)

        # Créer les paires
        for key, hemis in by_type_comp.items():
            for left_n, right_n in zip(hemis['left'], hemis['right']):
                self.contralateral_pairs.append((left_n, right_n))

        print(f"  Interhémisphère: {len(self.contralateral_pairs)} paires homologues")

    def _create_commissural_connections(self):
        """Crée les connexions traversant la ligne médiane."""
        # 15% des neurones ont des projections contralatérales
        all_neurons = list(self.network.neurons.values())
        n_commissural = int(len(all_neurons) * 0.15)

        for neuron in np.random.choice(all_neurons, n_commissural, replace=False):
            self.commissural_neurons.append(neuron)
            neuron.compartment = f"{neuron.compartment}_commissural"

    def get_interhemispheric_activity(self):
        """Compare l'activité des deux hémisphères."""
        left_act = np.mean([n.output for n in self.network.neurons.values() 
                           if n.hemisphere == 'left'])
        right_act = np.mean([n.output for n in self.network.neurons.values() 
                            if n.hemisphere == 'right'])
        return {
            'left': left_act,
            'right': right_act,
            'asymmetry': abs(left_act - right_act)
        }
