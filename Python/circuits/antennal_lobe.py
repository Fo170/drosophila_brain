"""
circuits/antennal_lobe.py
Circuit du Lobe Antennaire (AL) - Traitement olfactif
ORN → Glomérules → PN (Projection Neurons)
Architecture: 176 ORN → 15 glomérules virtuels → 210 PN
"""

import numpy as np
from config import N_ORN_GLOMERULI, P_CONNECT_SENSORY_TO_PN
from core.neuron import Neuron
from core.synapse import Synapse


class AntennalLobeCircuit:
    """
    Lobe Antennaire - Premier relais du traitement olfactif.

    Architecture biologique:
        ORN (olfactory receptor neurons)
          ↓
        Glomérules (15 zones de convergence)
          ↓
        PN (projection neurons) → MB + LH
        LN (local neurons) → inhibition latérale

    Dans le simulateur:
        - 176 ORN regroupés en 15 glomérules
        - Chaque glomérule projette sur ~14 PN
        - Lateral inhibition entre glomérules
    """

    def __init__(self, network):
        self.network = network
        self.glomeruli = {}  # glom_id -> liste d'ORN
        self.pn_by_glom = {}  # glom_id -> liste de PN
        self.local_neurons = []  # LN pour inhibition

        self._organize_glomeruli()
        self._create_lateral_inhibition()

    def _organize_glomeruli(self):
        """Organise les ORN en glomérules virtuels."""
        orn_neurons = self.network.sensory_neurons['ORN']

        # Distribuer les ORN en glomérules
        for i, orn in enumerate(orn_neurons):
            glom_id = i % N_ORN_GLOMERULI
            if glom_id not in self.glomeruli:
                self.glomeruli[glom_id] = []
            self.glomeruli[glom_id].append(orn)
            orn.compartment = f'AL_glom_{glom_id}'

        # Associer les PN aux glomérules
        pn_neurons = self.network.pn_neurons
        pn_per_glom = len(pn_neurons) // N_ORN_GLOMERULI

        for g in range(N_ORN_GLOMERULI):
            start_idx = g * pn_per_glom
            end_idx = start_idx + pn_per_glom + (1 if g < len(pn_neurons) % N_ORN_GLOMERULI else 0)
            self.pn_by_glom[g] = pn_neurons[start_idx:min(end_idx, len(pn_neurons))]

            for pn in self.pn_by_glom[g]:
                pn.compartment = f'AL_glom_{g}'

        print(f"  AL: {len(self.glomeruli)} glomérules, "
              f"{len(orn_neurons)} ORN, {len(pn_neurons)} PN")

    def _create_lateral_inhibition(self):
        """Crée l'inhibition latérale entre glomérules (LN)."""
        # Créer des neurones locaux pour chaque glomérule
        for g in range(N_ORN_GLOMERULI):
            ln = Neuron(
                max(self.network.neurons.keys()) + 1 + g,
                'LN', 'left', f'AL_glom_{g}'
            )
            self.network.neurons[ln.id] = ln
            self.local_neurons.append(ln)

            # LN reçoit des ORN du même glomérule
            for orn in self.glomeruli.get(g, []):
                ln.add_presynaptic(orn, 0.3, 'a_d')
                orn.add_postsynaptic(ln, 0.3, 'a_d')

            # LN inhibe les PN des autres glomérules
            for other_g in range(N_ORN_GLOMERULI):
                if other_g != g:
                    for pn in self.pn_by_glom.get(other_g, []):
                        ln.add_postsynaptic(pn, -0.4, 'a_d')  # Inhibition
                        pn.add_presynaptic(ln, -0.4, 'a_d')

    def present_odor(self, odor_pattern, intensity=0.8):
        """
        Présente un motif d'odeur au lobe antennaire.

        Args:
            odor_pattern: Liste de 15 valeurs [0-1] (activation par glomérule)
            intensity: Intensité globale
        """
        if len(odor_pattern) != N_ORN_GLOMERULI:
            odor_pattern = [0.5] * N_ORN_GLOMERULI

        for g, activation in enumerate(odor_pattern):
            glom_intensity = intensity * activation

            # Activer les ORN du glomérule
            for orn in self.glomeruli.get(g, []):
                noise = np.random.normal(0, 0.05)
                orn.V = glom_intensity + noise
                orn.output = orn.sigmoid(orn.V)

    def get_glomerular_activity(self):
        """Retourne l'activité moyenne par glomérule."""
        activity = {}
        for g, orns in self.glomeruli.items():
            act = np.mean([orn.output for orn in orns]) if orns else 0
            activity[f'glom_{g}'] = act
        return activity

    def get_pn_activity(self):
        """Retourne l'activité des PN par glomérule."""
        activity = {}
        for g, pns in self.pn_by_glom.items():
            act = np.mean([pn.output for pn in pns]) if pns else 0
            activity[f'PN_glom_{g}'] = act
        return activity
