"""
pathways/feedback_loops.py
Boucles de feedback dans le cerveau
Récurrence, efference copy, prédictions
"""

import numpy as np


class FeedbackLoops:
    """
    Boucles de feedback du connectome.

    Types de feedback:
        1. MBON → DAN → KC (apprentissage)
        2. DN → interneurons (efference copy)
        3. Output → Input (ré-entrée)
        4. Interhémisphérique (gauche ↔ droite)

    Statistiques:
        - 41% des neurones reçoivent des entrées récurrentes
        - DANs parmi les plus récurrents
    """

    def __init__(self, network):
        self.network = network

        self.loops = {
            'mb_recurrent': [],      # MBON → DAN → KC
            'dn_feedback': [],       # DN → interneurons
            'output_input': [],      # Output → sensory
            'interhemispheric': []   # Gauche ↔ Droite
        }

        self._identify_loops()

    def _identify_loops(self):
        """Identifie les boucles de feedback dans le réseau."""
        # MBON → DAN → KC
        for mbon in self.network.mbon_neurons:
            for post_mbon, w1, _ in mbon.postsynaptic:
                if post_mbon.type == 'DAN':
                    for post_dan, w2, _ in post_mbon.postsynaptic:
                        if post_dan.type == 'KC':
                            self.loops['mb_recurrent'].append(
                                (mbon, post_mbon, post_dan)
                            )

        # DN → interneurons
        for dn in self.network.dn_neurons['VNC'] + self.network.dn_neurons['SEZ']:
            for post, w, _ in dn.postsynaptic:
                if post.type in ['IN', 'PN', 'LHN']:
                    self.loops['dn_feedback'].append((dn, post))

        print(f"  Boucles identifiées:")
        print(f"    MB récurrent: {len(self.loops['mb_recurrent'])}")
        print(f"    DN feedback: {len(self.loops['dn_feedback'])}")

    def get_recurrent_activity(self):
        """Mesure l'activité dans les boucles récurrentes."""
        mb_loop_act = 0
        if self.loops['mb_recurrent']:
            mb_loop_act = np.mean([
                mbon.output + dan.output + kc.output
                for mbon, dan, kc in self.loops['mb_recurrent']
            ]) / 3

        dn_feedback_act = 0
        if self.loops['dn_feedback']:
            dn_feedback_act = np.mean([
                dn.output + intern.output
                for dn, intern in self.loops['dn_feedback']
            ]) / 2

        return {
            'mb_recurrent': mb_loop_act,
            'dn_feedback': dn_feedback_act
        }

    def get_recurrence_fraction(self):
        """Calcule la fraction de neurones en boucles récurrentes."""
        recurrent_neurons = set()

        for loop_type, loops in self.loops.items():
            for loop in loops:
                for neuron in loop:
                    recurrent_neurons.add(neuron.id)

        return len(recurrent_neurons) / len(self.network.neurons)
