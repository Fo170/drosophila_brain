"""
circuits/brain_vnc_interface.py
Interface Cerveau - Moelle Épinière (VNC)
Descending Neurons (DNs) et Ascending Neurons (ANs)
"""

import numpy as np


class BrainVNCInterface:
    """
    Interface entre le cerveau et la moelle épinière ventrale (VNC).

    Architecture:
        Cerveau
          ↓ DNsVNC (Descending Neurons to VNC) - 180 neurones
        VNC
          ↓ ANs (Ascending Neurons) - 200 neurones
        Cerveau

    Fonction:
        - DNsVNC: Commandes motrices (locomotion, action)
        - DNsSEZ: Commandes comportementales (alimentation)
        - ANs: Feedback sensoriel du corps (proprioception, toucher)
        - Efference copy: DNs → interneurons (prédiction)
    """

    def __init__(self, network):
        self.network = network
        self.dn_vnc = network.dn_neurons['VNC']
        self.dn_sez = network.dn_neurons['SEZ']
        self.ans = network.sensory_neurons.get('mechano', [])  # ANs regroupés

        self._organize_motor_commands()

    def _organize_motor_commands(self):
        """Organise les DNs par type de commande motrice."""
        self.motor_commands = {
            'forward': [],      # Avancer
            'backward': [],     # Reculer
            'turn_left': [],    # Tourner gauche
            'turn_right': [],   # Tourner droite
            'stop': []          # Arrêt
        }

        # Distribuer les DNVNC
        dn_list = self.dn_vnc
        n_per_cmd = len(dn_list) // len(self.motor_commands)

        for i, cmd in enumerate(self.motor_commands.keys()):
            start = i * n_per_cmd
            end = start + n_per_cmd + (1 if i < len(dn_list) % len(self.motor_commands) else 0)
            self.motor_commands[cmd] = dn_list[start:min(end, len(dn_list))]

            for dn in self.motor_commands[cmd]:
                dn.compartment = f'DN_{cmd}'

    def get_motor_output(self):
        """
        Calcule la commande motrice à partir de l'activité des DNs.

        Returns:
            Dict avec speed, turn_angle
        """
        cmd_activity = {}
        for cmd, dns in self.motor_commands.items():
            cmd_activity[cmd] = np.mean([dn.output for dn in dns]) if dns else 0

        # Calculer la commande résultante
        forward = cmd_activity.get('forward', 0)
        backward = cmd_activity.get('backward', 0)
        left = cmd_activity.get('turn_left', 0)
        right = cmd_activity.get('turn_right', 0)

        speed = (forward - backward) * 2.0  # Vitesse linéaire
        turn = (right - left) * np.pi / 2   # Angle de rotation

        return {
            'speed': np.clip(speed, -2.0, 2.0),
            'turn_angle': np.clip(turn, -np.pi, np.pi),
            'command_activities': cmd_activity
        }

    def get_ascending_input(self):
        """Retourne l'activité des neurones ascendants (feedback du corps)."""
        return np.mean([an.output for an in self.ans]) if self.ans else 0

    def get_efference_copy_strength(self):
        """Mesure la force de l'efference copy (DN → interneurons)."""
        feedback_strength = 0
        for dn in self.dn_vnc:
            for post, weight, _ in dn.postsynaptic:
                if post.type in ['IN', 'PN', 'LHN']:
                    feedback_strength += abs(weight) * dn.output
        return feedback_strength
