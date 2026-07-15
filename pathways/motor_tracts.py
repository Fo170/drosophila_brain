"""
pathways/motor_tracts.py
Faisceaux de commandes motrices
Organisation des outputs en tracts fonctionnels
"""

import numpy as np


class MotorTracts:
    """
    Organisation des faisceaux moteurs.

    Architecture:
        Cerveau (CN, MBON)
          ↓
        DNs (Descending Neurons)
          ↓
        Tracts moteurs
          ↓
        Muscles / Glandes

    Commandes:
        - Locomotion: DNVNC → muscles segmentaires
        - Comportement: DNSEZ → alimentation, nettoyage
        - Endocrine: RGN → glande annulaire → hormones
    """

    def __init__(self, network):
        self.network = network

        self.motor_tracts = {
            'locomotion': {
                'neurons': network.dn_neurons['VNC'],
                'target': 'muscles',
                'function': 'mouvement',
                'color': 'cyan'
            },
            'behavior': {
                'neurons': network.dn_neurons['SEZ'],
                'target': 'behaviors',
                'function': 'action',
                'color': 'magenta'
            },
            'endocrine': {
                'neurons': network.rgn_neurons,
                'target': 'ring_gland',
                'function': 'hormones',
                'color': 'gold'
            }
        }

        self._organize_commands()

    def _organize_commands(self):
        """Organise les commandes motrices."""
        # Locomotion: gauche/droite, avant/arrière
        self.locomotion_commands = {
            'forward': [],
            'backward': [],
            'left': [],
            'right': [],
            'stop': []
        }

        dn_vnc = self.motor_tracts['locomotion']['neurons']
        n_per = len(dn_vnc) // len(self.locomotion_commands)

        for i, cmd in enumerate(self.locomotion_commands.keys()):
            start = i * n_per
            end = start + n_per
            self.locomotion_commands[cmd] = dn_vnc[start:end]

        # Comportement: manger, nettoyer, etc.
        self.behavior_commands = {
            'feed': [],
            'groom': [],
            'rest': []
        }

        dn_sez = self.motor_tracts['behavior']['neurons']
        n_per = len(dn_sez) // len(self.behavior_commands)

        for i, cmd in enumerate(self.behavior_commands.keys()):
            start = i * n_per
            end = start + n_per
            self.behavior_commands[cmd] = dn_sez[start:end]

    def get_motor_command(self):
        """
        Calcule la commande motrice globale.

        Returns:
            Dict avec type de commande et intensité
        """
        # Locomotion
        loco = {}
        for cmd, neurons in self.locomotion_commands.items():
            loco[cmd] = np.mean([n.output for n in neurons]) if neurons else 0

        # Comportement
        behav = {}
        for cmd, neurons in self.behavior_commands.items():
            behav[cmd] = np.mean([n.output for n in neurons]) if neurons else 0

        # Endocrine
        endo = np.mean([n.output for n in self.motor_tracts['endocrine']['neurons']])

        return {
            'locomotion': loco,
            'behavior': behav,
            'endocrine': endo
        }

    def get_dominant_command(self):
        """Retourne la commande dominante."""
        commands = self.get_motor_command()

        # Trouver la locomotion dominante
        loco = commands['locomotion']
        max_loco = max(loco, key=loco.get)

        # Trouver le comportement dominant
        behav = commands['behavior']
        max_behav = max(behav, key=behav.get)

        return {
            'locomotion': (max_loco, loco[max_loco]),
            'behavior': (max_behav, behav[max_behav]),
            'endocrine': commands['endocrine']
        }
