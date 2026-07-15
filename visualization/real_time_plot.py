"""
visualization/real_time_plot.py
Visualisation temps réel de l'activité des 3016 neurones
Panneaux: carte d'activité 2D, graphiques temporels, contrôles
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Ou 'Qt5Agg' selon l'environnement
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from config import VIZ_FPS, VIZ_UPDATE_EVERY, HEATMAP_RESOLUTION


class RealTimeVisualizer:
    """
    Visualiseur temps réel du cerveau de Drosophila.

    Layout:
        ┌─────────────────┬─────────────────┐
        │  CARTE 2D       │  ACTIVITÉ       │
        │  (neurones      │  TEMPORELLE     │
        │   colorés)      │  (courbes)      │
        ├─────────────────┴─────────────────┤
        │  CONTRÔLES + INFO                 │
        └─────────────────────────────────────┘
    """

    def __init__(self, network):
        self.network = network
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.scatters = {}

        # Positions 2D des neurones (layout force-directed simplifié)
        self.positions = self._compute_layout()

        # Historique pour graphiques
        self.time_history = []
        self.activity_history = {
            'sensory': [], 'PN': [], 'KC': [], 'MBON': [], 'DAN': [],
            'CN': [], 'DN_VNC': [], 'DN_SEZ': [], 'RGN': [], 'total': []
        }

        self._setup_plots()

    def _compute_layout(self):
        """
        Calcule les positions 2D des neurones.
        Organisation anatomique: inputs à gauche, outputs à droite.
        """
        positions = np.zeros((len(self.network.neurons), 2))

        # Positions par type (layout simplifié)
        type_positions = {
            'ORN': (-0.8, 0.6), 'GRN': (-0.8, 0.3), 'PR': (-0.8, 0.0),
            'thermo': (-0.8, -0.3), 'mechano': (-0.8, -0.5), 'proprio': (-0.8, -0.7),
            'AN': (-0.6, -0.4),
            'PN': (-0.4, 0.3),
            'KC': (0.0, 0.5), 'MBON': (0.2, 0.5), 'DAN': (0.1, 0.7),
            'LHN': (-0.2, -0.2),
            'CN': (0.4, 0.0),
            'DNVNC': (0.8, 0.3), 'DNSEZ': (0.8, 0.0), 'RGN': (0.8, -0.3),
            'IN': (0.0, -0.5)
        }

        for nid, neuron in self.network.neurons.items():
            base_pos = type_positions.get(neuron.type, (0.0, 0.0))
            # Ajouter du bruit pour disperser
            noise = np.random.normal(0, 0.08, 2)
            positions[nid] = base_pos + noise

        return positions

    def _setup_plots(self):
        """Configure les sous-graphiques."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#1a1a2e')

        # === PANNEAU GAUCHE: Carte d'activité 2D ===
        ax_map = self.fig.add_subplot(2, 2, 1)
        ax_map.set_facecolor('#16213e')
        ax_map.set_title('Carte d\'activité cérébrale (3016 neurones)', 
                        color='white', fontsize=12, fontweight='bold')
        ax_map.set_xlim(-1.2, 1.2)
        ax_map.set_ylim(-1.0, 1.0)
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        # Scatter plot des neurones
        colors = self._get_neuron_colors()
        self.scatters['main'] = ax_map.scatter(
            self.positions[:, 0], self.positions[:, 1],
            c=colors, s=8, alpha=0.6, cmap='hot'
        )

        # Légende des régions
        self._add_region_labels(ax_map)

        # === PANNEAU DROITE HAUT: Activité temporelle ===
        ax_time = self.fig.add_subplot(2, 2, 2)
        ax_time.set_facecolor('#16213e')
        ax_time.set_title('Activité par région', color='white', fontsize=12)
        ax_time.set_xlabel('Temps (ms)', color='white')
        ax_time.set_ylabel('Activité moyenne', color='white')
        ax_time.tick_params(colors='white')
        ax_time.set_ylim(0, 1)

        # Lignes pour chaque région
        region_colors = {
            'sensory': '#ff6b6b', 'PN': '#feca57', 'KC': '#48dbfb',
            'MBON': '#ff9ff3', 'DAN': '#54a0ff', 'CN': '#5f27cd',
            'DN_VNC': '#1dd1a1', 'DN_SEZ': '#00d2d3', 'RGN': '#ff9f43'
        }

        for region, color in region_colors.items():
            line, = ax_time.plot([], [], color=color, label=region, linewidth=1.5)
            self.lines[region] = line

        ax_time.legend(loc='upper right', facecolor='#16213e', 
                      edgecolor='white', labelcolor='white', fontsize=8)
        ax_time.grid(True, alpha=0.3, color='gray')

        # === PANNEAU DROITE BAS: Histogramme + Info ===
        ax_hist = self.fig.add_subplot(2, 2, 4)
        ax_hist.set_facecolor('#16213e')
        ax_hist.set_title('Distribution des poids synaptiques', color='white', fontsize=12)
        ax_hist.set_xlabel('Poids', color='white')
        ax_hist.set_ylabel('Fréquence', color='white')
        ax_hist.tick_params(colors='white')

        self.bars = None

        # === PANNEAU GAUCHE BAS: Info texte ===
        ax_info = self.fig.add_subplot(2, 2, 3)
        ax_info.set_facecolor('#16213e')
        ax_info.axis('off')

        self.info_text = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes,
                                     color='white', fontsize=10, verticalalignment='top',
                                     fontfamily='monospace')

        self.axes = {
            'map': ax_map, 'time': ax_time, 
            'hist': ax_hist, 'info': ax_info
        }

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)

    def _get_neuron_colors(self):
        """Retourne les couleurs basées sur l'activité actuelle."""
        colors = np.zeros(len(self.network.neurons))
        for nid, neuron in self.network.neurons.items():
            colors[nid] = neuron.output
        return colors

    def _add_region_labels(self, ax):
        """Ajoute les labels des régions anatomiques."""
        labels = {
            'INPUTS': (-0.9, 0.85), 'AL/MB': (0.0, 0.85),
            'LH': (-0.2, -0.1), 'OUTPUTS': (0.9, 0.85)
        }
        for text, pos in labels.items():
            ax.text(pos[0], pos[1], text, color='cyan', fontsize=9,
                   ha='center', fontweight='bold', alpha=0.7)

    def update(self, frame=None):
        """Met à jour l'affichage (appelée par animation)."""
        # Mise à jour de la carte d'activité
        colors = self._get_neuron_colors()
        self.scatters['main'].set_array(colors)

        # Mise à jour des graphiques temporels
        if len(self.time_history) > 0:
            for region in self.activity_history.keys():
                if region in self.lines:
                    self.lines[region].set_data(
                        self.time_history, 
                        self.activity_history[region]
                    )

            # Ajuster les limites x
            if len(self.time_history) > 1:
                self.axes['time'].set_xlim(
                    max(0, self.time_history[-1] - 500),
                    self.time_history[-1] + 50
                )

        # Mise à jour histogramme
        weights = [syn.weight for syn in self.network.synapses if syn.plastic]
        if weights:
            self.axes['hist'].clear()
            self.axes['hist'].set_facecolor('#16213e')
            self.axes['hist'].hist(weights, bins=30, color='#48dbfb', alpha=0.7, edgecolor='white')
            self.axes['hist'].set_title('Distribution des poids synaptiques', color='white')
            self.axes['hist'].set_xlabel('Poids', color='white')
            self.axes['hist'].set_ylabel('Fréquence', color='white')
            self.axes['hist'].tick_params(colors='white')

        # Mise à jour info texte
        state = self.network.get_state()
        region_act = self.network.get_region_activity()

        info_str = f"""
╔══════════════════════════════════════╗
║  SIMULATION CERVEAU DROSOPHILA       ║
╠══════════════════════════════════════╣
║ Temps:     {state['time']:>8.1f} ms              ║
║ Pas:       {state['step']:>8d}                  ║
║ Neurones:  {state['n_neurons']:>8d}                  ║
║ Synapses:  {state['n_synapses']:>8d}                  ║
╠══════════════════════════════════════╣
║ ACTIVITÉ PAR RÉGION:                 ║
║  Sensoriel:  {region_act['sensory']:.3f}               ║
║  PN:         {region_act['PN']:.3f}               ║
║  KC (MB):    {region_act['KC']:.3f}               ║
║  MBON:       {region_act['MBON']:.3f}               ║
║  DAN:        {region_act['DAN']:.3f}               ║
║  CN:         {region_act['CN']:.3f}               ║
║  DN_VNC:     {region_act['DN_VNC']:.3f}               ║
║  DN_SEZ:     {region_act['DN_SEZ']:.3f}               ║
╠══════════════════════════════════════╣
║ Neurones actifs: {state['active_neurons']:>4d}              ║
║ Activité moyenne: {state['mean_activity']:.3f}             ║
╚══════════════════════════════════════╝
"""
        self.info_text.set_text(info_str)

        return list(self.scatters.values()) + list(self.lines.values())

    def record_state(self):
        """Enregistre l'état actuel pour les graphiques temporels."""
        self.time_history.append(self.network.time)
        region_act = self.network.get_region_activity()

        for region in self.activity_history.keys():
            if region in region_act:
                self.activity_history[region].append(region_act[region])
            elif region == 'total':
                self.activity_history[region].append(
                    np.mean([n.output for n in self.network.neurons.values()])
                )

        # Limiter l'historique
        max_hist = 2000
        if len(self.time_history) > max_hist:
            self.time_history = self.time_history[-max_hist:]
            for region in self.activity_history:
                self.activity_history[region] = self.activity_history[region][-max_hist:]

    def show(self):
        """Affiche la figure."""
        plt.show()

    def start_animation(self, interval=1000//VIZ_FPS):
        """Démarre l'animation temps réel."""
        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=interval, blit=False,
            cache_frame_data=False
        )
        plt.show()

    def save_frame(self, filename):
        """Sauvegarde une image statique."""
        self.update()
        self.fig.savefig(filename, dpi=150, facecolor='#1a1a2e')
