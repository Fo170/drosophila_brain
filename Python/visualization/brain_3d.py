"""
visualization/brain_3d.py
Visualisation 3D du cerveau de Drosophila
Vue anatomique avec régions colorées
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Brain3DVisualizer:
    """
    Visualisation 3D du cerveau de larve de Drosophila.

    Layout anatomique:
        - X: gauche (-) / droite (+)
        - Y: antérieur (-) / postérieur (+)
        - Z: ventral (-) / dorsal (+)

    Régions:
        - AL (Lobe Antennaire): vert
        - MB (Mushroom Body): magenta
        - LH (Corne Latérale): cyan
        - VNC interface: jaune
        - Outputs: rouge
    """

    def __init__(self, network):
        self.network = network
        self.fig = None
        self.ax = None

        # Positions 3D par type
        self.positions_3d = self._compute_3d_layout()

        # Couleurs par type
        self.type_colors = {
            'ORN': '#2ecc71',      # Vert
            'GRN': '#f1c40f',      # Jaune
            'PR': '#3498db',       # Bleu
            'thermo': '#e74c3c',   # Rouge
            'mechano': '#9b59b6',  # Violet
            'proprio': '#e67e22',  # Orange
            'AN': '#95a5a6',       # Gris
            'PN': '#1abc9c',       # Turquoise
            'KC': '#ff6b6b',       # Rose
            'MBON': '#f368e0',     # Magenta
            'DAN': '#ff9f43',      # Orange vif
            'LHN': '#00d2d3',      # Cyan
            'CN': '#5f27cd',       # Violet foncé
            'DNVNC': '#ff4757',    # Rouge vif
            'DNSEZ': '#ff6348',    # Rouge orangé
            'RGN': '#ffa502',      # Orange doré
            'IN': '#a4b0be'        # Gris clair
        }

    def _compute_3d_layout(self):
        """Calcule les positions 3D des neurones."""
        positions = np.zeros((len(self.network.neurons), 3))

        # Positions de base par type (layout anatomique simplifié)
        type_base_3d = {
            'ORN': (-0.6, -0.3, 0.2),
            'GRN': (-0.6, -0.3, -0.1),
            'PR': (-0.6, 0.2, 0.3),
            'thermo': (-0.6, 0.0, -0.2),
            'mechano': (-0.6, 0.2, -0.3),
            'proprio': (-0.6, 0.3, -0.2),
            'AN': (-0.4, 0.5, -0.3),
            'PN': (-0.2, -0.2, 0.1),
            'KC': (0.0, 0.0, 0.3),
            'MBON': (0.2, 0.0, 0.3),
            'DAN': (0.1, 0.2, 0.4),
            'LHN': (-0.1, -0.3, 0.0),
            'CN': (0.3, -0.1, 0.1),
            'DNVNC': (0.6, 0.2, 0.0),
            'DNSEZ': (0.6, -0.1, 0.1),
            'RGN': (0.6, 0.0, -0.2),
            'IN': (0.0, 0.3, -0.1)
        }

        for nid, neuron in self.network.neurons.items():
            base = type_base_3d.get(neuron.type, (0.0, 0.0, 0.0))
            # Ajouter du bruit pour disperser
            noise = np.random.normal(0, 0.05, 3)
            positions[nid] = base + noise

        return positions

    def setup_plot(self):
        """Configure le graphique 3D."""
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.patch.set_facecolor('#1a1a2e')

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#16213e')

        self.ax.set_xlabel('Gauche ← → Droite', color='white', fontsize=10)
        self.ax.set_ylabel('Antérieur ← → Postérieur', color='white', fontsize=10)
        self.ax.set_zlabel('Ventral ← → Dorsal', color='white', fontsize=10)

        self.ax.set_xlim(-0.8, 0.8)
        self.ax.set_ylim(-0.5, 0.6)
        self.ax.set_zlim(-0.4, 0.5)

        # Couleur des axes
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('white')
        self.ax.yaxis.pane.set_edgecolor('white')
        self.ax.zaxis.pane.set_edgecolor('white')
        self.ax.xaxis.pane.set_alpha(0.1)
        self.ax.yaxis.pane.set_alpha(0.1)
        self.ax.zaxis.pane.set_alpha(0.1)

        self.ax.tick_params(colors='white')

        self.ax.set_title('Cerveau de Drosophila melanogaster (Larve) - Vue 3D\n'
                         '3016 neurones, 548000 synapses',
                         color='white', fontsize=14, fontweight='bold', pad=20)

    def plot_neurons(self, show_active_only=False, alpha=0.6):
        """Affiche les neurones en 3D."""
        if self.ax is None:
            self.setup_plot()

        # Regrouper par type pour la légende
        plotted_types = set()

        for nid, neuron in self.network.neurons.items():
            pos = self.positions_3d[nid]
            color = self.type_colors.get(neuron.type, '#ffffff')

            # Taille basée sur le degré (hubs plus gros)
            size = 20 + 5 * (len(neuron.presynaptic) + len(neuron.postsynaptic))

            # Alpha basé sur l'activité
            if show_active_only and neuron.output < 0.3:
                continue

            alpha_val = max(0.3, neuron.output) if not show_active_only else 0.8

            label = neuron.type if neuron.type not in plotted_types else None
            if label:
                plotted_types.add(neuron.type)

            self.ax.scatter(pos[0], pos[1], pos[2],
                          c=color, s=size, alpha=alpha_val,
                          label=label, edgecolors='none')

        # Légende
        self.ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),
                      facecolor='#1a1a2e', edgecolor='white',
                      labelcolor='white', fontsize=8)

    def plot_connections(self, max_connections=5000, alpha=0.1):
        """Affiche les connexions synaptiques (sous-échantillonnées)."""
        if self.ax is None:
            return

        # Sous-échantillonner pour la performance
        synapses_sample = np.random.choice(
            self.network.synapses,
            min(max_connections, len(self.network.synapses)),
            replace=False
        )

        for syn in synapses_sample:
            pre_pos = self.positions_3d[syn.pre_id]
            post_pos = self.positions_3d[syn.post_id]

            # Couleur selon le type de synapse
            color_map = {'a_d': '#00ff00', 'a_a': '#ffff00', 
                        'd_d': '#ff00ff', 'd_a': '#00ffff'}
            color = color_map.get(syn.syn_type, '#ffffff')

            # Épaisseur selon le poids
            lw = 0.5 + syn.weight * 2

            self.ax.plot([pre_pos[0], post_pos[0]],
                        [pre_pos[1], post_pos[1]],
                        [pre_pos[2], post_pos[2]],
                        color=color, alpha=alpha, linewidth=lw)

    def plot_region_boundaries(self):
        """Dessine les frontières des régions anatomiques."""
        if self.ax is None:
            return

        # AL (Lobe Antennaire)
        self._draw_region_box((-0.7, -0.4, -0.1), (0.1, 0.3, 0.4), 
                             'AL', '#2ecc71', alpha=0.1)

        # MB (Mushroom Body)
        self._draw_region_box((-0.1, -0.1, 0.2), (0.3, 0.2, 0.3),
                             'MB', '#ff6b6b', alpha=0.1)

        # LH (Lateral Horn)
        self._draw_region_box((-0.2, -0.4, -0.1), (0.1, 0.2, 0.2),
                             'LH', '#00d2d3', alpha=0.1)

        # Outputs
        self._draw_region_box((0.5, -0.2, -0.3), (0.2, 0.5, 0.4),
                             'Outputs', '#ff4757', alpha=0.1)

    def _draw_region_box(self, origin, size, label, color, alpha=0.1):
        """Dessine une boîte représentant une région."""
        x, y, z = origin
        dx, dy, dz = size

        # Points du cube
        points = [
            [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
            [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
        ]

        # Arêtes
        edges = [
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]
        ]

        for edge in edges:
            p1, p2 = points[edge[0]], points[edge[1]]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=color, alpha=alpha*2, linewidth=1)

        # Label
        self.ax.text(x+dx/2, y+dy/2, z+dz/2, label, color=color, fontsize=9)

    def update_activity(self):
        """Met à jour les couleurs selon l'activité."""
        if self.ax is None:
            return

        # Recalculer les couleurs basées sur l'activité
        # (Nécessite de redessiner)
        self.ax.clear()
        self.setup_plot()
        self.plot_neurons(show_active_only=False)
        self.plot_region_boundaries()

    def show(self):
        """Affiche la visualisation."""
        if self.ax is None:
            self.setup_plot()
            self.plot_neurons()
            self.plot_region_boundaries()
        plt.show()

    def save(self, filename):
        """Sauvegarde l'image."""
        self.fig.savefig(filename, dpi=150, facecolor='#1a1a2e')
        print(f"✓ Image 3D sauvegardée: {filename}")
