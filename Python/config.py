"""
config.py
Paramètres globaux du simulateur du cerveau de larve de Drosophila melanogaster
Basé sur Winding et al. 2023 (Science) - Connectome complet 3016 neurones, 548000 synapses
"""

import numpy as np

# =============================================================================
# PARAMÈTRES DU CONNECTOME
# =============================================================================
N_NEURONS = 3016           # Nombre total de neurones dans le connectome
N_SYNAPSES = 548000        # Nombre total de synapses
N_HEMISPHERES = 2         # Gauche / Droite
N_CELL_TYPES = 93         # Types cellulaires par clustering spectral

# =============================================================================
# PARAMÈTRES DU NEURONE (Sigmoïde + Fuite Temporelle)
# =============================================================================
TAU_MEMBRANE = 10.0       # Constante de temps de fuite (ms) - temps caractéristique
V_REST = 0.0              # Potentiel de repos (normalisé 0-1)
V_THRESHOLD = 0.5         # Seuil d'activation (pour affichage, pas de spike dur)
TAU_REFRACTORY = 2.0      # Période réfractaire (ms)

# Paramètres de la sigmoïde
SIGMOID_STEEPNESS = 10.0  # Pente de la sigmoïde
SIGMOID_OFFSET = 0.0      # Décalage horizontal

# =============================================================================
# PARAMÈTRES SYNAPTIQUES
# =============================================================================
SYNAPSE_TYPES = {
    'a_d': {'name': 'axo-dendritic', 'fraction': 0.666, 'weight_scale': 1.0},
    'a_a': {'name': 'axo-axonique', 'fraction': 0.258, 'weight_scale': 0.7},
    'd_d': {'name': 'dendro-dendritique', 'fraction': 0.058, 'weight_scale': 0.5},
    'd_a': {'name': 'dendro-axonique', 'fraction': 0.018, 'weight_scale': 0.4}
}

# Seuils de force synaptique (basés sur l'article)
WEAK_SYNAPSE_THRESHOLD = 2      # 1-2 synapses = faible
STRONG_SYNAPSE_THRESHOLD = 5      # ≥5 synapses = forte
MAX_SYNAPSE_STRENGTH = 50       # Force maximale observée

# Échelle de poids pour la simulation
WEIGHT_SCALE = 0.1              # Facteur global de mise à l'échelle

# =============================================================================
# PARAMÈTRES DE DYNAMIQUE TEMPORELLE
# =============================================================================
DT = 0.1                        # Pas de temps (ms)
T_MAX = 1000.0                  # Durée max d'une simulation (ms)
N_STEPS = int(T_MAX / DT)

# =============================================================================
# PARAMÈTRES D'APPRENTISSAGE (Hebb modulé par DANs)
# =============================================================================
LEARNING_RATE = 0.01            # Taux d'apprentissage η
DAN_REWARD_SCALE = 1.0          # Échelle DAN pour récompense
DAN_PUNISH_SCALE = -0.5         # Échelle DAN pour punition
STDP_WINDOW = 20.0              # Fenêtre temporelle STDP (ms)
LTD_FACTOR = 0.5                # Facteur de dépression à long terme

# =============================================================================
# PARAMÈTRES DE CONNECTIVITÉ RÉALISTES (basés sur Winding et al.)
# =============================================================================
# Densité du réseau
NETWORK_DENSITY = 0.01          # ~1% de connexions possibles (larve = dense)

# Probabilités de connexion par type
P_CONNECT_SENSORY_TO_PN = 0.3   # ORN/GRN → PN
P_CONNECT_PN_TO_MB = 0.25       # PN → KC (lobe antennaire → mushroom body)
P_CONNECT_PN_TO_LH = 0.2        # PN → LH (corne latérale)
P_CONNECT_KC_TO_MBON = 0.15     # KC → MBON (sparsité du MB)
P_CONNECT_MBON_TO_CN = 0.4      # MBON → CN (convergence)
P_CONNECT_CN_TO_OUTPUT = 0.3    # CN → DNs (sortie)
P_CONNECT_MBON_TO_MBIN = 0.2    # MBON → MBIN/DAN (feedback)
P_CONNECT_DAN_TO_KC = 0.1       # DAN → KC (modulation)
P_CONNECT_INTERHEMISPHERIC = 0.15  # Contralatéral
P_CONNECT_DN_TO_VNC = 0.35      # DN → VNC
P_CONNECT_DN_FEEDBACK = 0.1     # DN → interneurons (feedback)

# =============================================================================
# REGROUPEMENT INTELLIGENT (simplification)
# =============================================================================
# Capteurs regroupés en faisceaux fonctionnels
N_ORN_GLOMERULI = 15            # 176 ORN → 15 glomérules virtuels
N_GRN_ZONES = 8                 # 42 GRN → 8 zones gustatives
N_VISUAL_COLUMNS = 5            # 29 PR → 5 colonnes visuelles
N_SOMATO_TRACTS = 3             # Thermo + mécano + proprio → 3 faisceaux

# Mushroom Body groupé
N_KC_COMPARTMENTS = 7           # 176 KC → 7 compartiments
N_MBON_TYPES = 15               # 48 MBON → 15 types
N_DAN_CLUSTERS = 10             # 30 MBIN/DAN → 10 clusters

# =============================================================================
# PARAMÈTRES DE VISUALISATION
# =============================================================================
VIZ_FPS = 30                    # Images par seconde pour temps réel
VIZ_UPDATE_EVERY = 10           # Mettre à jour l'affichage tous les N pas
HEATMAP_RESOLUTION = 50         # Résolution carte de chaleur

# =============================================================================
# PARAMÈTRES DU MONDE VIRTUEL (Phase 2)
# =============================================================================
WORLD_SIZE = (100, 100)         # Taille du monde 2D (unités arbitraires)
N_ODOR_SOURCES = 5              # Sources d'odeur
N_FOOD_SOURCES = 3              # Sources de nourriture
N_THREAT_ZONES = 2              # Zones de danger (chaleur)

# Stimuli par défaut
DEFAULT_STIMULUS_DURATION = 50.0  # ms
DEFAULT_STIMULUS_INTENSITY = 0.8  # 0-1

# =============================================================================
# SEED POUR REPRODUCTIBILITÉ
# =============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
