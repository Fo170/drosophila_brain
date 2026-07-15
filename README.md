# Simulateur de Cerveau de *Drosophila melanogaster* (Larve)

[![Tests](https://img.shields.io/badge/tests-7%2F7%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![C++](https://img.shields.io/badge/C%2B%2B-Qt6-green)]()
[![License](https://img.shields.io/badge/license-GPL--3.0-orange)]()

Simulateur biologiquement réaliste du connectome complet du cerveau de larve de *Drosophila melanogaster*, basé sur l'étude de **Winding et al. (2023)** publiée dans *Science*.

Deux implémentations : **Python** (prototypage, tests) et **C++ Qt6** (performance, rendu OpenGL temps réel).

## Caractéristiques

- **3016 neurones** avec dynamique temporelle réaliste (fuite + sigmoïde)
- **~548 000 synapses** avec 4 types biologiques (a-d, a-a, d-d, d-a) et distribution réaliste
- **Apprentissage DAN-modulé** : STDP + règle 3 facteurs (Δw = η × DAN × pré × post)
- **Propagation récurrente** avec boucles de feedback (MBON↔DAN↔KC, efference copy, interhémisphérique)
- **Monde virtuel 2D/3D** avec odeurs (attractif/aversif/neutre), nourriture, dangers, obstacles
- **Visualisation temps réel** 2D (C++ Qt6 QOpenGLWidget) et anatomique 3D (Python)

## Architecture du Cerveau Simulé

```
┌─────────────────────────────────────────────────────────────┐
│              CERVEAU DE LARVE (3 016 neurones)              │
│                   548 000 synapses | 2 hémisphères          │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  INPUTS       │    │  INTERNEURONES│    │  OUTPUTS      │
│  (477 SN)     │───▶│  (2 118)      │───▶│  (418)        │
│  ORN/GRN/PR   │    │  PN/KC/MBON   │    │  DNVNC/DNSEZ  │
│  thermo/méca │    │  DAN/CN/LHN   │    │  RGN          │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
   Neurones              Réseau de traitement   Commandes
   sensoriels            et intégration         motrices/endocrines
```

## Implémentation C++ Qt6

Le répertoire `QT/` contient une réécriture complète en **C++20 / Qt6 / QOpenGLWidget / Eigen3** pour des performances maximales.

### Architecture C++

```
QT/
├── CMakeLists.txt
├── build/
│   └── drosophila_brain          # Binaire compilé (~815 Ko)
└── src/
    ├── main.cpp                  # Point d'entrée
    ├── config.h                  # Paramètres globaux (3016 N, 548K S, etc.)
    ├── core/
    │   ├── neuron.h/.cpp         # Neurone : V(t) fuite + σ(V) sigmoïde
    │   ├── synapse.h/.cpp        # Synapse : STDP, traces pré/post, 4 types
    │   └── network.h/.cpp        # Réseau 3016 N, connectome structuré
    ├── world/
    │   └── world_3d.h/.cpp       # Monde 2D avec entités + boucle d'interaction
    ├── render/
    │   └── glwidget.h/.cpp       # Rendu OpenGL : shaders points/lignes + border
    └── ui/
        ├── mainwindow.h/.cpp     # Fenêtre 1400×900 fixe, 4 panneaux latéraux
        └── insect_ctrl.h/.cpp    # Contrôles manuels (stimulus, reward, punish)
```

### Différences clés Python → C++

| Aspect | Python | C++ Qt6 |
|--------|--------|---------|
| Langage | Python 3.12 | C++20 |
| Rendu | Matplotlib / Mayavi | QOpenGLWidget (shaders GLSL) |
| Performance | ~17 pas/s | ~200+ pas/s (estimation) |
| Connectome | All-to-all aléatoire | Structuré : 7 sous-types d'IN |
| DNVNC | 180 indifférenciés | 4 groupes fonctionnels (FWD/LTL/LTR/BWD) |
| `reset()` | Poids réinitialisés | Reconstruction aléatoire complète |
| Caméra | Suivi automatique | Vue fixe (pan glisser-souris) |
| Fenêtre | Redimensionnable | Fixe 1400×900 |
| Panneaux UI | 4 panneaux temps réel | 4 panneaux (infos, contrôles, stimuli, légende) |
| Distribution synaptique | Simplifiée | Proche du réel (a-d 66%, a-a 24.5%, d-d 6.4%, d-a 3.1%) |

### Connectome Structuré (C++)

Le réseau utilise **7 sous-types d'interneurones** pour un connectome réaliste :

| Sous-type IN | Effectif | Rôle |
|-------------|----------|------|
| LN (Local Neurons) | 150 | Inhibition latérale AL |
| MB-FBN (FeedBack) | 80 | Boucle MBON→DAN→KC |
| MB-FFN (FeedForward) | 70 | Feed-forward KC→MBON |
| Pre-DNVNC | 60 | Prémoteur locomotion |
| Pre-DNSEZ | 50 | Prémoteur comportement |
| HEMI (Hémisphérique) | 89 | Communication gauche↔droite |
| GEN_IN (Générique) | 1000 | Connexions de fond, densité ~5% |

**Synapses totales** : ~535 000 (vs 548 000 cible), densité ~5.9%.

### Build et Lancement (C++)

```bash
# Prérequis : Qt6, Eigen3, CMake, compilateur C++20
sudo apt install qt6-base-dev libeigen3-dev cmake g++

cd QT && mkdir -p build && cd build
cmake .. && make -j$(nproc)
./drosophila_brain
```

### Contrôles (C++)

| Touche | Action |
|--------|--------|
| `R` | Récompense (DAN = +0.8) |
| `P` | Punition (DAN = -0.4) |
| `O` | Stimulus olfactif |
| `G` | Stimulus gustatif |
| `V` | Stimulus visuel |
| `T` | Stimulus thermique |
| `M` | Stimulation mécano |
| `S` | Step (pas de simulation) |
| `Espace` | Play/Pause |
| Glisser-souris | Pan (déplacement vue) |
| Molette | Zoom |

## Structure du Projet

```
drosophila_brain/
├── main.py                      # Point d'entrée / simulation standard
├── main_advanced.py             # Simulation avancée (circuits + monde 3D)
├── test_validation.py           # Suite de tests (7/7 passants)
├── config.py                    # Paramètres globaux
├── launch.sh                    # Script de lancement
├── requirements.txt             # Dépendances Python
│
├── core/                        # CŒUR DU MODÈLE NEURONAL
│   ├── neuron.py                # Neurone : sigmoïde + fuite temporelle
│   ├── synapse.py               # Synapse : 4 types + plasticité STDP
│   └── network.py               # Réseau 3016 neurones, propagation dynamique
│
├── circuits/                    # CIRCUITS ANATOMIQUES
│   ├── antennal_lobe.py         # Lobe antennaire : 176 ORN → 15 glomérules → PN
│   ├── mushroom_body.py         # Mushroom Body : 176 KC → 48 MBON → 30 DAN
│   ├── lateral_horn.py          # Corne latérale : valeurs innées
│   ├── interhemispheric.py      # Communication gauche/droite
│   └── brain_vnc_interface.py   # Cerveau ↔ moelle épinière
│
├── pathways/                    # FAISCEAUX DE NEURONES
│   ├── sensory_tracts.py        # 6 modalités sensorielles regroupées
│   ├── motor_tracts.py          # Commandes locomotion/comportement/endocrine
│   └── feedback_loops.py        # Boucles récurrentes (41% des neurones)
│
├── data/
│   └── connectome_synthetic.py  # Matrice 3016×3016 réaliste
│
├── learning/
│   └── reinforcement_learning.py # Hebb modulé par DANs (3 facteurs)
│
├── visualization/               # VISUALISATION
│   ├── real_time_plot.py        # 2D temps réel (activité + courbes + histogramme)
│   └── brain_3d.py              # 3D anatomique (régions colorées + connexions)
│
├── world/                       # MONDE VIRTUEL
│   ├── world.py                 # Monde 2D (odeur, nourriture, danger)
│   └── world_3d.py              # Monde 3D avancé (obstacles, lumière)
│
└── utils/
    ├── stats.py                 # Statistiques réseau (hubs, clustering)
    └── io.py                    # Sauvegarde/chargement états

QT/                              # Implémentation C++ Qt6 performante
├── CMakeLists.txt
├── build/
│   └── drosophila_brain         # Binaire compilé
└── src/
    ├── main.cpp
    ├── config.h
    ├── core/                    # Neurone, Synapse, Réseau (connectome structuré)
    ├── world/                   # Monde virtuel 2D
    ├── render/                  # QOpenGLWidget (shaders GLSL)
    └── ui/                      # Fenêtre + contrôles
```

## Installation

```bash
# Cloner ou extraire l'archive
cd drosophila_brain

# Installer les dépendances
pip install -r requirements.txt

# Rendre le script exécutable
chmod +x launch.sh
```

## Utilisation

### Lancement rapide

```bash
# Démonstration simple
python3 main.py

# Tests de validation (7/7)
./launch.sh --test

# Simulation avancée complète
./launch.sh --advanced

# Mode interactif
./launch.sh --interactive

# Visualisation temps réel
./launch.sh --visual
```

### Mode interactif

```
> stimulus olfactory 0.8    # Appliquer un stimulus olfactif
> reward                  # Appliquer une récompense
> punish                  # Appliquer une punition
> step                    # Avancer d'un pas de simulation
> status                  # Voir l'état du réseau
> quit                    # Quitter
```

## Modèle de Neurone

```
        entrées e₁...eₙ
           ↓
    ┌─────────────────────┐
    │  w₁, w₂, ... wₙ     │  ← poids synaptiques
    └─────────────────────┘
           ↓
      Σ(wᵢ × eᵢ) + biais   ← sommateur
           ↓
    dV/dt = -(V - V_rest)/τ + I_syn   ← fuite temporelle (τ = 10 ms)
           ↓
       V(t) intégré
           ↓
    σ(V) = 1 / (1 + e^(-10(V-0.5)))   ← sigmoïde standard
           ↓
        sortie ∈ [0,1]     ← taux de décharge
```

**Caractéristiques :**
- **Fuite temporelle** : mémoire courte du potentiel (τ = 10 ms)
- **Période réfractaire** : 2 ms après activation forte
- **4 types de synapses** : axo-dendritique (66.6%), axo-axonique (25.8%), dendro-dendritique (5.8%), dendro-axonique (1.8%)

## Apprentissage DAN-Modulé

| Événement | Signal DAN | Effet sur KC→MBON |
|-----------|-----------|-------------------|
| Sucre (récompense) | DAN = +1.0 | **Potentiation** (renforcement) |
| Choc (punition) | DAN = -0.5 | **Dépression** (affaiblissement) |
| Pas de signal | DAN = -0.001 | Oubli lent |

**Règle de trois facteurs :** `Δw = η × DAN(t) × (pré × post)`

## Visualisation Graphique

### 1. Visualisation 2D Temps Réel (`--visual`)

```bash
python3 main.py --visual
```

Affiche 4 panneaux :
- **Carte d'activité** : 3016 neurones colorés selon leur activation (rouge = actif, bleu = inactif)
- **Courbes temporelles** : activité par région (sensoriel, KC, MBON, DAN, outputs)
- **Histogramme** : distribution des poids synaptiques
- **Info texte** : temps, neurones actifs, activité moyenne

### 2. Visualisation 3D Anatomique

```python
from visualization.brain_3d import Brain3DVisualizer
from core.network import BrainNetwork

net = BrainNetwork()
viz = Brain3DVisualizer(net)
viz.plot_neurons()
viz.plot_region_boundaries()
viz.show()  # ou viz.save("brain.png")
```

Affiche :
- **Positions 3D** des neurones selon l'anatomie (X: gauche/droite, Y: antérieur/postérieur, Z: ventral/dorsal)
- **Régions colorées** : AL (vert), MB (rose), LH (cyan), Outputs (rouge)
- **Connexions** : lignes entre neurones connectés (sous-échantillonnées pour la performance)

### 3. Monde Virtuel 2D/3D

```python
from world.world_3d import VirtualWorld3D

world = VirtualWorld3D()
# La larve se déplace selon les commandes motrices du cerveau
```

### Légende des couleurs (C++ Qt6)

**Monde virtuel :**
| Couleur | Élément |
|---------|---------|
| <font color="#00ffff">`#00ffff`</font> (cyan) | Larve (position courante) |
| <font color="#00ff00">`#00ff00`</font> (vert) | Odeur attractive |
| <font color="#ff9900">`#ff9900`</font> (orange) | Odeur aversive |
| <font color="#888888">`#888888`</font> (gris) | Odeur neutre |
| <font color="#ffd700">`#ffd700`</font> (jaune) | Nourriture disponible |
| <font color="#ff0000">`#ff0000`</font> (rouge) | Zone de danger |
| <font color="#4d4d4d">`#4d4d4d`</font> (marron) | Nourriture consumée |
| <font color="#cccccc">`#cccccc`</font> (gris clair) | Obstacle |
| <font color="#666666">`#666666`</font> (gris moyen) | Trajectoire parcourue |

**Courbes d'activité cérébrale :**
| Couleur | Région |
|---------|--------|
| <font color="#ff6b6b">`#ff6b6b`</font> (rouge) | Sensoriel (entrées) |
| <font color="#48dbfb">`#48dbfb`</font> (cyan) | KC (Mushroom Body) |
| <font color="#ff9ff3">`#ff9ff3`</font> (rose) | MBON (sorties MB) |
| <font color="#54a0ff">`#54a0ff`</font> (bleu) | DAN (dopamine) |
| <font color="#1dd1a1">`#1dd1a1`</font> (vert) | DN (moteur) |

## Tests

Suite de 7 tests validant tous les composants :

| # | Test | Description |
|---|------|-------------|
| 1 | Neurone | Fuite, sigmoïde, réfractarité, synapses |
| 2 | Synapse | STDP, potentiation, dépression, limites |
| 3 | Connectome | Statistiques réalistes (3016 neurones, densité ~2%) |
| 4 | Propagation | Stimulus → activation → reset |
| 5 | Apprentissage | Signaux DAN, mises à jour, historique |
| 6 | Performance | Vitesse (~17 pas/s), mémoire (~92 MB) |
| 7 | Olfaction | Plasticité KC→MBON (récompense +, punition -) |

```bash
./launch.sh --test
```

## Résultats Typiques

```
============================================================
  RÉSULTATS: 7/7 tests réussis
============================================================

Construction du réseau cérébral (3016 neurones)...
✓ Réseau construit: 3016 neurones, 60126 synapses

[Performance]
  ✓ 100 pas en 5.9s (58.75ms/pas)
  ✓ Vitesse: 17 pas/seconde
  ✓ Mémoire utilisée: 92.5 MB

[Apprentissage Olfactif]
  ✓ Poids KC→MBON avant récompense: 0.023
  ✓ Poids KC→MBON après récompense: 1.829  (+7870%)
  ✓ Poids KC→MBON après punition: 0.002   (-99.9%)
  ✓ Plasticité fonctionnelle confirmée
```

## Références

- **Winding et al. (2023)**. *The connectome of an insect brain*. Science, 379(6636), eadd9330.
- **Eichler et al. (2017)**. *The complete connectome of a learning and memory centre in an insect brain*. Nature, 548(7666), 175-182.

## Auteur

**FOURNET Olivier** - olivier.fournet@free.fr

## Licence

GPL-3.0
