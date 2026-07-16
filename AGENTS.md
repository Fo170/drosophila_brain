# AGENTS - Guide des Agents du Simulateur

Ce document décrit les agents (modules autonomes) du simulateur de cerveau de *Drosophila melanogaster*.

---

## Agent : Neurone (`core/neuron.py`)

**Rôle** : Unité de calcul fondamentale avec dynamique temporelle.

**Entrées** :
- Signaux des neurones présynaptiques (0-1)
- Poids synaptiques (float)
- Type de synapse (a-d, a-a, d-d, d-a)

**Sorties** :
- Potentiel de membrane V(t) (float, fuite exponentielle)
- Sortie sigmoïde σ(V) ∈ [0,1] (taux de décharge)
- État actif/inactif (bool)

**Équations** :
```
dV/dt = -(V - V_rest)/τ + I_syn
σ(V) = 1 / (1 + e^(-10(V-0.5)))
```

**Mémoire** :
- Potentiel V (persiste entre pas de temps)
- Timer réfractaire
- Historique des spikes

---

## Agent : Synapse (`core/synapse.py`)

**Rôle** : Connexion pondérée avec plasticité biologique.

**Types** :
| Type | Fraction | Fonction | Poids relatif |
|------|----------|----------|---------------|
| a-d (axo-dendritique) | 66.6% | Transmission standard | 1.0 |
| a-a (axo-axonique) | 25.8% | Modulation présynaptique | 0.7 |
| d-d (dendro-dendritique) | 5.8% | Interaction locale | 0.5 |
| d-a (dendro-axonique) | 1.8% | Feedback vers l'axon | 0.4 |

**Plasticité** :
- **STDP** : traces pré/post avec fenêtre temporelle (τ = 20 ms)
- **Règle 3 facteurs** : Δw = η × DAN(t) × (pré × post)
- **Limites** : poids ∈ [0, 2.0]

---

## Agent : Réseau Cérébral (`core/network.py`)

**Rôle** : Orchestration de 3016 neurones en propagation dynamique.

**Architecture** :
```
Inputs (477) → PNs (210) → [KC (176) + LHN (50)] → CN (108) → Outputs (418)
                              ↓
                           MBON (48) ↔ DAN (30)
```

**Ordre de mise à jour** (par groupe) :
1. Sensoriels (ORN, GRN, PR, thermo, mécano, proprio)
2. Projection Neurons (PN)
3. Lateral Horn Neurons (LHN)
4. Kenyon Cells (KC)
5. Mushroom Body Output Neurons (MBON)
6. Dopaminergic Neurons (DAN)
7. Convergence Neurons (CN)
8. Descending Neurons (DNVNC, DNSEZ)
9. Ring Gland Neurons (RGN)
10. Interneurons (IN)

**Sous-types d'interneurones** (implémentation C++ structurée) :

| Sous-type IN | Effectif | Connexions | Rôle |
|-------------|----------|-----------|------|
| LN | 150 | AL (glomérules) | Inhibition latérale |
| MB-FBN | 80 | MBON ↔ DAN ↔ KC | Boucle d'apprentissage |
| MB-FFN | 70 | KC → MBON | Feed-forward |
| Pre-DNVNC | 60 | CN → DNVNC | Prémoteur locomotion |
| Pre-DNSEZ | 50 | CN → DNSEZ | Prémoteur comportement |
| HEMI | 89 | G↔D hémisphères | Communication interhémisphérique |
| GEN_IN | 1000 | Tous | Connexions de fond (densité ~5%) |

**Boucles récurrentes** :
- MBON → DAN → KC (apprentissage)
- DN → interneurons (efference copy)
- Interhémisphérique (gauche ↔ droite)

---

## Agent : Connectome Synthétique (`data/connectome_synthetic.py` / `core/network.h`)

**Rôle** : Génère une matrice de connectivité 3016×3016 réaliste.

**Statistiques cibles** (Winding et al. 2023) :
- 548 000 synapses
- Densité ~2%
- 41% de neurones en boucles récurrentes
- 73% des hubs liés au Mushroom Body

**Implémentation Python** (`data/connectome_synthetic.py`):
1. Assignation des types de neurones par effectifs
2. Connexions par région (sensory→PN, PN→MB, etc.)
3. Boucles récurrentes
4. Connexions interhémisphériques
5. Connexions de fond pour atteindre la densité cible

**Implémentation C++** (`core/network.h`) — connectome structuré avec 7 sous-types d'IN :
1. Connexions sensorielles structurées (ORN→glomérules→PN, GRN→SEZ, PR→centres visuels, etc.)
2. Connexions PN→KC (glomérule→compartiment) et PN→LHN (toutes modalités)
3. Connexions KC→MBON par compartiment (176 KC × 7 compartiments → 48 MBON)
4. Boucle MBON→DAN→KC (apprentissage récurrent)
5. Connexions MBON/LHN/PN→CN (convergence)
6. CN→DN (groupe fonctionnel : FWD/LTL/LTR/BWD)
7. Connexions DN→IN (efference copy) et IN→DN (proprioception prédictive)
8. Connexions interhémisphériques (HEMI : gauche↔droite)
9. Connexions de fond (GEN_IN) : ~5% densité
10. Distribution biologique des types synaptiques : a-d 66%, a-a 24.5%, d-d 6.4%, d-a 3.1%

**Résultat C++** : ~535 000 synapses (~548 000 cible), densité ~5.9%

---

## Agent : Générateur SVG du Connectome (`vue connectome/generateur_connectome_drosophila.py`)

**Rôle** : Génère des diagrammes SVG interactifs du connectome larvaire.

**Architecture fonctionnelle** (10 régions cérébrales) :
| Région | Neurones | Type | Rôle |
|--------|----------|------|------|
| AL (Lobe antennaire) | 285 | sensoriel | Olfaction primaire |
| BO (Organes de Bolwig) | 24 | sensoriel | Photoréception |
| CH (Organes chordotonaux) | 156 | sensoriel | Mécano/proprioception |
| MB (Mushroom Body) | 250 | intégrateur | Apprentissage et mémoire |
| LH (Corne latérale) | 380 | intégrateur | Intégration multimodale |
| SLP (Plaque supra-latérale) | 420 | intégrateur | Cortex associatif |
| SIP (Plaque intermediaire) | 310 | intégrateur | Transition sensorimotrice |
| CRE (Crête cérébrale) | 290 | intégrateur | Coordination motrice |
| SMP (Plaque supra-médiane) | 340 | intégrateur | Modulation peptidergique |
| INP (Protuberance intercalaire) | 180 | commissure | Communication interhémisphérique |
| VNC (Cordon nerveux ventral) | 381 | moteur | Motoneurones et locomotion |

**Types de faisceaux** :
| Type | Couleur | Exemple |
|------|---------|---------|
| Sensoriel ↑ | Orange | AL → LH (28 500 syn.) |
| Moteur ↓ | Rouge | CRE → VNC (18 600 syn.) |
| Récurrent ↻ | Rose | MB ↔ MB (15 400 syn.) |
| Commissure ↔ | Cyan | INP ↔ AL (8 600 syn.) |
| Associatif ⟷ | Violet | LH → SLP (22 400 syn.) |
| Afférent ↑ | Vert | VNC → CH (11 200 syn.) |

**Interactivité** :
- **Survol région** : cercle illuminé, texte en blanc
- **Survol connexion** : trait devient très visible
- **Tooltips** (`<title>`) : infos détaillées en survol
- **Barre de navigation** : bascule entre 5 vues
- **Régions cliquables** : lien vers la vue correspondante

**Vues générées** :
- `full` — Vue d'ensemble complète (tous les faisceaux)
- `sensory` — Flux sensoriels ascendants uniquement
- `motor` — Flux moteurs descendants uniquement
- `recurrent` — Boucles récurrentes et commissures
- `associative` — Réseau associatif intra-cérébral

**Données JSON** : `connectome_data.json` exporté pour utilisation externe (D3.js, Cytoscape)

---

## Agent : Apprentissage DAN-Modulé (`learning/reinforcement_learning.py`)

**Rôle** : Modifie les poids KC→MBON selon les signaux de récompense/punition.

**Architecture** :
```
Stimulus → KC → MBON → Action
              ↑
        Récompense/Punition → DANs → Modulation
```

**Signaux DAN** :
- **Récompense** (sucre) : DAN = +0.8 → potentiation
- **Punition** (choc) : DAN = -0.4 → dépression
- **Neutre** : DAN = -0.001 → oubli lent

**Synchronisation** :
- Modifie `syn.weight` (objet Synapse)
- ET met à jour les poids dans `neuron.presynaptic` / `neuron.postsynaptic`

---

## Agent : Lobe Antennaire (`circuits/antennal_lobe.py`)

**Rôle** : Premier relais du traitement olfactif.

**Architecture** :
```
176 ORN → 15 glomérules virtuels → 210 PN
                ↓
         Inhibition latérale (LN)
```

**Fonction** :
- Convergence spatiale des ORN par glomérule
- Inhibition latérale entre glomérules (contrast enhancement)
- Transformation odeur → motif d'activation PN

---

## Agent : Mushroom Body (`circuits/mushroom_body.py`)

**Rôle** : Centre d'apprentissage et mémoire associative.

**Architecture** :
```
176 KC (7 compartiments) → 48 MBON (15 types) → CN
         ↓                        ↑
      DAN (30, 10 clusters) ← MBON feedback
```

**Sparsité** : ~15% des KC actifs par stimulus (représentation éparses)

**Valence** : MBON positifs (approche) vs négatifs (fuite)

---

## Agent : Corne Latérale (`circuits/lateral_horn.py`)

**Rôle** : Centre des valeurs innées (non apprises).

**Architecture** :
```
PN (toutes modalités) → 50 LHN → CN
```

**Valences** :
- Attractives (1/3 des LHN) : odeurs alimentaires
- Aversives (1/3 des LHN) : CO₂, danger
- Neutres (1/3 des LHN)

**Différence avec MB** : pas de plasticité, réponses fixes.

---

## Agent : Interface Cerveau-VNC (`circuits/brain_vnc_interface.py`)

**Rôle** : Traduit les décisions cérébrales en commandes motrices.

**Architecture** :
```
Cerveau (CN, MBON)
   ↓
DNVNC (180) → muscles segmentaires (locomotion)
  ├─ FWD (60) : avancer
  ├─ LTL (45) : tourner à gauche
  ├─ LTR (45) : tourner à droite
  └─ BWD (30) : reculer
DNSEZ (54) → comportements (alimentation)
  200 AN ← feedback sensoriel du corps
```

**Commandes locomotrices** :
- **Formule mouvement** : `vitesse = FWD − BWD`, `virage = LTR − LTL`
- Permet toutes les combinaisons : avancer, reculer, tourner, arrêt

**Efference copy** : DNs → interneurons (prédiction du mouvement)

---

## Agent : Faisceaux Sensoriels (`pathways/sensory_tracts.py`)

**Rôle** : Regroupe les neurones sensoriels par modalité.

**Modalités** :
| Modalité | Neurones | Cible | Rôle |
|----------|----------|-------|------|
| Olfaction | 176 ORN | AL | Identification odeurs |
| Gustation | 42 GRN | SEZ | Goût alimentaire |
| Vision | 29 PR | Centres visuels | Phototaxie |
| Thermo | 8 thermoR | Centres thermiques | Évitement chaleur |
| Mécano | 10 mécanoR + 200 AN | VNC | Toucher, proprioception |
| Proprio | 12 proprioR | VNC | Position corporelle |

---

## Agent : Faisceaux Moteurs (`pathways/motor_tracts.py`)

**Rôle** : Organise les commandes de sortie.

**Commandes** (C++ — 4 groupes fonctionnels DNVNC) :
- **Locomotion** : DNVNC → `vitesse = FWD − BWD`, `virage = LTR − LTL`
- **Comportement** : DNSEZ → manger/nettoyer/reposer
- **Endocrine** : RGN → hormones (croissance, métabolisme)

---

## Agent : Boucles de Feedback (`pathways/feedback_loops.py`)

**Rôle** : Identifie et mesure la récurrence dans le réseau.

**Types de boucles** :
1. **MB récurrent** : MBON → DAN → KC (apprentissage)
2. **DN feedback** : DN → interneurons (efference copy)
3. **Output-Input** : sorties → capteurs (ré-entrée)
4. **Interhémisphérique** : gauche ↔ droite

**Statistique** : 41% des neurones en boucles récurrentes.

---

## Agent : Monde Virtuel 2D (`world/world.py`)

**Rôle** : Environnement simple pour l'insecte.

**Entités** :
- Sources d'odeur (attractives/aversives)
- Nourriture (sucre, levure)
- Zones de danger (chaleur)

**Boucle** :
```
Monde → Stimuli sensoriels → Cerveau → Action → Monde
```

---

## Agent : Monde Virtuel 3D (`world/world_3d.py`)

**Rôle** : Environnement 3D avancé.

**Entités supplémentaires** :
- Obstacles (parois)
- Lumière directionnelle (phototaxie)
- Odeurs avec décroissance 3D
- Trajectoire de la larve

**Capteurs 3D** :
- Olfaction : gradient spatial
- Visuel : direction de la lumière
- Proprioception : orientation 3D (heading, pitch, speed)

---

## Agent : Visualisation 2D Python (`visualization/real_time_plot.py`)

**Rôle** : Affiche l'activité cérébrale en temps réel (Python).

**Panneaux** :
1. **Carte 2D** : 3016 neurones colorés (rouge = actif, bleu = inactif)
2. **Courbes temporelles** : activité par région
3. **Histogramme** : distribution des poids synaptiques
4. **Info texte** : temps, neurones actifs, activité moyenne

**Mise à jour** : toutes les 10 pas de simulation.

---

## Agent : Visualisation GLWidget (`render/glwidget.h/.cpp` — C++ Qt6)

**Rôle** : Rendu OpenGL temps réel performant du cerveau et du monde virtuel.

**Pipeline** :
- **Shaders GLSL** pour les points (neurones, entités du monde) et lignes (trajectoire)
- **Pipeline fixe** pour la bordure du monde
- Mise à jour différée (dirty flag) des VBO pour minimiser les transferts GPU

**Éléments affichés** :
- **Points colorés** : larve (blanc), odeurs attractives (vert), odeurs aversives (rouge), nourriture (jaune), obstacles (gris)
- **Trajectoire** : lignes grises continues de la larve
- **Bordure** : rectangle gris-bleu à 95% de la zone visible

**Caméra** :
- **Fixée** sur le centre du monde (pas de suivi automatique)
- **Pan** : glisser-souris
- **Zoom** : molette

**Fenêtre** : 1400×900 pixels, redimensionnement désactivé (`setFixedSize`)

**Panneaux** :
1. **Carte 2D** : 3016 neurones colorés (rouge = actif, bleu = inactif)
2. **Courbes temporelles** : activité par région
3. **Histogramme** : distribution des poids synaptiques
4. **Info texte** : temps, neurones actifs, activité moyenne

**Mise à jour** : toutes les 10 pas de simulation.

---

## Agent : Visualisation 3D Python (`visualization/brain_3d.py`)

**Rôle** : Vue anatomique 3D du cerveau.

**Coordonnées** :
- X : gauche (-) → droite (+)
- Y : antérieur (-) → postérieur (+)
- Z : ventral (-) → dorsal (+)

**Éléments** :
- Points colorés par type de neurone
- Boîtes transparentes pour les régions (AL, MB, LH, Outputs)
- Connexions sous-échantillonnées (5000 max)

---

## Agent : Statistiques (`utils/stats.py`)

**Rôle** : Analyse topologique du réseau.

**Métriques** :
- Distribution des degrés (in/out)
- Identification des hubs (percentile 95)
- Coefficient de clustering
- Densité du réseau

---

## Agent : IO (`utils/io.py`)

**Rôle** : Persistance des états du réseau.

**Formats** :
- **Pickle** : état complet (neurones + synapses)
- **NPZ** : matrices de connectivité
- **CSV** : log d'activité temporelle

---

## Interactions entre Agents

```
┌─────────────────────────────────────────────────────────────┐
│                        FLUX DE DONNÉES                       │
└─────────────────────────────────────────────────────────────┘

Monde Virtuel ──► Sensory Tracts ──► Circuits (AL, MB, LH)
                                         │
                                         ▼
                              Brain Network (3016 neurones)
                                         │
                                         ▼
                              Motor Tracts ──► Monde Virtuel
                                         │
                                         ▼
                              Feedback Loops ──► Brain Network
                                         │
                                         ▼
                                Learning (DAN-modulé) ──► Synapses
                                           │
                                           ▼
                                Visualization (Python 2D/3D + C++ GLWidget)
                                           │
                                           ▼
                             SVG Generator (diagrammes interactifs)
```

---

## Paramètres Clés (`config.py`)

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| N_NEURONS | 3016 | Neurones totaux |
| N_SYNAPSES | 548000 | Synapses totales |
| TAU_MEMBRANE | 10.0 ms | Constante de fuite |
| DT | 0.1 ms | Pas de temps |
| LEARNING_RATE | 0.01 | Taux d'apprentissage |
| DAN_REWARD_SCALE | 1.0 | Échelle DAN récompense |
| DAN_PUNISH_SCALE | -0.5 | Échelle DAN punition |
| N_KC_COMPARTMENTS | 7 | Compartiments KC |
| N_MBON_TYPES | 15 | Types MBON |
| N_DAN_CLUSTERS | 10 | Clusters DAN |

---

## Performance

| Métrique | Valeur |
|----------|--------|
| Vitesse simulation | ~17 pas/seconde |
| Mémoire | ~92 MB |
| Temps pour 100 pas | ~5.9 secondes |
| Neurones actifs (stimulus) | ~133 (4.4%) |

---

*Document généré automatiquement le 2026-07-16*
