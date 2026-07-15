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

**Boucles récurrentes** :
- MBON → DAN → KC (apprentissage)
- DN → interneurons (efference copy)
- Interhémisphérique (gauche ↔ droite)

---

## Agent : Connectome Synthétique (`data/connectome_synthetic.py`)

**Rôle** : Génère une matrice de connectivité 3016×3016 réaliste.

**Statistiques cibles** (Winding et al. 2023) :
- 548 000 synapses
- Densité ~2%
- 41% de neurones en boucles récurrentes
- 73% des hubs liés au Mushroom Body

**Génération** :
1. Assignation des types de neurones par effectifs
2. Connexions par région (sensory→PN, PN→MB, etc.)
3. Boucles récurrentes
4. Connexions interhémisphériques
5. Connexions de fond pour atteindre la densité cible

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
180 DNVNC → muscles segmentaires (locomotion)
 54 DNSEZ → comportements (alimentation)
  200 AN ← feedback sensoriel du corps
```

**Commandes locomotrices** :
- Forward / Backward / Turn Left / Turn Right / Stop

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

**Commandes** :
- **Locomotion** : DNVNC → avancer/reculer/tourner/arrêt
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

## Agent : Visualisation 2D (`visualization/real_time_plot.py`)

**Rôle** : Affiche l'activité cérébrale en temps réel.

**Panneaux** :
1. **Carte 2D** : 3016 neurones colorés (rouge = actif, bleu = inactif)
2. **Courbes temporelles** : activité par région
3. **Histogramme** : distribution des poids synaptiques
4. **Info texte** : temps, neurones actifs, activité moyenne

**Mise à jour** : toutes les 10 pas de simulation.

---

## Agent : Visualisation 3D (`visualization/brain_3d.py`)

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
                              Visualization (2D/3D)
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

*Document généré automatiquement le 2026-07-15*
