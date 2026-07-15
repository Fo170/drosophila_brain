#ifndef DROSOPHILA_NETWORK_H
#define DROSOPHILA_NETWORK_H

/*
 * network.h — Réseau de neurones du cerveau de Drosophila
 *
 * Architecture complète de 3016 neurones et ~548000 synapses,
 * basée sur le connectome de Winding et al. 2023 (Science).
 *
 * Flux de l'information :
 *   Entrées sensorielles (477) → PN (210) → [KC (176) + LHN (50)]
 *   → CN (108) → Sorties motrices (418)
 *                ↕
 *         MBON (48) ↔ DAN (30)
 *
 * Plasticité synaptique :
 *   STDP (Spike-Timing Dependent Plasticity) modulé par les signaux DAN
 *   (Dopaminergic Neurons). Une trace pré et post-synaptique est maintenue
 *   pour chaque synapse plastique. Le signal DAN (récompense/punition)
 *   déclenche la mise à jour du poids :
 *     Δw = η × DAN × (trace_pre × trace_post)
 */

#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <deque>
#include <array>

// ---------------------------------------------------------------------------
// Types de neurones (basés sur les types du connectome réel)
// ---------------------------------------------------------------------------
enum class NeuronType {
    ORN,       // Olfactory Receptor Neurons
    GRN,       // Gustatory Receptor Neurons
    PR,        // Photoreceptors
    THERMO,    // Thermoreceptors
    MECHANO,   // Mechanoreceptors
    PROPRIO,   // Proprioceptors
    AN,        // Ascending Neurons (remontée sensorielle du corps)
    PN,        // Projection Neurons
    KC,        // Kenyon Cells (Mushroom Body)
    MBON,      // Mushroom Body Output Neurons
    DAN,       // Dopaminergic Neurons (modulation apprentissage)
    LHN,       // Lateral Horn Neurons (voies innées)
    CN,        // Convergence Neurons
    DNVNC,     // Descending Neurons to VNC (commandes locomotrices)
    DNSEZ,     // Descending Neurons to SEZ (commandes comportementales)
    RGN,       // Ring Gland Neurons (hormones)
    IN         // Interneurons
};

// ---------------------------------------------------------------------------
// Types de synapses biologiques
//   a-d (66.6%) : axo-dendritique  → transmission standard (poids 1.0)
//   a-a (25.8%) : axo-axonique    → modulation présynaptique (poids 0.7)
//   d-d (5.8%)  : dendro-dendritique → interaction locale (poids 0.5)
//   d-a (1.8%)  : dendro-axonique → feedback (poids 0.4)
// ---------------------------------------------------------------------------
enum class SynapseType {
    A_D, A_A, D_D, D_A
};

// ---------------------------------------------------------------------------
// Neuron — Unité de calcul
//   Utilise un modèle à fuite (leaky integrator) avec activation sigmoïde.
//   Équation :
//     dV/dt = -(V - V_rest)/τ + I_syn
//     σ(V) = 1 / (1 + e^(-10(V-0.5)))
//   Le potentiel V s'accumule via les entrées synaptiques et fuit vers 0.
//   La sortie σ(V) ∈ [0,1] représente le taux de décharge.
// ---------------------------------------------------------------------------
struct Neuron {
    int id = 0;                     // Identifiant unique (0-3015)
    NeuronType type = NeuronType::IN;
    std::string hemisphere = "left";
    std::string compartment;        // Sous-région (glomérule, compartiment, etc.)

    float V = 0.0f;                 // Potentiel de membrane
    float output = 0.0f;            // Sortie sigmoïde σ(V)
    float refractory_timer = 0.0f;  // Temps restant en période réfractaire (ms)
    bool is_active = false;         // True si V > seuil + 0.3

    int n_presynaptic = 0;          // Nombre de synapses entrantes
    int n_postsynaptic = 0;         // Nombre de synapses sortantes
};

// ---------------------------------------------------------------------------
// Synapse — Connexion pondérée avec plasticité STDP
//
//   Traces STDP :
//     La trace pre (resp. post) augmente quand le neurone pré (resp. post)
//     est actif, et décroît exponentiellement avec une constante τ = 20ms.
//     Ces traces permettent de mesurer la corrélation temporelle entre
//     l'activité pré et post-synaptique.
//
//   Mise à jour du poids (STDP modulé par DAN) :
//     Quand un signal DAN arrive (récompense/punition) :
//       Δw = η × DAN × (pre_trace × post_trace)
//     Si DAN > 0 (récompense) : potentiation des synapses corrélées
//     Si DAN < 0 (punition) : dépression des synapses corrélées (×0.5)
//
//   Poids initial :
//     Basé sur le nombre de synapses physiques (log normalization)
//     et le type de synapse, avec ±20% de variabilité biologique.
// ---------------------------------------------------------------------------
struct Synapse {
    int pre_id;                     // ID du neurone présynaptique
    int post_id;                    // ID du neurone postsynaptique
    SynapseType type = SynapseType::A_D;
    float weight = 0.1f;            // Poids synaptique actuel
    float weight_initial = 0.1f;    // Poids initial (pour l'oubli progressif)
    bool plastic = true;            // True si la synapse est modifiable
    float pre_trace = 0.0f;         // Trace d'activité présynaptique (STDP)
    float post_trace = 0.0f;        // Trace d'activité postsynaptique (STDP)
};

// ---------------------------------------------------------------------------
// RegionActivity — Activité moyenne par région cérébrale
// ---------------------------------------------------------------------------
struct RegionActivity {
    float sensory = 0.0f;   // Moyenne de tous les sensoriels
    float PN = 0.0f;
    float KC = 0.0f;
    float MBON = 0.0f;
    float DAN = 0.0f;
    float CN = 0.0f;
    float DN_VNC = 0.0f;
    float DN_SEZ = 0.0f;
    float RGN = 0.0f;
};

// ---------------------------------------------------------------------------
// NetworkState — État complet du réseau à un instant donné
// ---------------------------------------------------------------------------
struct NetworkState {
    float time = 0.0f;
    int step = 0;
    int n_active = 0;
    float mean_activity = 0.0f;
    RegionActivity region_act;
    std::vector<float> neuron_outputs;  // Sortie de chaque neurone (pour affichage)
};

// ===========================================================================
// BrainNetwork — Réseau cérébral complet
// ===========================================================================
class BrainNetwork {
public:
    BrainNetwork(int seed = 42);

    // Exécute un pas de simulation (mise à jour de tous les neurones)
    void step(float dt);

    // Applique un stimulus sensoriel (olfaction, goût, vision, etc.)
    // active les neurones sensoriels correspondants
    void apply_stimulus(const std::string& type, float intensity, float duration);

    // Applique un signal dopaminergique (DAN) à toutes les synapses plastiques
    //   signal > 0 → récompense → potentiation des synapses actives
    //   signal < 0 → punition → dépression des synapses actives
    //   Déclenche la mise à jour STDP des poids synaptiques
    void apply_dan_signal(float signal, float dt = 0.001f);

    // Réinitialise complètement le réseau
    void reset();

    // Accesseurs pour les statistiques
    RegionActivity get_region_activity() const;
    NetworkState get_state() const;

    // Activité motrice par groupe fonctionnel
    //   forward  — propulsion avant (60 neurones)
    //   left_turn  — contraction côté gauche → virage à droite (45)
    //   right_turn — contraction côté droit → virage à gauche (45)
    //   backward  — propulsion arrière (30)
    float get_dn_vnc_forward() const;
    float get_dn_vnc_left_turn() const;
    float get_dn_vnc_right_turn() const;
    float get_dn_vnc_backward() const;

    const std::vector<Neuron>& neurons() const { return neurons_; }
    const std::vector<Synapse>& synapses() const { return synapses_; }
    int n_active_neurons() const;
    float mean_activity() const;
    const std::vector<int>& neurons_of_type(NeuronType t) const {
        return type_indices_.at(t);
    }
    float time() const { return time_; }

    // Accès aux indices des neurones DAN pour l'apprentissage
    const std::vector<int>& dan_neurons() const {
        return type_indices_.at(NeuronType::DAN);
    }

private:
    void build_network();       // Crée tous les neurones
    void create_synapses();     // Crée toutes les connexions synaptiques
    void add_synapse(int pre, int post, SynapseType type, float prob,
                     int min_n = 1, int max_n = 10);

    void update_stimuli(float dt);  // Supprime les stimuli expirés

    static float sigmoid(float x);

    // Données du réseau
    std::vector<Neuron> neurons_;
    std::vector<Synapse> synapses_;
    std::unordered_map<NeuronType, std::vector<int>> type_indices_;

    float time_ = 0.0f;
    int current_step_ = 0;

    // Stimuli actuellement appliqués
    struct ActiveStimulus {
        std::string type;
        float intensity;
        float duration;
        float start_time;
    };
    std::unordered_map<std::string, ActiveStimulus> active_stimuli_;
    int stim_counter_ = 0;

    mutable std::mt19937 rng_;  // Générateur aléatoire (seed = reproductibilité)

    // ======================== PARAMÈTRES ========================

    // Constantes du neurone
    static constexpr float TAU_MEMBRANE = 10.0f;       // (ms)
    static constexpr float V_REST = 0.0f;
    static constexpr float V_THRESHOLD = 0.5f;
    static constexpr float TAU_REFRACTORY = 2.0f;      // (ms)
    static constexpr float SIGMOID_STEEPNESS = 10.0f;
    static constexpr float SIGMOID_OFFSET = 0.0f;

    // Taille du réseau
    static constexpr int N_NEURONS = 3016;

    // Paramètres STDP
    static constexpr float STDP_WINDOW = 20.0f;        // τ des traces (ms)
    static constexpr float WEIGHT_SCALE = 0.1f;        // Échelle globale des poids
    static constexpr float LEARNING_RATE = 0.01f;      // η (taux apprentissage)
    static constexpr float MAX_WEIGHT = 5.0f * WEIGHT_SCALE;  // 0.5

    // Probabilités de connexion (basées sur Winding et al. 2023)
    static constexpr float P_SENSORY_TO_PN = 0.3f;
    static constexpr float P_PN_TO_MB = 0.25f;
    static constexpr float P_PN_TO_LH = 0.2f;
    static constexpr float P_KC_TO_MBON = 0.15f;
    static constexpr float P_MBON_TO_CN = 0.4f;
    static constexpr float P_CN_TO_OUTPUT = 0.3f;
    static constexpr float P_MBON_TO_DAN = 0.2f;
    static constexpr float P_DAN_TO_KC = 0.1f;
    static constexpr float P_INTERHEMISPHERIC = 0.15f;
    static constexpr float P_DN_FEEDBACK = 0.1f;

    // Regroupements fonctionnels
    static constexpr int N_ORN_GLOMERULI = 15;
    static constexpr int N_GRN_ZONES = 8;
    static constexpr int N_VISUAL_COLUMNS = 5;
    static constexpr int N_SOMATO_TRACTS = 3;
    static constexpr int N_KC_COMPARTMENTS = 7;
    static constexpr int N_MBON_TYPES = 15;
    static constexpr int N_DAN_CLUSTERS = 10;
};

#endif
