#include "network.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>

// ===========================================================================
// Constructeur : initialise le générateur aléatoire et construit le réseau
// ===========================================================================
BrainNetwork::BrainNetwork(int seed) : rng_(seed) {
    build_network();
}

// ===========================================================================
// sigmoid — Fonction d'activation biologique
//   σ(V) = 1 / (1 + e^(-10(V-0.5)))
//   Donne une sortie ∈ [0,1] avec une transition raide autour de V=0.5
// ===========================================================================
float BrainNetwork::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-SIGMOID_STEEPNESS
                                    * (x - V_THRESHOLD + SIGMOID_OFFSET)));
}

// ===========================================================================
// build_network — Crée les 3016 neurones avec leurs types et régions
//
// Ordre de création (important pour les IDs) :
//   1. Sensoriels (ORN, GRN, PR, thermo, mechano, proprio, AN)
//   2. Projection Neurons (PN)
//   3. Mushroom Body (KC, MBON, DAN)
//   4. Lateral Horn (LHN)
//   5. Convergence (CN)
//   6. Outputs (DNVNC, DNSEZ, RGN)
//   7. Interneurons restants
// ===========================================================================
void BrainNetwork::build_network() {
    neurons_.clear();
    synapses_.clear();
    type_indices_.clear();
    neurons_.reserve(N_NEURONS);
    int id = 0;

    // Lambda pour créer un groupe de neurones
    auto make_neurons = [&](int count, NeuronType type,
                            const std::string& hemi,
                            const std::string& comp_prefix,
                            int group_size) {
        for (int i = 0; i < count; i++) {
            Neuron n;
            n.id = id;
            n.type = type;
            n.hemisphere = hemi;
            if (group_size > 0) {
                n.compartment = comp_prefix + std::to_string(i / group_size);
            } else {
                n.compartment = comp_prefix;
            }
            neurons_.push_back(n);
            type_indices_[type].push_back(id);
            id++;
        }
    };

    auto make_grouped = [&](int total, NeuronType type,
                            const std::string& prefix, int n_groups) {
        make_neurons(total, type, "left", prefix + "_",
                     std::max(1, total / n_groups));
    };

    // ---- 1. Entrées sensorielles (477 neurones) ----
    make_grouped(176, NeuronType::ORN,   "AL_glom",  N_ORN_GLOMERULI);
    make_grouped(42,  NeuronType::GRN,   "GRN_zone", N_GRN_ZONES);
    make_grouped(29,  NeuronType::PR,    "VIS_col",  N_VISUAL_COLUMNS);
    make_neurons(8,   NeuronType::THERMO, "left", "thermo", 0);
    make_neurons(10,  NeuronType::MECHANO,"left", "mechano", 0);
    make_neurons(12,  NeuronType::PROPRIO,"left", "proprio", 0);
    make_neurons(200, NeuronType::AN,    "left", "VNC_A1_", 50);

    // ---- 2. Projection Neurons (210) ----
    make_grouped(210, NeuronType::PN,    "PN",      30);

    // ---- 3. Mushroom Body ----
    make_grouped(176, NeuronType::KC,    "MB_comp", N_KC_COMPARTMENTS);
    make_grouped(48,  NeuronType::MBON,  "MBON_type", N_MBON_TYPES);
    make_grouped(30,  NeuronType::DAN,   "DAN_clust", N_DAN_CLUSTERS);

    // ---- 4. Lateral Horn (50) ----
    make_neurons(50,  NeuronType::LHN,   "left", "LH_", 10);

    // ---- 5. Convergence Neurons (108) ----
    make_grouped(108, NeuronType::CN,    "CN",     20);

    // ---- 6. Descending Neurons ----
    // DNVNC : 4 groupes fonctionnels pour une locomotion réaliste
    //   Forward (60) : propulsion avant (les deux côtés en phase)
    //   Left-turn (45) : contraction côté gauche → virage à droite
    //   Right-turn (45) : contraction côté droit → virage à gauche
    //   Backward (30) : propulsion arrière
    make_neurons(60,  NeuronType::DNVNC, "left", "DNVNC_FWD", 0);
    make_neurons(45,  NeuronType::DNVNC, "left", "DNVNC_LTL", 0);
    make_neurons(45,  NeuronType::DNVNC, "left", "DNVNC_LTR", 0);
    make_neurons(30,  NeuronType::DNVNC, "left", "DNVNC_BWD", 0);
    make_grouped(54,  NeuronType::DNSEZ, "DNSEZ_", 10);
    make_grouped(184, NeuronType::RGN,   "RGN_",   30);

    // ---- 7. Interneurons restants ----
    int remaining = N_NEURONS - (int)neurons_.size();
    if (remaining > 0) {
        make_grouped(remaining, NeuronType::IN, "IN_", 50);
    }

    // Vérification du compte total
    assert((int)neurons_.size() == N_NEURONS);

    // Création des connexions synaptiques
    create_synapses();
}

// ===========================================================================
// create_synapses — Crée ~548000 synapses entre les neurones
//
//   Poids initial d'une synapse :
//     weight = log1p(n_synapses) / log1p(50) × type_scale × WEIGHT_SCALE
//              × variabilité_biologique(±20%)
//
//   Types de connexions par région :
//     Sensoriel → PN : transmission sensorielle
//     PN → KC : entrée du Mushroom Body (apprentissage)
//     PN → LHN : voie innée (pas d'apprentissage)
//     KC → MBON : sortie du MB
//     MBON → CN : intégration des valeurs
//     LHN → CN : valeurs innées
//     CN → DN/RGN : commandes motrices
//     MBON → DAN : feedback dopaminergique
//     DAN → KC : modulation de l'apprentissage
// ===========================================================================
void BrainNetwork::create_synapses() {
    auto& orns  = type_indices_[NeuronType::ORN];
    auto& grns  = type_indices_[NeuronType::GRN];
    auto& prs   = type_indices_[NeuronType::PR];
    auto& therm = type_indices_[NeuronType::THERMO];
    auto& mech  = type_indices_[NeuronType::MECHANO];
    auto& prop  = type_indices_[NeuronType::PROPRIO];
    auto& ans   = type_indices_[NeuronType::AN];
    auto& pns   = type_indices_[NeuronType::PN];
    auto& kcs   = type_indices_[NeuronType::KC];
    auto& mbons = type_indices_[NeuronType::MBON];
    auto& dans  = type_indices_[NeuronType::DAN];
    auto& lhns  = type_indices_[NeuronType::LHN];
    auto& cns   = type_indices_[NeuronType::CN];
    auto& dnvnc = type_indices_[NeuronType::DNVNC];
    auto& dnsez = type_indices_[NeuronType::DNSEZ];
    auto& rgns  = type_indices_[NeuronType::RGN];
    auto& ins   = type_indices_[NeuronType::IN];

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Fonction pour ajouter une synapse avec poids initial aléatoire
    auto add_syn_with_weight = [&](int pre, int post, SynapseType type,
                                   float prob, int min_n = 1, int max_n = 10) {
        if (dist(rng_) < prob) {
            // Nombre de synapses physiques (1-50)
            int ns = std::uniform_int_distribution<int>(min_n, max_n)(rng_);

            // Poids basé sur le nombre de synapses (loi log)
            float norm = std::log1p((float)ns) / std::log1p(50.0f);

            // Échelle selon le type de synapse
            float type_scale = 1.0f;
            switch (type) {
                case SynapseType::A_D: type_scale = 1.0f; break;
                case SynapseType::A_A: type_scale = 0.7f; break;
                case SynapseType::D_D: type_scale = 0.5f; break;
                case SynapseType::D_A: type_scale = 0.4f; break;
            }

            // Variabilité biologique : ±20% aléatoire
            float var = std::uniform_real_distribution<float>(0.8f, 1.2f)(rng_);
            float w = norm * type_scale * WEIGHT_SCALE * var;

            Synapse s;
            s.pre_id = pre;
            s.post_id = post;
            s.type = type;
            s.weight = std::max(0.0f, w);
            s.weight_initial = s.weight;  // Sauvegarde pour l'oubli progressif
            // Seules les synapses a-d et d-a sont plastiques (modifiables)
            s.plastic = (type == SynapseType::A_D || type == SynapseType::D_A);
            synapses_.push_back(s);

            neurons_[pre].n_postsynaptic++;
            neurons_[post].n_presynaptic++;
        }
    };

    // Fonction pour connecter deux groupes de neurones
    auto connect_groups = [&](const std::vector<int>& pre,
                              const std::vector<int>& post,
                              SynapseType type, float prob,
                              int mn = 1, int mx = 8) {
        for (int p : pre)
            for (int q : post)
                add_syn_with_weight(p, q, type, prob, mn, mx);
    };

    // Regroupe tous les neurones sensoriels
    auto all_sensory = [&]() -> std::vector<int> {
        std::vector<int> r;
        for (auto* v : {&orns, &grns, &prs, &therm, &mech, &prop, &ans})
            r.insert(r.end(), v->begin(), v->end());
        return r;
    };

    // --- Feedforward : sensoriel → PN → MB/LH → CN → sortie ---
    connect_groups(all_sensory(), pns, SynapseType::A_D,
                   P_SENSORY_TO_PN, 1, 8);
    connect_groups(pns, kcs, SynapseType::A_D, P_PN_TO_MB, 1, 5);
    connect_groups(pns, lhns, SynapseType::A_D, P_PN_TO_LH, 1, 6);
    connect_groups(kcs, mbons, SynapseType::A_D, P_KC_TO_MBON, 1, 3);
    connect_groups(mbons, cns, SynapseType::A_D, P_MBON_TO_CN, 1, 8);
    connect_groups(lhns, cns, SynapseType::A_D, P_MBON_TO_CN * 0.8f, 1, 6);

    // CN → Descending Neurons (commandes motrices)
    for (int cn : cns) {
        for (int dn : dnvnc)
            add_syn_with_weight(cn, dn, SynapseType::A_D,
                                P_CN_TO_OUTPUT, 1, 10);
        for (int dn : dnsez)
            add_syn_with_weight(cn, dn, SynapseType::A_D,
                                P_CN_TO_OUTPUT, 1, 10);
        for (int rg : rgns)
            add_syn_with_weight(cn, rg, SynapseType::A_D,
                                P_CN_TO_OUTPUT * 0.5f, 1, 5);
    }

    // --- Boucles récurrentes d'apprentissage ---
    // MBON → DAN (feedback, type a-a : modulation)
    connect_groups(mbons, dans, SynapseType::A_A, P_MBON_TO_DAN, 1, 5);
    // DAN → KC (modulation, type d-a : feedback dendro-axonique)
    connect_groups(dans, kcs, SynapseType::D_A, P_DAN_TO_KC, 1, 4);

    // DN → Interneurons (efference copy : prédiction du mouvement)
    for (int dn : dnvnc)
        for (int i = 0; i < 100 && i < (int)ins.size(); i++)
            add_syn_with_weight(dn, ins[i], SynapseType::D_A,
                                P_DN_FEEDBACK, 1, 3);
    for (int dn : dnsez)
        for (int i = 0; i < 100 && i < (int)ins.size(); i++)
            add_syn_with_weight(dn, ins[i], SynapseType::D_A,
                                P_DN_FEEDBACK, 1, 3);
}

// ===========================================================================
// add_synapse — Version simplifiée pour ajouter une synapse unique
//   Utilisée quand on veut créer une synapse sans passer par connect_groups
// ===========================================================================
void BrainNetwork::add_synapse(int pre, int post, SynapseType type,
                               float prob, int min_n, int max_n) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(rng_) < prob) {
        int ns = std::uniform_int_distribution<int>(min_n, max_n)(rng_);
        float norm = std::log1p((float)ns) / std::log1p(50.0f);
        float type_scale = 1.0f;
        switch (type) {
            case SynapseType::A_D: type_scale = 1.0f; break;
            case SynapseType::A_A: type_scale = 0.7f; break;
            case SynapseType::D_D: type_scale = 0.5f; break;
            case SynapseType::D_A: type_scale = 0.4f; break;
        }
        float var = std::uniform_real_distribution<float>(0.8f, 1.2f)(rng_);
        float w = norm * type_scale * WEIGHT_SCALE * var;

        Synapse s;
        s.pre_id = pre;
        s.post_id = post;
        s.type = type;
        s.weight = std::max(0.0f, w);
        s.weight_initial = s.weight;
        s.plastic = (type == SynapseType::A_D || type == SynapseType::D_A);
        synapses_.push_back(s);
    }
}

// ===========================================================================
// apply_stimulus — Active un groupe de neurones sensoriels
//
//   Quand un stimulus est appliqué, tous les neurones du type correspondant
//   sont activés avec l'intensité spécifiée (+ bruit gaussien).
//
//   La durée du stimulus est enregistrée ; après expiration, les neurones
//   retournent à leur comportement par défaut (régis par l'équation de fuite).
// ===========================================================================
void BrainNetwork::apply_stimulus(const std::string& type, float intensity,
                                   float duration) {
    std::string key = type + "_" + std::to_string(stim_counter_++);
    active_stimuli_[key] = {type, intensity, duration, time_};

    // Mapping entre les noms de stimulus et les types de neurones
    static const std::unordered_map<std::string, NeuronType> s2t = {
        {"olfactory", NeuronType::ORN},
        {"gustatory", NeuronType::GRN},
        {"visual", NeuronType::PR},
        {"thermal", NeuronType::THERMO},
        {"mechano", NeuronType::MECHANO},
        {"proprioceptive", NeuronType::PROPRIO}
    };

    auto it = s2t.find(type);
    if (it == s2t.end()) return;

    NeuronType nt = it->second;
    auto& idxs = type_indices_[nt];

    // Activation des neurones avec bruit biologique
    std::normal_distribution<float> noise(0.0f, 0.1f);
    for (int idx : idxs) {
        neurons_[idx].V = intensity + noise(rng_);
        neurons_[idx].output = sigmoid(neurons_[idx].V);
    }
}

// ===========================================================================
// update_stimuli — Supprime les stimuli dont la durée est expirée
// ===========================================================================
void BrainNetwork::update_stimuli(float dt) {
    std::vector<std::string> expired;
    for (auto& [key, stim] : active_stimuli_) {
        if (time_ - stim.start_time > stim.duration)
            expired.push_back(key);
    }
    for (const auto& k : expired)
        active_stimuli_.erase(k);
}

// ===========================================================================
// step — Exécute un pas de simulation du réseau
//
//   Ordre des opérations :
//     1. Nettoyage des stimuli expirés
//     2. Calcul du courant synaptique entrant pour chaque neurone
//        I_syn[post] = Σ (weight × type_factor × output_pre) sur toutes les synapses
//     3. Mise à jour de chaque neurone (équation de fuite + sigmoïde)
//     4. Mise à jour des traces STDP (pré/post)
//     5. Le signal DAN et la plasticité sont gérés par apply_dan_signal()
// ===========================================================================
void BrainNetwork::step(float dt) {
    update_stimuli(dt);

    // Facteurs multiplicatifs par type de synapse
    // A_D=1.0, A_A=0.7, D_D=0.5, D_A=0.4
    static const float type_factor[] = {1.0f, 0.7f, 0.5f, 0.4f};

    // ---- 1. Calcul du courant synaptique pour tous les neurones ----
    //   On itère linéairement sur toutes les synapses (≈548k).
    //   C'est l'opération la plus coûteuse du pas de simulation.
    std::vector<float> I_syn(neurons_.size(), 0.0f);
    for (const auto& s : synapses_) {
        I_syn[s.post_id] += s.weight
                          * type_factor[(int)s.type]
                          * neurons_[s.pre_id].output;
    }

    // ---- 2. Mise à jour de chaque neurone ----
    //   Équation : V += (-(V - V_rest)/τ + I_syn) × dt
    //   Si le neurone est en période réfractaire, V est forcé à 0
    //   et la sortie est presque nulle.
    for (auto& n : neurons_) {
        if (n.refractory_timer > 0.0f) {
            // Période réfractaire : le neurone ne répond pas
            n.refractory_timer -= dt;
            n.V = V_REST * 0.5f;
            n.output = sigmoid(n.V);
            n.is_active = false;
            continue;
        }

        // Intégration temporelle (fuite + entrée synaptique)
        float dV = (-(n.V - V_REST) / TAU_MEMBRANE + I_syn[n.id]) * dt;
        n.V += dV;
        n.V = std::clamp(n.V, -2.0f, 2.0f);
        n.output = sigmoid(n.V);

        // Détection d'activité (V > seuil + marge)
        if (n.V > V_THRESHOLD + 0.3f) {
            n.refractory_timer = TAU_REFRACTORY;
            n.is_active = true;
        } else {
            n.is_active = false;
        }
    }

    // ---- 3. Mise à jour des traces STDP ----
    //   Chaque trace décroît exponentiellement avec τ = STDP_WINDOW.
    //   Quand le neurone pré (resp. post) est actif, sa trace augmente.
    //   Ces traces servent à mesurer la corrélation pré-post pour STDP.
    for (auto& s : synapses_) {
        if (s.plastic) {
            s.pre_trace *= std::exp(-dt / STDP_WINDOW);
            s.post_trace *= std::exp(-dt / STDP_WINDOW);
            if (neurons_[s.pre_id].is_active) s.pre_trace += 1.0f;
            if (neurons_[s.post_id].is_active) s.post_trace += 1.0f;
        }
    }

    time_ += dt;
    current_step_++;
}

// ===========================================================================
// apply_dan_signal — Applique un signal dopaminergique (STDP modulé)
//
//   Règle des 3 facteurs :
//     Δw = η × DAN × (pre_trace × post_trace)
//
//   - DAN > 0 (récompense) :
//       Les synapses dont les traces pré et post sont élevées sont renforcées
//       (potentiation hebbienne). La corrélation pré-post est positive.
//
//   - DAN < 0 (punition) :
//       Les mêmes synapses sont affaiblies (×0.5 pour asymétrie biologique).
//       La dépression est plus faible que la potentiation.
//
//   - DAN ≈ 0 (aucun signal) :
//       Oubli progressif : le poids tend lentement vers sa valeur initiale.
//
//   Le poids est clampé entre 0.0 et MAX_WEIGHT (5×WEIGHT_SCALE = 0.5).
//
//   NOTE : Cette méthode itère sur TOUTES les synapses plastiques (≈400k).
//   C'est volontairement coûteux car c'est le mécanisme d'apprentissage
//   principal. En C++, l'opération prend ~1-2ms, ce qui est acceptable
//   car elle n'est appelée que lors d'événements significatifs.
// ===========================================================================
void BrainNetwork::apply_dan_signal(float signal, float dt) {
    // Ignorer les signaux trop faibles (bruit de fond)
    if (std::abs(signal) < 0.0001f) return;

    for (auto& s : synapses_) {
        if (!s.plastic) continue;

        // Corrélation pré-post (mesure de l'activité simultanée)
        float correlation = s.pre_trace * s.post_trace;

        float delta;
        if (signal > 0.0f) {
            // Récompense : potentiation des synapses corrélées
            //   Δw = η × DAN × (pre_trace × post_trace)
            delta = LEARNING_RATE * signal * correlation;
        } else if (signal < 0.0f) {
            // Punition : dépression des synapses corrélées
            //   Facteur 0.5 pour asymétrie biologique
            delta = LEARNING_RATE * signal * correlation * 0.5f;
        } else {
            // Oubli progressif (pas de signal)
            delta = -0.001f * (s.weight - s.weight_initial) * dt;
        }

        // Application de la mise à jour avec clamping
        s.weight += delta;
        s.weight = std::clamp(s.weight, 0.0f, MAX_WEIGHT);
    }
}

// ===========================================================================
// get_region_activity — Calcule l'activité moyenne par région cérébrale
// ===========================================================================
RegionActivity BrainNetwork::get_region_activity() const {
    RegionActivity ra;
    auto avg = [&](NeuronType t) -> float {
        auto it = type_indices_.find(t);
        if (it == type_indices_.end() || it->second.empty()) return 0.0f;
        float sum = 0.0f;
        for (int i : it->second) sum += neurons_[i].output;
        return sum / (float)it->second.size();
    };

    ra.sensory = (avg(NeuronType::ORN) + avg(NeuronType::GRN)
                + avg(NeuronType::PR) + avg(NeuronType::THERMO)
                + avg(NeuronType::MECHANO) + avg(NeuronType::PROPRIO)
                + avg(NeuronType::AN)) / 7.0f;
    ra.PN = avg(NeuronType::PN);
    ra.KC = avg(NeuronType::KC);
    ra.MBON = avg(NeuronType::MBON);
    ra.DAN = avg(NeuronType::DAN);
    ra.CN = avg(NeuronType::CN);
    ra.DN_VNC = avg(NeuronType::DNVNC);
    ra.DN_SEZ = avg(NeuronType::DNSEZ);
    ra.RGN = avg(NeuronType::RGN);
    return ra;
}

// ===========================================================================
// get_state — Retourne l'état complet du réseau pour l'affichage
// ===========================================================================
NetworkState BrainNetwork::get_state() const {
    NetworkState s;
    s.time = time_;
    s.step = current_step_;
    s.n_active = n_active_neurons();
    s.mean_activity = mean_activity();
    s.region_act = get_region_activity();
    s.neuron_outputs.resize(neurons_.size());
    for (size_t i = 0; i < neurons_.size(); i++)
        s.neuron_outputs[i] = neurons_[i].output;
    return s;
}

// ===========================================================================
// n_active_neurons — Compte les neurones actifs (V > seuil)
// ===========================================================================
int BrainNetwork::n_active_neurons() const {
    int c = 0;
    for (auto& n : neurons_)
        if (n.is_active) c++;
    return c;
}

// ===========================================================================
// mean_activity — Moyenne des sorties de tous les neurones
// ===========================================================================
float BrainNetwork::mean_activity() const {
    float sum = 0.0f;
    for (auto& n : neurons_) sum += n.output;
    return sum / (float)neurons_.size();
}

// ===========================================================================
// Accesseurs DNVNC par groupe fonctionnel
//
//   Les 180 DNVNC sont répartis en 4 groupes qui contrôlent différentes
//   actions motrices. Chaque groupe est identifié par son compartment :
//
//     DNVNC_FWD — propulsion avant (60 neurones)
//     DNVNC_LTL — virage à droite / contraction gauche (45)
//     DNVNC_LTR — virage à gauche / contraction droite (45)
//     DNVNC_BWD — propulsion arrière (30)
//
//   La locomotion est calculée dans mainwindow.cpp à partir de ces 4
//   valeurs :
//     speed = FWD - BWD    (avance moins recule)
//     turn  = LTR - LTL    (tourne à gauche si LTR > LTL, droite sinon)
// ===========================================================================
float BrainNetwork::get_dn_vnc_forward() const {
    float sum = 0.0f; int cnt = 0;
    for (const auto& n : neurons_)
        if (n.type == NeuronType::DNVNC && n.compartment == "DNVNC_FWD")
            { sum += n.output; cnt++; }
    return cnt > 0 ? sum / (float)cnt : 0.0f;
}

float BrainNetwork::get_dn_vnc_left_turn() const {
    float sum = 0.0f; int cnt = 0;
    for (const auto& n : neurons_)
        if (n.type == NeuronType::DNVNC && n.compartment == "DNVNC_LTL")
            { sum += n.output; cnt++; }
    return cnt > 0 ? sum / (float)cnt : 0.0f;
}

float BrainNetwork::get_dn_vnc_right_turn() const {
    float sum = 0.0f; int cnt = 0;
    for (const auto& n : neurons_)
        if (n.type == NeuronType::DNVNC && n.compartment == "DNVNC_LTR")
            { sum += n.output; cnt++; }
    return cnt > 0 ? sum / (float)cnt : 0.0f;
}

float BrainNetwork::get_dn_vnc_backward() const {
    float sum = 0.0f; int cnt = 0;
    for (const auto& n : neurons_)
        if (n.type == NeuronType::DNVNC && n.compartment == "DNVNC_BWD")
            { sum += n.output; cnt++; }
    return cnt > 0 ? sum / (float)cnt : 0.0f;
}

// ===========================================================================
// reset — Réinitialise complètement le réseau avec de nouvelles valeurs
//
//   Contrairement à une simple remise à zéro, cette méthode reconstruit
//   intégralement le réseau : nouveaux poids aléatoires, nouvelles
//   connexions. C'est équivalent à redémarrer avec un nouveau cerveau.
//
//   Pour préserver la reproductibilité, le seed n'est pas modifié ;
//   l'état du générateur aléatoire au moment de l'appel détermine
//   la nouvelle configuration.
// ===========================================================================
void BrainNetwork::reset() {
    // Nettoyage complet des données
    neurons_.clear();
    synapses_.clear();
    type_indices_.clear();
    active_stimuli_.clear();
    time_ = 0.0f;
    current_step_ = 0;
    stim_counter_ = 0;
    // Reconstruction complète avec nouveaux poids aléatoires
    build_network();
}
