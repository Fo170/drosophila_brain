#include "network.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>

BrainNetwork::BrainNetwork(int seed) : rng_(seed) {
    build_network();
}

float BrainNetwork::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-SIGMOID_STEEPNESS * (x - V_THRESHOLD + SIGMOID_OFFSET)));
}

void BrainNetwork::build_network() {
    neurons_.reserve(N_NEURONS);
    int id = 0;

    auto make_neurons = [&](int count, NeuronType type, const std::string& hemi,
                            const std::string& comp_prefix, int group_size) {
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

    auto make_grouped = [&](int total, NeuronType type, const std::string& prefix,
                            int n_groups) {
        make_neurons(total, type, "left", prefix + "_",
                     std::max(1, total / n_groups));
    };

    make_grouped(176, NeuronType::ORN,  "AL_glom",   N_ORN_GLOMERULI);
    make_grouped(42,  NeuronType::GRN,  "GRN_zone",  N_GRN_ZONES);
    make_grouped(29,  NeuronType::PR,   "VIS_col",   N_VISUAL_COLUMNS);
    make_neurons(8,   NeuronType::THERMO,  "left", "thermo", 0);
    make_neurons(10,  NeuronType::MECHANO, "left", "mechano", 0);
    make_neurons(12,  NeuronType::PROPRIO, "left", "proprio", 0);
    make_neurons(200, NeuronType::AN,   "left", "VNC_A1_", 50);
    make_grouped(210, NeuronType::PN,   "PN",     30);
    make_grouped(176, NeuronType::KC,   "MB_comp", N_KC_COMPARTMENTS);
    make_grouped(48,  NeuronType::MBON, "MBON_type", N_MBON_TYPES);
    make_grouped(30,  NeuronType::DAN,  "DAN_clust", N_DAN_CLUSTERS);
    make_neurons(50,  NeuronType::LHN,  "left", "LH_", 10);
    make_grouped(108, NeuronType::CN,   "CN",     20);
    make_grouped(180, NeuronType::DNVNC,"DNVNC_",  30);
    make_grouped(54,  NeuronType::DNSEZ,"DNSEZ_",  10);
    make_grouped(184, NeuronType::RGN,  "RGN_",    30);

    int remaining = N_NEURONS - (int)neurons_.size();
    if (remaining > 0) {
        make_grouped(remaining, NeuronType::IN, "IN_", 50);
    }

    assert((int)neurons_.size() == N_NEURONS);
    create_synapses();
}

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
    std::uniform_int_distribution<int> n_syn(1, 10);

    auto add_syn_lambda = [&](int pre, int post, SynapseType type, float prob,
                              int min_n = 1, int max_n = 10) {
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
            s.plastic = (type == SynapseType::A_D || type == SynapseType::D_A);
            synapses_.push_back(s);

            neurons_[pre].n_postsynaptic++;
            neurons_[post].n_presynaptic++;
        }
    };

    auto connect_groups = [&](const std::vector<int>& pre, const std::vector<int>& post,
                              SynapseType type, float prob, int mn = 1, int mx = 8) {
        for (int p : pre)
            for (int q : post)
                add_syn_lambda(p, q, type, prob, mn, mx);
    };

    auto all_sensory = [&]() -> std::vector<int> {
        std::vector<int> r;
        for (auto* v : {&orns, &grns, &prs, &therm, &mech, &prop, &ans})
            r.insert(r.end(), v->begin(), v->end());
        return r;
    };

    connect_groups(all_sensory(), pns, SynapseType::A_D, P_SENSORY_TO_PN, 1, 8);
    connect_groups(pns, kcs, SynapseType::A_D, P_PN_TO_MB, 1, 5);
    connect_groups(pns, lhns, SynapseType::A_D, P_PN_TO_LH, 1, 6);
    connect_groups(kcs, mbons, SynapseType::A_D, P_KC_TO_MBON, 1, 3);
    connect_groups(mbons, cns, SynapseType::A_D, P_MBON_TO_CN, 1, 8);
    connect_groups(lhns, cns, SynapseType::A_D, P_MBON_TO_CN * 0.8f, 1, 6);

    for (int cn : cns) {
        for (int dn : dnvnc) add_syn_lambda(cn, dn, SynapseType::A_D, P_CN_TO_OUTPUT, 1, 10);
        for (int dn : dnsez) add_syn_lambda(cn, dn, SynapseType::A_D, P_CN_TO_OUTPUT, 1, 10);
        for (int rg : rgns) add_syn_lambda(cn, rg, SynapseType::A_D, P_CN_TO_OUTPUT * 0.5f, 1, 5);
    }

    connect_groups(mbons, dans, SynapseType::A_A, P_MBON_TO_DAN, 1, 5);
    connect_groups(dans, kcs, SynapseType::D_A, P_DAN_TO_KC, 1, 4);

    for (int dn : dnvnc)
        for (int i = 0; i < 100 && i < (int)ins.size(); i++)
            add_syn_lambda(dn, ins[i], SynapseType::D_A, P_DN_FEEDBACK, 1, 3);
    for (int dn : dnsez)
        for (int i = 0; i < 100 && i < (int)ins.size(); i++)
            add_syn_lambda(dn, ins[i], SynapseType::D_A, P_DN_FEEDBACK, 1, 3);
}

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
        s.plastic = (type == SynapseType::A_D || type == SynapseType::D_A);
        synapses_.push_back(s);
    }
}

void BrainNetwork::apply_stimulus(const std::string& type, float intensity, float duration) {
    std::string key = type + "_" + std::to_string(stim_counter_++);
    active_stimuli_[key] = {type, intensity, duration, time_};

    static const std::unordered_map<std::string, NeuronType> s2t = {
        {"olfactory", NeuronType::ORN}, {"gustatory", NeuronType::GRN},
        {"visual", NeuronType::PR}, {"thermal", NeuronType::THERMO},
        {"mechano", NeuronType::MECHANO}, {"proprioceptive", NeuronType::PROPRIO}
    };

    auto it = s2t.find(type);
    if (it == s2t.end()) return;
    NeuronType nt = it->second;
    auto& idxs = type_indices_[nt];

    std::normal_distribution<float> noise(0.0f, 0.1f);
    for (int idx : idxs) {
        neurons_[idx].V = intensity + noise(rng_);
        neurons_[idx].output = sigmoid(neurons_[idx].V);
    }
}

void BrainNetwork::update_stimuli(float dt) {
    std::vector<std::string> expired;
    for (auto& [key, stim] : active_stimuli_) {
        if (time_ - stim.start_time > stim.duration)
            expired.push_back(key);
    }
    for (const auto& k : expired)
        active_stimuli_.erase(k);
}

void BrainNetwork::step(float dt) {
    update_stimuli(dt);

    static const float type_factor[] = {1.0f, 0.7f, 0.5f, 0.4f};

    std::vector<float> I_syn(neurons_.size(), 0.0f);
    for (const auto& s : synapses_) {
        I_syn[s.post_id] += s.weight * type_factor[(int)s.type] * neurons_[s.pre_id].output;
    }

    for (auto& n : neurons_) {
        if (n.refractory_timer > 0.0f) {
            n.refractory_timer -= dt;
            n.V = V_REST * 0.5f;
            n.output = sigmoid(n.V);
            n.is_active = false;
            continue;
        }

        float dV = (-(n.V - V_REST) / TAU_MEMBRANE + I_syn[n.id]) * dt;
        n.V += dV;
        n.V = std::clamp(n.V, -2.0f, 2.0f);
        n.output = sigmoid(n.V);

        if (n.V > V_THRESHOLD + 0.3f) {
            n.refractory_timer = TAU_REFRACTORY;
            n.is_active = true;
        } else {
            n.is_active = false;
        }
    }

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

RegionActivity BrainNetwork::get_region_activity() const {
    RegionActivity ra;
    auto avg = [&](NeuronType t) -> float {
        auto it = type_indices_.find(t);
        if (it == type_indices_.end() || it->second.empty()) return 0.0f;
        float sum = 0.0f;
        for (int i : it->second) sum += neurons_[i].output;
        return sum / (float)it->second.size();
    };

    ra.sensory = (avg(NeuronType::ORN) + avg(NeuronType::GRN) + avg(NeuronType::PR) +
                  avg(NeuronType::THERMO) + avg(NeuronType::MECHANO) +
                  avg(NeuronType::PROPRIO) + avg(NeuronType::AN)) / 7.0f;
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

int BrainNetwork::n_active_neurons() const {
    int c = 0;
    for (auto& n : neurons_)
        if (n.is_active) c++;
    return c;
}

float BrainNetwork::mean_activity() const {
    float sum = 0.0f;
    for (auto& n : neurons_) sum += n.output;
    return sum / (float)neurons_.size();
}

float BrainNetwork::get_dn_vnc_left_activity() const {
    auto it = type_indices_.find(NeuronType::DNVNC);
    if (it == type_indices_.end() || it->second.empty()) return 0.0f;
    int half = (int)it->second.size() / 2;
    float sum = 0.0f;
    for (int i = 0; i < half; i++) sum += neurons_[it->second[i]].output;
    return half > 0 ? sum / (float)half : 0.0f;
}

float BrainNetwork::get_dn_vnc_right_activity() const {
    auto it = type_indices_.find(NeuronType::DNVNC);
    if (it == type_indices_.end() || it->second.empty()) return 0.0f;
    int half = (int)it->second.size() / 2;
    float sum = 0.0f;
    for (int i = half; i < (int)it->second.size(); i++) sum += neurons_[it->second[i]].output;
    return (it->second.size() - half) > 0 ? sum / (float)(it->second.size() - half) : 0.0f;
}

void BrainNetwork::reset() {
    for (auto& n : neurons_) {
        n.V = 0.0f;
        n.output = 0.0f;
        n.refractory_timer = 0.0f;
        n.is_active = false;
    }
    for (auto& s : synapses_) {
        s.pre_trace = 0.0f;
        s.post_trace = 0.0f;
    }
    time_ = 0.0f;
    current_step_ = 0;
    active_stimuli_.clear();
}
