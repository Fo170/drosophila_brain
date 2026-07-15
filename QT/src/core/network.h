#ifndef DROSOPHILA_NETWORK_H
#define DROSOPHILA_NETWORK_H

#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <deque>
#include <array>

enum class NeuronType {
    ORN, GRN, PR, THERMO, MECHANO, PROPRIO, AN,
    PN, KC, MBON, DAN, LHN, CN,
    DNVNC, DNSEZ, RGN, IN
};

enum class SynapseType {
    A_D, A_A, D_D, D_A
};

struct Neuron {
    int id = 0;
    NeuronType type = NeuronType::IN;
    std::string hemisphere = "left";
    std::string compartment;

    float V = 0.0f;
    float output = 0.0f;
    float refractory_timer = 0.0f;
    bool is_active = false;

    int n_presynaptic = 0;
    int n_postsynaptic = 0;
};

struct Synapse {
    int pre_id;
    int post_id;
    SynapseType type = SynapseType::A_D;
    float weight = 0.1f;
    bool plastic = true;
    float pre_trace = 0.0f;
    float post_trace = 0.0f;
};

struct RegionActivity {
    float sensory = 0.0f;
    float PN = 0.0f;
    float KC = 0.0f;
    float MBON = 0.0f;
    float DAN = 0.0f;
    float CN = 0.0f;
    float DN_VNC = 0.0f;
    float DN_SEZ = 0.0f;
    float RGN = 0.0f;
};

struct NetworkState {
    float time = 0.0f;
    int step = 0;
    int n_active = 0;
    float mean_activity = 0.0f;
    RegionActivity region_act;
    std::vector<float> neuron_outputs;
};

class BrainNetwork {
public:
    BrainNetwork(int seed = 42);

    void step(float dt);
    void apply_stimulus(const std::string& type, float intensity, float duration);
    void reset();

    RegionActivity get_region_activity() const;
    NetworkState get_state() const;

    float get_dn_vnc_left_activity() const;
    float get_dn_vnc_right_activity() const;

    const std::vector<Neuron>& neurons() const { return neurons_; }
    const std::vector<Synapse>& synapses() const { return synapses_; }

    int n_active_neurons() const;
    float mean_activity() const;

    const std::vector<int>& neurons_of_type(NeuronType t) const { return type_indices_.at(t); }

    float time() const { return time_; }

private:
    void build_network();
    void create_synapses();
    void add_synapse(int pre, int post, SynapseType type, float prob,
                     int min_n = 1, int max_n = 10);

    void update_stimuli(float dt);

    static float sigmoid(float x);

    std::vector<Neuron> neurons_;
    std::vector<Synapse> synapses_;
    std::unordered_map<NeuronType, std::vector<int>> type_indices_;

    float time_ = 0.0f;
    int current_step_ = 0;

    struct ActiveStimulus {
        std::string type;
        float intensity;
        float duration;
        float start_time;
    };
    std::unordered_map<std::string, ActiveStimulus> active_stimuli_;
    int stim_counter_ = 0;

    mutable std::mt19937 rng_;

    static constexpr float TAU_MEMBRANE = 10.0f;
    static constexpr float V_REST = 0.0f;
    static constexpr float V_THRESHOLD = 0.5f;
    static constexpr float TAU_REFRACTORY = 2.0f;
    static constexpr float SIGMOID_STEEPNESS = 10.0f;
    static constexpr float SIGMOID_OFFSET = 0.0f;

    static constexpr int N_NEURONS = 3016;
    static constexpr float STDP_WINDOW = 20.0f;
    static constexpr float WEIGHT_SCALE = 0.1f;

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

    static constexpr int N_ORN_GLOMERULI = 15;
    static constexpr int N_GRN_ZONES = 8;
    static constexpr int N_VISUAL_COLUMNS = 5;
    static constexpr int N_SOMATO_TRACTS = 3;
    static constexpr int N_KC_COMPARTMENTS = 7;
    static constexpr int N_MBON_TYPES = 15;
    static constexpr int N_DAN_CLUSTERS = 10;
};

#endif
