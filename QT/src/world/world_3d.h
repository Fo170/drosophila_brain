#ifndef DROSOPHILA_WORLD_3D_H
#define DROSOPHILA_WORLD_3D_H

#include <vector>
#include <string>
#include <random>
#include <Eigen/Core>

struct OdorSource {
    Eigen::Vector3f pos;
    float intensity;
    float decay;
    int type; // 0=attractive, 1=aversive, 2=neutral
    int quality; // 0=food, 1=danger, 2=mate, 3=home
};

struct FoodSource {
    Eigen::Vector3f pos;
    float reward;
    float radius;
    int nutrient_type; // 0=sugar, 1=yeast, 2=protein
    bool consumed = false;
    float amount;
};

struct ThreatZone {
    Eigen::Vector3f pos;
    float punish;
    float radius;
    int type; // 0=heat, 1=desiccation, 2=toxin, 3=predator
    float intensity;
};

struct Obstacle3D {
    Eigen::Vector3f pos;
    Eigen::Vector3f size;
    int type = 0;
};

struct SensoryInput3D {
    struct Olfactory {
        float total = 0.0f;
        float attractive = 0.0f;
        float aversive = 0.0f;
    } olfactory;
    float gustatory = 0.0f;
    float visual = 0.0f;
    float thermal = 0.0f;
    float mechanosensory = 0.0f;
    struct Proprioceptive {
        float heading = 0.0f;
        float pitch = 0.0f;
        float speed = 0.0f;
    } proprioceptive;
};

struct WorldEvents {
    float reward = 0.0f;
    float punishment = 0.0f;
    bool food_eaten = false;
    bool threat_encountered = false;
    std::string odor_detected;
};

class VirtualWorld3D {
public:
    VirtualWorld3D(const Eigen::Vector3f& size = Eigen::Vector3f(50, 50, 20), int seed = 42);

    SensoryInput3D get_sensory_input_3d() const;
    void move_insect_3d(float speed, float turn_yaw, float turn_pitch);
    WorldEvents check_interactions_3d();

    const Eigen::Vector3f& insect_pos() const { return insect_pos_; }
    const Eigen::Vector3f& insect_orientation() const { return insect_orientation_; }
    float insect_speed() const { return insect_speed_; }

    const std::vector<OdorSource>& odor_sources() const { return odor_sources_; }
    const std::vector<FoodSource>& food_sources() const { return food_sources_; }
    const std::vector<ThreatZone>& threat_zones() const { return threat_zones_; }
    const std::vector<Obstacle3D>& obstacles() const { return obstacles_; }
    const std::vector<Eigen::Vector3f>& trajectory() const { return trajectory_; }

    Eigen::Vector3f world_size() const { return size_; }

    int food_consumed() const { return food_consumed_; }
    int threats_encountered() const { return threats_encountered_; }
    float distance_traveled() const { return distance_traveled_; }

private:
    void generate_odor_sources(int n);
    void generate_food_sources(int n);
    void generate_threat_zones(int n);
    void generate_obstacles(int n);

    Eigen::Vector3f size_;
    Eigen::Vector3f insect_pos_;
    Eigen::Vector3f insect_orientation_;
    float insect_speed_ = 0.0f;

    std::vector<OdorSource> odor_sources_;
    std::vector<FoodSource> food_sources_;
    std::vector<ThreatZone> threat_zones_;
    std::vector<Obstacle3D> obstacles_;
    std::vector<Eigen::Vector3f> trajectory_;

    int food_consumed_ = 0;
    int threats_encountered_ = 0;
    float distance_traveled_ = 0.0f;

    mutable std::mt19937 rng_;
};

#endif
