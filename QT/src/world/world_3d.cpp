#include "world_3d.h"
#include <algorithm>
#include <numeric>
#include <cmath>

VirtualWorld3D::VirtualWorld3D(const Eigen::Vector3f& size, int seed)
    : size_(size), rng_(seed) {
    insect_pos_ = Eigen::Vector3f(size.x() * 0.5f, size.y() * 0.5f, size.z() * 0.5f);
    insect_orientation_ = Eigen::Vector3f::UnitX();
    insect_speed_ = 0.0f;
    trajectory_.push_back(insect_pos_);
    generate_odor_sources(8);
    generate_food_sources(5);
    generate_threat_zones(3);
    generate_obstacles(10);
}

void VirtualWorld3D::generate_odor_sources(int n) {
    std::uniform_real_distribution<float> pos_dist(2.0f, 0.0f);
    std::uniform_real_distribution<float> f_dist(0.5f, 1.0f);
    std::uniform_real_distribution<float> decay_dist(0.02f, 0.1f);
    std::uniform_int_distribution<int> type_dist(0, 2);
    std::uniform_int_distribution<int> qual_dist(0, 3);

    for (int i = 0; i < n; i++) {
        OdorSource o;
        o.pos = Eigen::Vector3f(
            std::uniform_real_distribution<float>(2, size_.x() - 2)(rng_),
            std::uniform_real_distribution<float>(2, size_.y() - 2)(rng_),
            std::uniform_real_distribution<float>(2, size_.z() - 2)(rng_)
        );
        o.intensity = f_dist(rng_);
        o.decay = decay_dist(rng_);
        o.type = type_dist(rng_);
        o.quality = qual_dist(rng_);
        odor_sources_.push_back(o);
    }
}

void VirtualWorld3D::generate_food_sources(int n) {
    for (int i = 0; i < n; i++) {
        FoodSource f;
        f.pos = Eigen::Vector3f(
            std::uniform_real_distribution<float>(5, size_.x() - 5)(rng_),
            std::uniform_real_distribution<float>(5, size_.y() - 5)(rng_),
            std::uniform_real_distribution<float>(2, size_.z() - 2)(rng_)
        );
        f.reward = std::uniform_real_distribution<float>(0.7f, 1.0f)(rng_);
        f.radius = std::uniform_real_distribution<float>(2.0f, 5.0f)(rng_);
        f.nutrient_type = std::uniform_int_distribution<int>(0, 2)(rng_);
        f.consumed = false;
        f.amount = std::uniform_real_distribution<float>(0.5f, 1.0f)(rng_);
        food_sources_.push_back(f);
    }
}

void VirtualWorld3D::generate_threat_zones(int n) {
    for (int i = 0; i < n; i++) {
        ThreatZone t;
        t.pos = Eigen::Vector3f(
            std::uniform_real_distribution<float>(3, size_.x() - 3)(rng_),
            std::uniform_real_distribution<float>(3, size_.y() - 3)(rng_),
            std::uniform_real_distribution<float>(1, size_.z() - 1)(rng_)
        );
        t.punish = std::uniform_real_distribution<float>(0.5f, 1.0f)(rng_);
        t.radius = std::uniform_real_distribution<float>(3.0f, 8.0f)(rng_);
        t.type = std::uniform_int_distribution<int>(0, 3)(rng_);
        t.intensity = std::uniform_real_distribution<float>(0.3f, 1.0f)(rng_);
        threat_zones_.push_back(t);
    }
}

void VirtualWorld3D::generate_obstacles(int n) {
    for (int i = 0; i < n; i++) {
        Obstacle3D o;
        o.pos = Eigen::Vector3f(
            std::uniform_real_distribution<float>(0, size_.x())(rng_),
            std::uniform_real_distribution<float>(0, size_.y())(rng_),
            std::uniform_real_distribution<float>(0, size_.z())(rng_)
        );
        o.size = Eigen::Vector3f(
            std::uniform_real_distribution<float>(1, 5)(rng_),
            std::uniform_real_distribution<float>(1, 5)(rng_),
            std::uniform_real_distribution<float>(1, 3)(rng_)
        );
        o.type = 0;
        obstacles_.push_back(o);
    }
}

SensoryInput3D VirtualWorld3D::get_sensory_input_3d() const {
    SensoryInput3D stim;

    for (const auto& odor : odor_sources_) {
        float dist = (insect_pos_ - odor.pos).norm();
        float intensity = odor.intensity * std::exp(-dist * odor.decay);
        stim.olfactory.total += intensity;
        if (odor.type == 0) stim.olfactory.attractive += intensity;
        else if (odor.type == 1) stim.olfactory.aversive += intensity;
    }
    stim.olfactory.total = std::min(stim.olfactory.total, 1.0f);

    for (const auto& food : food_sources_) {
        if (!food.consumed) {
            float dist = (insect_pos_ - food.pos).norm();
            if (dist < food.radius) {
                stim.gustatory = std::max(stim.gustatory,
                    food.reward * (1.0f - dist / food.radius));
            }
        }
    }

    for (const auto& threat : threat_zones_) {
        float dist = (insect_pos_ - threat.pos).norm();
        if (dist < threat.radius) {
            stim.thermal = std::max(stim.thermal,
                threat.punish * (1.0f - dist / threat.radius));
        }
    }

    Eigen::Vector3f light_dir = Eigen::Vector3f(size_.x(), size_.y(), size_.z()) - insect_pos_;
    float light_dist = light_dir.norm();
    if (light_dist > 0.001f) {
        light_dir /= light_dist;
        float alignment = insect_orientation_.dot(light_dir);
        stim.visual = 0.3f + 0.7f * std::max(0.0f, alignment) * std::exp(-light_dist * 0.01f);
    }

    stim.mechanosensory = std::min(insect_speed_ / 2.0f, 1.0f);

    stim.proprioceptive.heading = std::atan2(insect_orientation_.y(), insect_orientation_.x());
    stim.proprioceptive.pitch = std::asin(std::clamp(insect_orientation_.z(), -1.0f, 1.0f));
    stim.proprioceptive.speed = insect_speed_;

    return stim;
}

void VirtualWorld3D::move_insect_3d(float speed, float turn_yaw, float turn_pitch) {
    float cos_y = std::cos(turn_yaw);
    float sin_y = std::sin(turn_yaw);
    float new_x = insect_orientation_.x() * cos_y - insect_orientation_.y() * sin_y;
    float new_y = insect_orientation_.x() * sin_y + insect_orientation_.y() * cos_y;
    insect_orientation_.x() = new_x;
    insect_orientation_.y() = new_y;

    float cos_p = std::cos(turn_pitch);
    float sin_p = std::sin(turn_pitch);
    new_x = insect_orientation_.x() * cos_p + insect_orientation_.z() * sin_p;
    float new_z = -insect_orientation_.x() * sin_p + insect_orientation_.z() * cos_p;
    insect_orientation_.x() = new_x;
    insect_orientation_.z() = new_z;

    float norm = insect_orientation_.norm();
    if (norm > 0.001f) insect_orientation_ /= norm;

    insect_speed_ = speed;
    Eigen::Vector3f movement = speed * insect_orientation_;
    Eigen::Vector3f new_pos = insect_pos_ + movement;

    for (const auto& obs : obstacles_) {
        Eigen::Vector3f obs_min = obs.pos - obs.size * 0.5f;
        Eigen::Vector3f obs_max = obs.pos + obs.size * 0.5f;
        if ((new_pos.array() > obs_min.array()).all() &&
            (new_pos.array() < obs_max.array()).all()) {
            insect_orientation_ *= -0.5f;
            new_pos = insect_pos_ + speed * insect_orientation_;
            break;
        }
    }

    for (int i = 0; i < 3; i++) {
        if (new_pos[i] < 0.0f || new_pos[i] > size_[i]) {
            insect_orientation_[i] *= -1.0f;
            new_pos[i] = std::clamp(new_pos[i], 0.0f, size_[i]);
        }
    }

    distance_traveled_ += (new_pos - insect_pos_).norm();
    insect_pos_ = new_pos;
    trajectory_.push_back(insect_pos_);
    if (trajectory_.size() > 10000) {
        trajectory_.erase(trajectory_.begin(), trajectory_.begin() + 5000);
    }
}

WorldEvents VirtualWorld3D::check_interactions_3d() {
    WorldEvents events;

    for (auto& food : food_sources_) {
        if (!food.consumed) {
            float dist = (insect_pos_ - food.pos).norm();
            if (dist < food.radius * 0.3f) {
                food.consumed = true;
                food.amount -= 0.2f;
                events.reward = food.reward;
                events.food_eaten = true;
                food_consumed_++;
            }
        }
    }

    for (auto& threat : threat_zones_) {
        float dist = (insect_pos_ - threat.pos).norm();
        if (dist < threat.radius * 0.5f) {
            events.punishment = std::max(events.punishment, threat.punish * 0.5f);
            events.threat_encountered = true;
            threats_encountered_++;
        }
    }

    for (const auto& odor : odor_sources_) {
        float dist = (insect_pos_ - odor.pos).norm();
        if (dist < 5.0f) {
            switch (odor.type) {
                case 0: events.odor_detected = "attractive"; break;
                case 1: events.odor_detected = "aversive"; break;
                default: events.odor_detected = "neutral"; break;
            }
            break;
        }
    }

    return events;
}
