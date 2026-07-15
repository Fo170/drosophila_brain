#include "world_3d.h"
#include <algorithm>
#include <numeric>
#include <cmath>

// ===========================================================================
// Constructeur : initialise le monde et génère les entités aléatoirement
// ===========================================================================
VirtualWorld3D::VirtualWorld3D(const Eigen::Vector3f& size, int seed)
    : size_(size), rng_(seed)
{
    // Départ au centre du monde, orientation vers l'avant (axe X)
    insect_pos_ = Eigen::Vector3f(size.x() * 0.5f, size.y() * 0.5f, size.z() * 0.5f);
    insect_orientation_ = Eigen::Vector3f::UnitX();
    insect_speed_ = 0.0f;
    trajectory_.push_back(insect_pos_);

    // Génération aléatoire des entités (le seed garantit la reproductibilité)
    generate_odor_sources(8);   // 8 sources d'odeur
    generate_food_sources(5);   // 5 zones de nourriture
    generate_threat_zones(3);   // 3 zones de danger
    generate_obstacles(10);     // 10 obstacles
}

// ===========================================================================
// Génération des sources d'odeur
//   Chaque odeur a :
//     - une position aléatoire
//     - une intensité (0.5-1.0)
//     - un facteur de décroissance (0.02-0.1) : plus il est grand, plus
//       l'odeur est localisée
//     - un type : attractive (0), aversive (1) ou neutre (2)
//     - une qualité sémantique (nourriture, danger, etc.)
// ===========================================================================
void VirtualWorld3D::generate_odor_sources(int n) {
    for (int i = 0; i < n; i++) {
        OdorSource o;
        o.pos = Eigen::Vector3f(
            std::uniform_real_distribution<float>(2, size_.x() - 2)(rng_),
            std::uniform_real_distribution<float>(2, size_.y() - 2)(rng_),
            std::uniform_real_distribution<float>(2, size_.z() - 2)(rng_)
        );
        o.intensity = std::uniform_real_distribution<float>(0.5f, 1.0f)(rng_);
        o.decay = std::uniform_real_distribution<float>(0.02f, 0.1f)(rng_);
        o.type = std::uniform_int_distribution<int>(0, 2)(rng_);
        o.quality = std::uniform_int_distribution<int>(0, 3)(rng_);
        odor_sources_.push_back(o);
    }
}

// ===========================================================================
// Génération des sources de nourriture
//   Chaque source a :
//     - une position aléatoire (évite les bords)
//     - une valeur de récompense (0.7-1.0)
//     - un rayon de détection gustative (2-5 unités)
//     - une quantité qui diminue à chaque consommation jusqu'à épuisement
// ===========================================================================
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

// ===========================================================================
// Génération des zones de danger
//   Chaque zone a :
//     - une position, un rayon, une intensité de punition
//     - un type (chaleur, dessication, toxine, prédateur)
// ===========================================================================
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

// ===========================================================================
// Génération des obstacles
//   Simples boîtes de taille aléatoire placées aléatoirement
// ===========================================================================
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

// ===========================================================================
// get_sensory_input_3d()
//   Calcule les entrées sensorielles à la position actuelle :
//     - Olfaction : somme des odeurs pondérées par la distance
//     - Gustation : proximité de nourriture
//     - Thermo : proximité de zones de danger
//     - Visuel : alignement avec la lumière directionnelle
//     - Mécano : vitesse de déplacement
//     - Proprioception : orientation et vitesse
//
//   Ces données sont envoyées au réseau de neurones.
// ===========================================================================
SensoryInput3D VirtualWorld3D::get_sensory_input_3d() const {
    SensoryInput3D stim;

    // --- Olfaction : chaque source d'odeur contribue selon la distance ---
    //   L'intensité perçue décroît exponentiellement avec la distance :
    //   I_pereue = I_source * exp(-distance * decay)
    //   decay élevé = odeur localisée, decay faible = odeur qui porte loin
    for (const auto& odor : odor_sources_) {
        float dist = (insect_pos_ - odor.pos).norm();
        float intensity = odor.intensity * std::exp(-dist * odor.decay);
        stim.olfactory.total += intensity;
        if (odor.type == 0) {
            stim.olfactory.attractive += intensity;  // odeur attractive
        } else if (odor.type == 1) {
            stim.olfactory.aversive += intensity;    // odeur aversive
        }
        // type 2 (neutre) : compté dans total mais pas dans attractive/aversive
    }
    stim.olfactory.total = std::min(stim.olfactory.total, 1.0f);

    // --- Gustation : détection de la nourriture à proximité immédiate ---
    //   Plus on est proche, plus le goût est fort
    for (const auto& food : food_sources_) {
        if (!food.consumed) {
            float dist = (insect_pos_ - food.pos).norm();
            if (dist < food.radius) {
                stim.gustatory = std::max(stim.gustatory,
                    food.reward * (1.0f - dist / food.radius));
            }
        }
    }

    // --- Thermo : détection des zones de danger ---
    for (const auto& threat : threat_zones_) {
        float dist = (insect_pos_ - threat.pos).norm();
        if (dist < threat.radius) {
            stim.thermal = std::max(stim.thermal,
                threat.punish * (1.0f - dist / threat.radius));
        }
    }

    // --- Visuel : lumière directionnelle ---
    //   L'intensité lumineuse dépend de l'alignement entre l'orientation de
    //   la larve et la direction de la source lumineuse
    Eigen::Vector3f light_dir = Eigen::Vector3f(size_.x(), size_.y(), size_.z())
                                - insect_pos_;
    float light_dist = light_dir.norm();
    if (light_dist > 0.001f) {
        light_dir /= light_dist;
        float alignment = insect_orientation_.dot(light_dir);
        stim.visual = 0.3f + 0.7f * std::max(0.0f, alignment)
                      * std::exp(-light_dist * 0.01f);
    }

    // --- Mécano : la vitesse crée une sensation mécanique ---
    stim.mechanosensory = std::min(insect_speed_ / 2.0f, 1.0f);

    // --- Proprioception : orientation et vitesse ---
    stim.proprioceptive.heading = std::atan2(insect_orientation_.y(),
                                             insect_orientation_.x());
    stim.proprioceptive.pitch = std::asin(
        std::clamp(insect_orientation_.z(), -1.0f, 1.0f));
    stim.proprioceptive.speed = insect_speed_;

    return stim;
}

// ===========================================================================
// move_insect_3d()
//   Déplace la larve en fonction des commandes motrices :
//     - speed : vitesse linéaire
//     - turn_yaw : rotation horizontale 
//     - turn_pitch : rotation verticale
//
//   Gère les collisions avec :
//     - Les obstacles (rebond)
//     - Les murs du monde (rebond)
// ===========================================================================
void VirtualWorld3D::move_insect_3d(float speed, float turn_yaw,
                                     float turn_pitch) {
    // ---- Rotation horizontale (yaw, autour de Z) ----
    float cos_y = std::cos(turn_yaw);
    float sin_y = std::sin(turn_yaw);
    float new_x = insect_orientation_.x() * cos_y
                - insect_orientation_.y() * sin_y;
    float new_y = insect_orientation_.x() * sin_y
                + insect_orientation_.y() * cos_y;
    insect_orientation_.x() = new_x;
    insect_orientation_.y() = new_y;

    // ---- Rotation verticale (pitch, autour de Y) ----
    float cos_p = std::cos(turn_pitch);
    float sin_p = std::sin(turn_pitch);
    new_x = insect_orientation_.x() * cos_p
          + insect_orientation_.z() * sin_p;
    float new_z = -insect_orientation_.x() * sin_p
                 + insect_orientation_.z() * cos_p;
    insect_orientation_.x() = new_x;
    insect_orientation_.z() = new_z;

    // Normalisation du vecteur d'orientation
    float norm = insect_orientation_.norm();
    if (norm > 0.001f) insect_orientation_ /= norm;

    // ---- Déplacement ----
    insect_speed_ = speed;
    Eigen::Vector3f movement = speed * insect_orientation_;
    Eigen::Vector3f new_pos = insect_pos_ + movement;

    // ---- Collision avec les obstacles (boîtes) ----
    //   Si la nouvelle position est à l'intérieur d'un obstacle,
    //   on rebondit (orientation inversée * 0.5 pour amortir)
    for (const auto& obs : obstacles_) {
        Eigen::Vector3f obs_min = obs.pos - obs.size * 0.5f;
        Eigen::Vector3f obs_max = obs.pos + obs.size * 0.5f;
        if ((new_pos.array() > obs_min.array()).all() &&
            (new_pos.array() < obs_max.array()).all()) {
            insect_orientation_ *= -0.5f;  // rebond amorti
            new_pos = insect_pos_ + speed * insect_orientation_;
            break;
        }
    }

    // ---- Collision avec les murs du monde ----
    //   On rebondit sur le mur et on clamp la position
    for (int i = 0; i < 3; i++) {
        if (new_pos[i] < 0.0f || new_pos[i] > size_[i]) {
            insect_orientation_[i] *= -1.0f;
            new_pos[i] = std::clamp(new_pos[i], 0.0f, size_[i]);
        }
    }

    // ---- Mise à jour de l'état ----
    distance_traveled_ += (new_pos - insect_pos_).norm();
    insect_pos_ = new_pos;

    // Enregistrement de la trajectoire (limité à 10000 points)
    trajectory_.push_back(insect_pos_);
    if (trajectory_.size() > 10000) {
        trajectory_.erase(trajectory_.begin(), trajectory_.begin() + 5000);
    }
}

// ===========================================================================
// check_interactions_3d()
//   Vérifie toutes les interactions entre la larve et le monde.
//   Retourne un WorldEvents qui sera utilisé pour l'apprentissage.
//
//   Types d'interactions :
//     1. Nourriture : si la larve est suffisamment proche, elle consomme
//        une partie de la nourriture → récompense (DAN positif fort)
//     2. Danger : si la larve entre dans une zone de danger
//        → punition (DAN négatif fort)
//     3. Odeurs attractives : approche progressive → récompense continue
//     4. Odeurs aversives : approche progressive → punition continue
//     5. Obstacles : collision → punition légère
// ===========================================================================
WorldEvents VirtualWorld3D::check_interactions_3d() {
    WorldEvents events;

    // -----------------------------------------------------------------------
    // 1. CONSOMMATION DE NOURRITURE
    //    La larve consomme 0.2 unité à chaque visite.
    //    Quand la quantité atteint 0, la nourriture est épuisée (consumed).
    //    Récompense proportionnelle à la valeur de la source.
    // -----------------------------------------------------------------------
    for (auto& food : food_sources_) {
        if (!food.consumed && food.amount > 0.0f) {
            float dist = (insect_pos_ - food.pos).norm();
            if (dist < food.radius * 0.3f) {
                // Consommation partielle : 0.2 unité par visite
                float eaten = std::min(0.2f, food.amount);
                food.amount -= eaten;
                if (food.amount <= 0.0f) {
                    food.consumed = true;
                }
                // Récompense proportionnelle à ce qui a été mangé
                events.reward = food.reward * (eaten / 0.2f);
                events.food_eaten = true;
                food_consumed_++;
            }
        }
    }

    // -----------------------------------------------------------------------
    // 2. ZONES DE DANGER
    //    Punition d'autant plus forte que la larve est proche du centre.
    // -----------------------------------------------------------------------
    for (auto& threat : threat_zones_) {
        float dist = (insect_pos_ - threat.pos).norm();
        if (dist < threat.radius * 0.5f) {
            events.punishment = std::max(events.punishment,
                threat.punish * (1.0f - dist / (threat.radius * 0.5f)));
            events.threat_encountered = true;
            threats_encountered_++;
        }
    }

    // -----------------------------------------------------------------------
    // 3 & 4. ODEURS (récompense/punition continues)
    //    À chaque pas, si la larve détecte une odeur attractive, elle reçoit
    //    une petite récompense continue. Pour une odeur aversive, une petite
    //    punition continue. Cela permet un apprentissage progressif par
    //    renforcement : la larve apprend à s'approcher des bonnes odeurs
    //    et à s'éloigner des mauvaises.
    // -----------------------------------------------------------------------
    for (const auto& odor : odor_sources_) {
        float dist = (insect_pos_ - odor.pos).norm();
        float intensity = odor.intensity * std::exp(-dist * odor.decay);
        if (intensity > 0.01f) {
            if (odor.type == 0) {
                // Odeur attractive → récompense continue
                events.odor_reward += intensity * 0.1f;
            } else if (odor.type == 1) {
                // Odeur aversive → punition continue
                events.odor_punish += intensity * 0.1f;
            }
        }
    }
    events.odor_reward = std::min(events.odor_reward, 0.5f);
    events.odor_punish = std::min(events.odor_punish, 0.5f);

    // -----------------------------------------------------------------------
    // 5. COLLISION AVEC OBSTACLE
    //    La collision est détectée en comparant la position actuelle (qui est
    //    déjà la position après move_insect_3d et donc après rebond) avec les
    //    boîtes des obstacles. Si on est à l'intérieur, c'est qu'on a rebondi,
    //    donc on donne une petite punition.
    // -----------------------------------------------------------------------
    for (const auto& obs : obstacles_) {
        Eigen::Vector3f obs_min = obs.pos - obs.size * 0.5f;
        Eigen::Vector3f obs_max = obs.pos + obs.size * 0.5f;
        if ((insect_pos_.array() > obs_min.array()).all() &&
            (insect_pos_.array() < obs_max.array()).all()) {
            events.obstacle_punish = 0.1f;  // punition légère pour collision
            events.obstacle_hit = true;
            break;
        }
    }

    // -----------------------------------------------------------------------
    // Enregistrement de l'odeur dominante pour l'affichage
    // -----------------------------------------------------------------------
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
