#ifndef DROSOPHILA_WORLD_3D_H
#define DROSOPHILA_WORLD_3D_H

/*
 * world_3d.h — Monde virtuel 2D/3D pour la larve de Drosophila
 *
 * Contient toutes les entités que la larve peut percevoir
 * et avec lesquelles elle peut interagir :
 *   - Sources d'odeur (attractives = nourriture, aversives = danger, neutres)
 *   - Zones de nourriture (consommables, récompense)
 *   - Zones de danger (punition thermique/chimique)
 *   - Obstacles (collision = rebond + punition)
 *   - Lumière directionnelle (phototaxie)
 */

#include <vector>
#include <string>
#include <random>
#include <Eigen/Core>

// ---------------------------------------------------------------------------
// Source d'odeur : émet une intensité qui décroît avec la distance
//   type 0 = attractive (récompense progressive à l'approche)
//   type 1 = aversive   (punition progressive à l'approche)
//   type 2 = neutre     (ni récompense ni punition)
// ---------------------------------------------------------------------------
struct OdorSource {
    Eigen::Vector3f pos;        // Position dans le monde
    float intensity;            // Intensité maximale (0-1)
    float decay;                // Facteur de décroissance exponentielle
    int type;                   // 0=attractive, 1=aversive, 2=neutral
    int quality;                // 0=food, 1=danger, 2=mate, 3=home
};

// ---------------------------------------------------------------------------
// Source de nourriture : zone avec récompense gustative
//   L'insecte peut la consommer progressivement (amount diminue)
//   Quand amount <= 0, elle est marquée consumed
// ---------------------------------------------------------------------------
struct FoodSource {
    Eigen::Vector3f pos;        // Position
    float reward;               // Valeur de récompense (0.7-1.0)
    float radius;               // Rayon de détection gustative
    int nutrient_type;          // 0=sugar, 1=yeast, 2=protein
    bool consumed = false;      // True quand la nourriture est épuisée
    float amount;               // Quantité restante (0.5-1.0, décrémente à chaque visite)
};

// ---------------------------------------------------------------------------
// Zone de danger : punition thermique/chimique
//   Plus l'insecte s'approche du centre, plus la punition est forte
// ---------------------------------------------------------------------------
struct ThreatZone {
    Eigen::Vector3f pos;        // Position
    float punish;               // Valeur de punition (0.5-1.0)
    float radius;               // Rayon de la zone de danger
    int type;                   // 0=heat, 1=desiccation, 2=toxin, 3=predator
    float intensity;            // Intensité (0.3-1.0)
};

// ---------------------------------------------------------------------------
// Obstacle : barrière physique
//   L'insecte rebondit dessus et reçoit une petite punition
// ---------------------------------------------------------------------------
struct Obstacle3D {
    Eigen::Vector3f pos;        // Position center
    Eigen::Vector3f size;       // Taille (demi-dimensions)
    int type = 0;
};

// ---------------------------------------------------------------------------
// Entrées sensorielles calculées à chaque pas
//   C'est ce que perçoit la larve dans son environnement immédiat.
//   Ces valeurs sont envoyées au réseau de neurones comme stimuli.
// ---------------------------------------------------------------------------
struct SensoryInput3D {
    struct Olfactory {
        float total = 0.0f;         // Somme de toutes les odeurs (0-1)
        float attractive = 0.0f;    // Intensité des odeurs attractives
        float aversive = 0.0f;      // Intensité des odeurs aversives
    } olfactory;
    float gustatory = 0.0f;         // Goût (nourriture à proximité)
    float visual = 0.0f;            // Lumière directionnelle
    float thermal = 0.0f;           // Chaleur (danger)
    float mechanosensory = 0.0f;    // Sensation mécanique (vitesse)
    struct Proprioceptive {
        float heading = 0.0f;       // Direction (radians)
        float pitch = 0.0f;         // Inclinaison (radians)
        float speed = 0.0f;         // Vitesse
    } proprioceptive;
};

// ---------------------------------------------------------------------------
// Événements qui déclenchent l'apprentissage
//   reward / punishment    → signaux DAN forts (manger, danger)
//   odor_reward/punish     → signaux DAN faibles et continus (approche d'odeur)
//   obstacle_punish        → signaux DAN faibles (collision)
// ---------------------------------------------------------------------------
struct WorldEvents {
    float reward = 0.0f;            // Récompense pour nourriture mangée
    float punishment = 0.0f;        // Punition pour zone de danger
    float odor_reward = 0.0f;       // Récompense continue (odeur attractive)
    float odor_punish = 0.0f;       // Punition continue (odeur aversive)
    float obstacle_punish = 0.0f;   // Punition pour collision obstacle
    bool food_eaten = false;        // True si nourriture consommée ce pas
    bool threat_encountered = false; // True si danger rencontré
    bool obstacle_hit = false;      // True si obstacle touché
    std::string odor_detected;      // Type d'odeur dominante à proximité
};

// ===========================================================================
// VirtualWorld3D — Monde virtuel contenant toutes les entités et la larve
// ===========================================================================
class VirtualWorld3D {
public:
    VirtualWorld3D(const Eigen::Vector3f& size = Eigen::Vector3f(50, 50, 20),
                   int seed = 42);

    // Calcule les entrées sensorielles à la position actuelle de la larve
    SensoryInput3D get_sensory_input_3d() const;

    // Déplace la larve en fonction des commandes motrices
    void move_insect_3d(float speed, float turn_yaw, float turn_pitch);

    // Vérifie les interactions et retourne les événements d'apprentissage
    WorldEvents check_interactions_3d();

    // Accesseurs
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

    Eigen::Vector3f size_;               // Taille du monde (x, y, z)
    Eigen::Vector3f insect_pos_;         // Position actuelle de la larve
    Eigen::Vector3f insect_orientation_; // Direction du regard
    float insect_speed_ = 0.0f;          // Vitesse actuelle

    std::vector<OdorSource> odor_sources_;
    std::vector<FoodSource> food_sources_;
    std::vector<ThreatZone> threat_zones_;
    std::vector<Obstacle3D> obstacles_;
    std::vector<Eigen::Vector3f> trajectory_;  // Historique des positions

    int food_consumed_ = 0;             // Compteur de nourriture mangée
    int threats_encountered_ = 0;        // Compteur de dangers rencontrés
    float distance_traveled_ = 0.0f;     // Distance totale parcourue

    mutable std::mt19937 rng_;          // Générateur aléatoire
};

#endif
