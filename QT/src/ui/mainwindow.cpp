#include "mainwindow.h"
#include "render/glwidget.h"
#include "render/brainchart.h"
#include "core/network.h"
#include "world/world_3d.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>

// ===========================================================================
// MainWindow : Fenêtre principale de l'application
//
//   Boucle de simulation :
//     Un QTimer (16ms = ~60 FPS) déclenche simulationStep().
//     Chaque tick exécute `steps_per_tick` × un pas de simulation.
//
//   Flux complet d'un pas :
//     1. get_sensory_input_3d()   → stimuli du monde
//     2. apply_stimulus()         → active les neurones sensoriels
//     3. network->step()          → propage dans tout le réseau
//     4. move_insect_3d()         → déplace la larve selon les DN
//     5. check_interactions_3d()  → détecte les événements
//     6. apply_dan_signal()       → apprentissage STDP
//     7. updateInfoPanel()        → rafraîchit l'UI
//     8. updateWorld()            → rafraîchit la vue 3D
//     9. addDataPoint()           → met à jour le graphique
// ===========================================================================

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("Drosophila Brain Simulator");
    resize(1400, 900);
    setFixedSize(1400, 900);

    // Création du cerveau (3016 neurones, 548k synapses) et du monde 3D
    // Le seed 42 garantit la reproductibilité des expériences
    network_ = new BrainNetwork(42);
    world_ = new VirtualWorld3D(Eigen::Vector3f(50, 50, 20), 42);

    setupUI();

    // Timer de simulation : ~60 images par seconde
    // Chaque tick exécute steps_per_tick pas de simulation
    sim_timer_ = new QTimer(this);
    connect(sim_timer_, &QTimer::timeout, this, &MainWindow::simulationStep);
    sim_timer_->start(16);  // ~60 FPS (1000/16 ≈ 62.5)
}

MainWindow::~MainWindow() {
    delete network_;
    delete world_;
}

// ===========================================================================
// setupUI — Interface minimale plein écran
//
//   La fenêtre est fixe (1400×900). La vue OpenGL occupe tout l'espace,
//   avec une barre de contrôles fine en haut et le graphique en bas.
// ===========================================================================
void MainWindow::setupUI() {
    auto* central = new QWidget(this);
    setCentralWidget(central);
    auto* main_layout = new QVBoxLayout(central);
    main_layout->setContentsMargins(0, 0, 0, 0);
    main_layout->setSpacing(0);

    // Barre de contrôles (haute de 28px, fond sombre)
    auto* ctrl_bar = new QWidget();
    ctrl_bar->setFixedHeight(28);
    ctrl_bar->setStyleSheet(
        "background:#161b22; border-bottom:1px solid #30363d;");
    auto* ctrl_layout = new QHBoxLayout(ctrl_bar);
    ctrl_layout->setContentsMargins(8, 0, 8, 0);

    QString btn_st =
        "QPushButton{background:#21262d;color:#c9d1d9;border:1px solid "
        "#30363d;border-radius:3px;padding:2px 10px;font:9px monospace;}"
        "QPushButton:hover{background:#30363d;}";

    pause_btn_ = new QPushButton(QString::fromUtf8("\u23F8"));
    pause_btn_->setFixedSize(28, 22);
    pause_btn_->setStyleSheet(btn_st);
    ctrl_layout->addWidget(pause_btn_);

    reset_btn_ = new QPushButton(QString::fromUtf8("\u21BA"));
    reset_btn_->setFixedSize(28, 22);
    reset_btn_->setStyleSheet(btn_st);
    ctrl_layout->addWidget(reset_btn_);

    ctrl_layout->addSpacing(12);

    auto* spd_lbl = new QLabel("V:");
    spd_lbl->setStyleSheet("color:#8b949e;font:9px monospace;");
    ctrl_layout->addWidget(spd_lbl);

    speed_slider_ = new QSlider(Qt::Horizontal);
    speed_slider_->setRange(1, 100);
    speed_slider_->setValue(steps_per_tick_);
    speed_slider_->setFixedWidth(80);
    speed_slider_->setStyleSheet(
        "QSlider::groove:horizontal{height:4px;background:#21262d;"
        "border-radius:2px;}"
        "QSlider::handle:horizontal{background:#58a6ff;width:10px;"
        "border-radius:5px;margin:-3px 0;}"
        "QSlider::sub-page:horizontal{background:#58a6ff;border-radius:2px;}");
    ctrl_layout->addWidget(speed_slider_);

    speed_label_val_ = new QLabel(QString::number(steps_per_tick_));
    speed_label_val_->setStyleSheet("color:#00ff88;font:9px monospace;");
    ctrl_layout->addWidget(speed_label_val_);

    ctrl_layout->addStretch();

    // Infos en ligne sur la barre
    auto make_info = [&](QLabel*& out) {
        out = new QLabel("—");
        out->setStyleSheet("color:#00ff88;font:9px monospace;"
                           "padding:0 6px;");
        ctrl_layout->addWidget(out);
    };
    time_label_ = new QLabel();
    time_label_->setStyleSheet("color:#8b949e;font:9px monospace;");
    ctrl_layout->addWidget(time_label_);
    make_info(pos_label_);
    make_info(speed_label_);
    make_info(active_label_);
    make_info(event_label_);

    main_layout->addWidget(ctrl_bar);

    // Vue OpenGL (remplit tout l'espace restant)
    gl_widget_ = new GLWidget();
    main_layout->addWidget(gl_widget_, 1);

    // Graphique cérébral (80px de haut)
    brain_chart_ = new BrainChart();
    brain_chart_->setFixedHeight(80);
    main_layout->addWidget(brain_chart_);

    connect(pause_btn_, &QPushButton::clicked, this, &MainWindow::togglePause);
    connect(reset_btn_, &QPushButton::clicked, this, &MainWindow::resetSimulation);
    connect(speed_slider_, &QSlider::valueChanged, this, &MainWindow::setSpeed);
}

// ===========================================================================
// simulationStep — Boucle principale de simulation (appelée par le timer)
//
//   Cette fonction est le cœur du programme. Elle est appelée ~60 fois/s.
//   Elle exécute `steps_per_tick` pas de simulation complets.
//
//   Chaque pas effectue le cycle complet :
//     Perception → Cerveau → Action → Interaction → Apprentissage
// ===========================================================================
void MainWindow::simulationStep() {
    if (paused_) return;

    for (int i = 0; i < steps_per_tick_; i++) {
        // ==================================================================
        // PHASE 1 : PERCEPTION — La larve perçoit son environnement
        //   On calcule les entrées sensorielles à la position actuelle
        //   (odeurs, goût, température, lumière, vitesse, orientation)
        // ==================================================================
        SensoryInput3D stim = world_->get_sensory_input_3d();

        // On applique les stimuli au réseau de neurones si l'intensité
        // dépasse le seuil de détection (0.1 pour la plupart, 0.3 pour la vue)
        if (stim.olfactory.total > 0.1f)
            network_->apply_stimulus("olfactory", stim.olfactory.total, 10.0f);
        if (stim.gustatory > 0.1f)
            network_->apply_stimulus("gustatory", stim.gustatory, 10.0f);
        if (stim.thermal > 0.1f)
            network_->apply_stimulus("thermal", stim.thermal, 10.0f);
        if (stim.visual > 0.3f)
            network_->apply_stimulus("visual", stim.visual, 10.0f);
        if (stim.mechanosensory > 0.1f)
            network_->apply_stimulus("mechano", stim.mechanosensory, 10.0f);

        // ==================================================================
        // PHASE 2 : TRAITEMENT CÉRÉBRAL — Le réseau de neurones propage
        //   l'information des entrées sensorielles vers les sorties motrices.
        //   3016 neurones, 548k synapses mis à jour.
        // ==================================================================
        network_->step(DT);

        // ==================================================================
        // PHASE 3 : COMMANDE MOTRICE — 4 groupes fonctionnels de DNVNC
        //
        //   Les 180 DNVNC sont répartis en 4 groupes contrôlant chacun
        //   un muscle ou une action différente (comme dans la réalité
        //   biologique où chaque motoneurone innerve un muscle spécifique).
        //
        //     forward  (60) : propulsion avant (péristaltisme)
        //     left-turn (45) : contraction des muscles du côté gauche
        //                      → la larve tourne à DROITE
        //     right-turn(45) : contraction des muscles du côté droit
        //                      → la larve tourne à GAUCHE
        //     backward (30) : propulsion arrière
        //
        //   vitesse = forward_act - backward_act
        //     positive → avance, négative → recule
        //
        //   virage = right_turn_act - left_turn_act
        //     positif → tourne à gauche, négatif → tourne à droite
        //
        //   Cette organisation permet à la larve de se déplacer dans
        //   toutes les directions, chose impossible avec le simple split
        //   gauche/droite où l'asymétrie initiale des poids aléatoires
        //   créait un biais de virage systématique.
        // ==================================================================
        float fwd_act = network_->get_dn_vnc_forward();
        float ltl_act = network_->get_dn_vnc_left_turn();
        float ltr_act = network_->get_dn_vnc_right_turn();
        float bwd_act = network_->get_dn_vnc_backward();

        float speed = (fwd_act - bwd_act) * 2.0f;
        float turn = (ltr_act - ltl_act) * 1.5f;

        world_->move_insect_3d(speed, turn, 0.0f);

        // ==================================================================
        // PHASE 4 : INTERACTIONS — Vérification des événements
        //
        //   check_interactions_3d() détecte :
        //     - Consommation de nourriture (reward)
        //     - Entrée dans zone de danger (punishment)
        //     - Proximité d'odeurs attractives (odor_reward)
        //     - Proximité d'odeurs aversives (odor_punish)
        //     - Collision avec obstacle (obstacle_punish)
        // ==================================================================
        WorldEvents events = world_->check_interactions_3d();

        // ==================================================================
        // PHASE 5 : APPRENTISSAGE — Signaux DAN
        //
        //   Chaque événement génère un signal dopaminergique (DAN) qui
        //   modifie les poids synaptiques via STDP :
        //
        //   Récompenses fortes (événements rares) :
        //     - Manger de la nourriture  → DAN = + reward × 1.0
        //     - Danger                   → DAN = - punish × 0.5
        //
        //   Récompenses faibles et continues (chaque pas) :
        //     - Odeur attractive         → DAN = + odor_reward × 0.5
        //     - Odeur aversive           → DAN = - odor_punish × 0.5
        //     - Collision obstacle       → DAN = - obstacle_punish × 2.0
        //
        //   Le signal DAN est envoyé au réseau via apply_dan_signal()
        //   qui met à jour tous les poids synaptiques plastiques.
        // ==================================================================
        float dan_signal = 0.0f;

        // Événements forts (récompense/punition explicite)
        if (events.reward > 0.0f) {
            // Manger de la nourriture → forte récompense
            dan_signal += events.reward * 1.0f;
            event_label_->setText(
                QString("Nourriture! +%1").arg(events.reward, 0, 'f', 2));
        }
        if (events.punishment > 0.0f) {
            // Zone de danger → forte punition
            dan_signal -= events.punishment * 0.5f;
            event_label_->setText(
                QString("Danger! -%1").arg(events.punishment, 0, 'f', 2));
        }

        // Signaux continus (odeurs à proximité)
        if (events.odor_reward > 0.0f) {
            // Odeur attractive → petite récompense continue
            dan_signal += events.odor_reward * 0.5f;
        }
        if (events.odor_punish > 0.0f) {
            // Odeur aversive → petite punition continue
            dan_signal -= events.odor_punish * 0.5f;
        }

        // Collision avec obstacle → punition
        if (events.obstacle_punish > 0.0f) {
            dan_signal -= events.obstacle_punish * 2.0f;
            event_label_->setText("Obstacle! -0.1");
        }

        // Application du signal DAN (STDP sur toutes les synapses plastiques)
        if (std::abs(dan_signal) > 0.001f) {
            network_->apply_dan_signal(dan_signal, DT);
            dan_sum_ += dan_signal;
            dan_count_++;
        }

        sim_time_ += DT;
    }

    // ======================================================================
    // PHASE 6 : MISE À JOUR DE L'AFFICHAGE — Rafraîchit l'interface
    // ======================================================================

    // Panneau d'informations
    updateInfoPanel();

    // Vue 3D (mise à jour des buffers GPU)
    gl_widget_->updateWorld(*world_);

    // Graphique d'activité cérébrale
    RegionActivity ra = network_->get_region_activity();
    brain_chart_->addDataPoint(sim_time_, {
        ra.sensory, ra.KC, ra.MBON, ra.DAN, ra.DN_VNC
    });
}

// ===========================================================================
// updateInfoPanel — Met à jour tous les labels du panneau d'information
// ===========================================================================
void MainWindow::updateInfoPanel() {
    NetworkState state = network_->get_state();
    auto pos = world_->insect_pos();

    time_label_->setText(QString::number(sim_time_, 'f', 1) + "s");
    QString pos_str = QString("(%1,%2)").arg(pos.x(),0,'f',1).arg(pos.y(),0,'f',1);
    pos_label_->setText(pos_str);
    speed_label_->setText(QString::number(world_->insect_speed(), 'f', 2));
    active_label_->setText(QString::number(state.n_active));
}

// ===========================================================================
// togglePause — Met en pause ou reprend la simulation
// ===========================================================================
void MainWindow::togglePause() {
    paused_ = !paused_;
    pause_btn_->setText(paused_
        ? QString::fromUtf8("\u25B6 Play")
        : QString::fromUtf8("\u23F8 Pause"));
}

// ===========================================================================
// resetSimulation — Réinitialise complètement la simulation
//
//   Remet à zéro :
//     - Le réseau de neurones (poids = valeurs initiales)
//     - Le monde (nouvelles positions aléatoires pour les entités)
//     - Le chronomètre et les statistiques
//     - Le graphique d'activité
// ===========================================================================
void MainWindow::resetSimulation() {
    network_->reset();
    delete world_;
    world_ = new VirtualWorld3D(Eigen::Vector3f(50, 50, 20), 42);
    brain_chart_->clear();
    sim_time_ = 0.0f;
    dan_sum_ = 0.0f;
    dan_count_ = 0;
    event_label_->setText("—");
    paused_ = false;
    pause_btn_->setText(QString::fromUtf8("\u23F8 Pause"));
}

// ===========================================================================
// setSpeed — Ajuste le nombre de pas de simulation par tick
//   Valeur élevée = simulation plus rapide
// ===========================================================================
void MainWindow::setSpeed(int value) {
    steps_per_tick_ = value;
    speed_label_val_->setText(QString::number(value));
}
