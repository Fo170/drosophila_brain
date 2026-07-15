#include "mainwindow.h"
#include "render/glwidget.h"
#include "render/brainchart.h"
#include "core/network.h"
#include "world/world_3d.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFrame>
#include <QLabel>
#include <QPixmap>
#include <QPainter>
#include <QApplication>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("Drosophila Brain Simulator");
    resize(1400, 900);

    network_ = new BrainNetwork(42);
    world_ = new VirtualWorld3D(Eigen::Vector3f(50, 50, 20), 42);

    setupUI();

    sim_timer_ = new QTimer(this);
    connect(sim_timer_, &QTimer::timeout, this, &MainWindow::simulationStep);
    sim_timer_->start(16);
}

MainWindow::~MainWindow() {
    delete network_;
    delete world_;
}

static const char* GROUP_STYLE =
    "QGroupBox { color: #58a6ff; font: bold 10px; border: 1px solid #30363d;"
    "border-radius: 4px; margin-top: 12px; padding-top: 14px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }";

void MainWindow::setupUI() {
    auto* central = new QWidget(this);
    setCentralWidget(central);
    auto* main_layout = new QHBoxLayout(central);
    main_layout->setContentsMargins(4, 4, 4, 4);
    main_layout->setSpacing(4);

    auto* left_panel = new QWidget();
    auto* left_layout = new QVBoxLayout(left_panel);
    left_layout->setContentsMargins(0, 0, 0, 0);
    left_layout->setSpacing(4);

    gl_widget_ = new GLWidget();
    brain_chart_ = new BrainChart();
    left_layout->addWidget(gl_widget_, 3);
    left_layout->addWidget(brain_chart_, 1);

    auto* right_panel = new QWidget();
    right_panel->setFixedWidth(320);
    auto* right_layout = new QVBoxLayout(right_panel);
    right_layout->setContentsMargins(0, 0, 0, 0);
    right_layout->setSpacing(6);

    right_layout->addWidget(createInfoPanel());
    right_layout->addWidget(createControlPanel());
    right_layout->addWidget(createStimuliPanel());
    right_layout->addWidget(createLegend());
    right_layout->addStretch();

    main_layout->addWidget(left_panel, 1);
    main_layout->addWidget(right_panel);
}

QWidget* MainWindow::createInfoPanel() {
    auto* group = new QGroupBox("Simulation");
    group->setStyleSheet(GROUP_STYLE);
    auto* layout = new QVBoxLayout(group);
    layout->setSpacing(3);

    auto make_row = [&](const QString& label, QLabel*& out) {
        auto* row = new QHBoxLayout();
        auto* lbl = new QLabel(label);
        lbl->setStyleSheet("color: #8b949e; font: 9px monospace;");
        out = new QLabel("—");
        out->setStyleSheet("color: #00ff88; font: 9px monospace;");
        row->addWidget(lbl);
        row->addWidget(out, 1, Qt::AlignRight);
        layout->addLayout(row);
    };

    make_row("Temps:",         time_label_);
    make_row("Position:",      pos_label_);
    make_row("Vitesse:",       speed_label_);
    make_row("Actifs:",        active_label_);
    make_row("Act. moyenne:",  mean_act_label_);
    make_row("Nourriture:",    food_label_);
    make_row("Menaces:",       threat_label_);
    make_row("Distance:",      dist_label_);
    make_row("DAN moyen:",     dan_label_);
    make_row("Événement:",     event_label_);

    return group;
}

QWidget* MainWindow::createControlPanel() {
    auto* group = new QGroupBox("Contrôles");
    group->setStyleSheet(GROUP_STYLE);
    auto* layout = new QVBoxLayout(group);
    layout->setSpacing(6);

    auto* btn_layout = new QHBoxLayout();
    pause_btn_ = new QPushButton(QString::fromUtf8("\u23F8 Pause"));
    reset_btn_ = new QPushButton(QString::fromUtf8("\u21BA Reset"));
    QString btn_style =
        "QPushButton { background: #21262d; color: white; border: 1px solid #30363d;"
        "border-radius: 4px; padding: 6px 16px; font: bold 10px; }"
        "QPushButton:hover { background: #30363d; }";
    pause_btn_->setStyleSheet(btn_style);
    reset_btn_->setStyleSheet(btn_style);
    btn_layout->addWidget(pause_btn_);
    btn_layout->addWidget(reset_btn_);
    layout->addLayout(btn_layout);

    auto* speed_layout = new QHBoxLayout();
    auto* speed_lbl = new QLabel("Vitesse:");
    speed_lbl->setStyleSheet("color: #8b949e; font: 9px monospace;");
    speed_slider_ = new QSlider(Qt::Horizontal);
    speed_slider_->setRange(1, 100);
    speed_slider_->setValue(steps_per_tick_);
    speed_slider_->setStyleSheet(
        "QSlider::groove:horizontal { height: 6px; background: #21262d; border-radius: 3px; }"
        "QSlider::handle:horizontal { background: #58a6ff; width: 14px; border-radius: 7px; margin: -4px 0; }"
        "QSlider::sub-page:horizontal { background: #58a6ff; border-radius: 3px; }");
    speed_label_val_ = new QLabel(QString::number(steps_per_tick_));
    speed_label_val_->setStyleSheet("color: #00ff88; font: 9px monospace;");
    speed_layout->addWidget(speed_lbl);
    speed_layout->addWidget(speed_slider_, 1);
    speed_layout->addWidget(speed_label_val_);
    layout->addLayout(speed_layout);

    connect(pause_btn_, &QPushButton::clicked, this, &MainWindow::togglePause);
    connect(reset_btn_, &QPushButton::clicked, this, &MainWindow::resetSimulation);
    connect(speed_slider_, &QSlider::valueChanged, this, &MainWindow::setSpeed);

    return group;
}

QWidget* MainWindow::createStimuliPanel() {
    auto* group = new QGroupBox("Stimuli");
    group->setStyleSheet(GROUP_STYLE);
    auto* layout = new QVBoxLayout(group);
    layout->setSpacing(4);

    struct { const char* name; QColor color; } bars[] = {
        {"Olfaction", QColor("#2ecc71")},
        {"Gustation", QColor("#f1c40f")},
        {"Thermique", QColor("#e74c3c")},
        {"Visuel",    QColor("#3498db")}
    };

    for (int i = 0; i < 4; i++) {
        auto* row = new QHBoxLayout();

        auto* lbl = new QLabel(bars[i].name);
        lbl->setFixedWidth(65);
        lbl->setStyleSheet("color: #8b949e; font: 9px monospace;");

        auto* bar = new QProgressBar();
        bar->setRange(0, 100);
        bar->setValue(0);
        bar->setTextVisible(false);
        bar->setFixedHeight(12);
        bar->setStyleSheet(
            QString("QProgressBar { background: #21262d; border: 1px solid #30363d;"
                    "border-radius: 3px; }"
                    "QProgressBar::chunk { background: %1; border-radius: 2px; }")
                .arg(bars[i].color.name()));

        auto* val = new QLabel("0.00");
        val->setFixedWidth(35);
        val->setStyleSheet("color: #00ff88; font: 8px monospace;");

        row->addWidget(lbl);
        row->addWidget(bar, 1);
        row->addWidget(val);
        layout->addLayout(row);

        stim_bars_[i] = {bar, val};
    }

    return group;
}

QWidget* MainWindow::createLegend() {
    auto* group = new QGroupBox("Légende");
    group->setStyleSheet(GROUP_STYLE);
    auto* layout = new QVBoxLayout(group);
    layout->setSpacing(2);

    struct { QColor color; const char* label; bool is_circle; } entries[] = {
        {QColor("#00ffff"), "Larve (position)",     true},
        {QColor("#00ff00"), "Odeur attractive",     true},
        {QColor("#ff9900"), "Odeur aversive",       true},
        {QColor("#888888"), "Odeur neutre",         true},
        {QColor("#ffd700"), "Nourriture disponible",true},
        {QColor("#ff0000"), "Zone de danger",       true},
        {QColor("#4d4d4d"), "Nourriture consumée",  true},
        {QColor("#cccccc"), "Obstacle",             true},
        {QColor("#666666"), "Trajectoire parcourue",false},
    };

    for (auto& e : entries) {
        auto* row = new QHBoxLayout();
        row->setSpacing(6);

        auto* swatch = new QLabel();
        swatch->setFixedSize(12, 12);
        if (e.is_circle) {
            QPixmap px(12, 12);
            px.fill(Qt::transparent);
            QPainter p(&px);
            p.setRenderHint(QPainter::Antialiasing);
            p.setPen(Qt::NoPen);
            p.setBrush(e.color);
            p.drawEllipse(1, 1, 10, 10);
            p.end();
            swatch->setPixmap(px);
        } else {
            QPixmap px(12, 4);
            px.fill(e.color);
            swatch->setPixmap(px);
        }

        auto* lbl = new QLabel(e.label);
        lbl->setStyleSheet("color: #c9d1d9; font: 9px monospace;");

        row->addWidget(swatch);
        row->addWidget(lbl, 1);
        layout->addLayout(row);
    }

    // Brain chart colors
    auto* sep = new QLabel("— Courbes activité cérébrale —");
    sep->setStyleSheet("color: #8b949e; font: 8px monospace; padding: 4px 0;");
    sep->setAlignment(Qt::AlignCenter);
    layout->addWidget(sep);

    struct { QColor color; const char* label; } traces[] = {
        {QColor("#ff6b6b"), "Sensoriel"},
        {QColor("#48dbfb"), "KC (MB)"},
        {QColor("#ff9ff3"), "MBON"},
        {QColor("#54a0ff"), "DAN"},
        {QColor("#1dd1a1"), "DN (moteur)"},
    };

    for (auto& t : traces) {
        auto* row = new QHBoxLayout();
        row->setSpacing(6);

        auto* swatch = new QLabel();
        swatch->setFixedSize(16, 4);
        QPixmap px(16, 4);
        px.fill(t.color);
        swatch->setPixmap(px);

        auto* lbl = new QLabel(t.label);
        lbl->setStyleSheet("color: #c9d1d9; font: 8px monospace;");

        row->addWidget(swatch);
        row->addWidget(lbl, 1);
        layout->addLayout(row);
    }

    return group;
}

void MainWindow::simulationStep() {
    if (paused_) return;

    for (int i = 0; i < steps_per_tick_; i++) {
        SensoryInput3D stim = world_->get_sensory_input_3d();

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

        network_->step(DT);

        float left_act = network_->get_dn_vnc_left_activity();
        float right_act = network_->get_dn_vnc_right_activity();
        float speed = (left_act + right_act) * 1.5f;
        float turn = (right_act - left_act) * 3.14159f * 0.5f;

        world_->move_insect_3d(speed, turn, 0.0f);

        WorldEvents events = world_->check_interactions_3d();
        if (events.reward > 0.0f) {
            dan_sum_ += events.reward;
            dan_count_++;
            event_label_->setText(QString("Nourriture! +%1").arg(events.reward, 0, 'f', 2));
        }
        if (events.punishment > 0.0f) {
            dan_sum_ -= events.punishment;
            dan_count_++;
            event_label_->setText(QString("Danger! -%1").arg(events.punishment, 0, 'f', 2));
        }

        sim_time_ += DT;
    }

    updateInfoPanel();
    gl_widget_->updateWorld(*world_);

    RegionActivity ra = network_->get_region_activity();
    brain_chart_->addDataPoint(sim_time_, {
        ra.sensory, ra.KC, ra.MBON, ra.DAN, ra.DN_VNC
    });
}

void MainWindow::updateInfoPanel() {
    NetworkState state = network_->get_state();
    auto pos = world_->insect_pos();

    time_label_->setText(QString::number(sim_time_, 'f', 1) + " ms");
    pos_label_->setText(QString("(%1, %2, %3)")
        .arg(pos.x(), 0, 'f', 1).arg(pos.y(), 0, 'f', 1).arg(pos.z(), 0, 'f', 1));
    speed_label_->setText(QString::number(world_->insect_speed(), 'f', 3));
    active_label_->setText(QString::number(state.n_active) +
        " / " + QString::number(network_->neurons().size()));
    mean_act_label_->setText(QString::number(state.mean_activity, 'f', 4));
    food_label_->setText(QString::number(world_->food_consumed()));
    threat_label_->setText(QString::number(world_->threats_encountered()));
    dist_label_->setText(QString::number(world_->distance_traveled(), 'f', 1));

    if (dan_count_ > 0) {
        dan_label_->setText(QString::number(dan_sum_ / dan_count_, 'f', 4));
    }

    SensoryInput3D stim = world_->get_sensory_input_3d();
    float vals[] = {stim.olfactory.total, stim.gustatory, stim.thermal, stim.visual};
    for (int i = 0; i < 4; i++) {
        stim_bars_[i].bar->setValue((int)(vals[i] * 100));
        stim_bars_[i].value->setText(QString::number(vals[i], 'f', 2));
    }
}

void MainWindow::togglePause() {
    paused_ = !paused_;
    pause_btn_->setText(paused_
        ? QString::fromUtf8("\u25B6 Play")
        : QString::fromUtf8("\u23F8 Pause"));
}

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

void MainWindow::setSpeed(int value) {
    steps_per_tick_ = value;
    speed_label_val_->setText(QString::number(value));
}
