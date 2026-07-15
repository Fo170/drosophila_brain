#ifndef DROSOPHILA_MAINWINDOW_H
#define DROSOPHILA_MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QWidget>

class GLWidget;
class BrainChart;
class BrainNetwork;
class VirtualWorld3D;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void simulationStep();
    void togglePause();
    void resetSimulation();
    void setSpeed(int value);

private:
    void setupUI();
    void updateInfoPanel();

    BrainNetwork* network_;
    VirtualWorld3D* world_;
    GLWidget* gl_widget_;
    BrainChart* brain_chart_;
    QTimer* sim_timer_;

    QLabel* time_label_;
    QLabel* pos_label_;
    QLabel* speed_label_;
    QLabel* active_label_;
    QLabel* event_label_;

    QPushButton* pause_btn_;
    QPushButton* reset_btn_;
    QSlider* speed_slider_;
    QLabel* speed_label_val_;

    bool paused_ = false;
    int steps_per_tick_ = 10;
    float sim_time_ = 0.0f;
    float dan_sum_ = 0.0f;
    int dan_count_ = 0;

    static constexpr float DT = 0.001f;
};

#endif
