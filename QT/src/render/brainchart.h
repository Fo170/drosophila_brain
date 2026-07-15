#ifndef DROSOPHILA_BRAINCHART_H
#define DROSOPHILA_BRAINCHART_H

#include <QWidget>
#include <QPainter>
#include <deque>
#include <array>

class BrainChart : public QWidget {
    Q_OBJECT
public:
    explicit BrainChart(QWidget* parent = nullptr);

    void addDataPoint(float time, const std::array<float, 5>& values);
    void clear();

protected:
    void paintEvent(QPaintEvent* e) override;

private:
    struct DataPoint {
        float time;
        float sensory, KC, MBON, DAN, DN_VNC;
    };

    std::deque<DataPoint> history_;
    float max_time_ = 0.0f;
    static constexpr int MAX_POINTS = 500;

    static const QColor COLORS[5];
};

#endif
