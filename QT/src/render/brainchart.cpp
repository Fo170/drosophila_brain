#include "brainchart.h"
#include <QPainterPath>
#include <algorithm>
#include <cmath>

const QColor BrainChart::COLORS[5] = {
    QColor("#ff6b6b"),  // sensory - red
    QColor("#48dbfb"),  // KC - cyan
    QColor("#ff9ff3"),  // MBON - pink
    QColor("#54a0ff"),  // DAN - blue
    QColor("#1dd1a1")   // DN_VNC - green
};

BrainChart::BrainChart(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(150);
    setAutoFillBackground(true);
}

void BrainChart::addDataPoint(float time, const std::array<float, 5>& values) {
    DataPoint dp;
    dp.time = time;
    dp.sensory = values[0];
    dp.KC = values[1];
    dp.MBON = values[2];
    dp.DAN = values[3];
    dp.DN_VNC = values[4];

    history_.push_back(dp);
    max_time_ = std::max(max_time_, time);

    if ((int)history_.size() > MAX_POINTS)
        history_.pop_front();

    update();
}

void BrainChart::clear() {
    history_.clear();
    max_time_ = 0.0f;
    update();
}

void BrainChart::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    // Background
    p.fillRect(rect(), QColor("#161b22"));

    if (history_.empty()) return;

    int m = 10;
    int w = width() - 2 * m;
    int h = height() - 2 * m;
    if (w <= 0 || h <= 0) return;

    // Border
    p.setPen(QColor("#30363d"));
    p.drawRect(m, m, w, h);

    // Grid
    p.setPen(QPen(QColor("#21262d"), 1));
    for (int i = 0; i < 4; i++) {
        int y = m + h * (i + 1) / 4;
        p.drawLine(m, y, m + w, y);
    }

    // Time window: use min/max of actual stored data
    float t_min = history_.front().time;
    float t_max = history_.back().time;
    float t_range = std::max(t_max - t_min, 0.01f);

    auto get_x = [&](float t) -> float {
        return m + w * (t - t_min) / t_range;
    };

    auto get_y = [&](float val) -> float {
        return m + h * (1.0f - std::clamp(val, 0.0f, 1.0f));
    };

    // Draw each trace
    auto draw_trace = [&](const std::deque<DataPoint>& data,
                          auto get_val, const QColor& color) {
        p.setPen(QPen(color, 1.5));
        QPainterPath path;
        bool first = true;
        for (const auto& dp : data) {
            if (dp.time < t_min) continue;
            float x = get_x(dp.time);
            float y = get_y(get_val(dp));
            if (first) { path.moveTo(x, y); first = false; }
            else path.lineTo(x, y);
        }
        p.drawPath(path);
    };

    draw_trace(history_, [](const DataPoint& d) { return d.sensory; },  COLORS[0]);
    draw_trace(history_, [](const DataPoint& d) { return d.KC; },       COLORS[1]);
    draw_trace(history_, [](const DataPoint& d) { return d.MBON; },     COLORS[2]);
    draw_trace(history_, [](const DataPoint& d) { return d.DAN; },      COLORS[3]);
    draw_trace(history_, [](const DataPoint& d) { return d.DN_VNC; },   COLORS[4]);

    // Labels
    p.setPen(Qt::white);
    p.setFont(QFont("monospace", 8));

    QFontMetrics fm(p.font());
    int label_x = m + 5;
    int label_y = m + 15;
    const char* names[] = {"Sensory", "KC", "MBON", "DAN", "DN"};
    for (int i = 0; i < 5; i++) {
        p.setPen(COLORS[i]);
        p.drawText(label_x, label_y + i * 14, names[i]);
    }

    // Title
    p.setPen(QColor("#8b949e"));
    p.drawText(m + w - fm.horizontalAdvance("Activité cérébrale") - 5,
               m + 15, "Activité cérébrale");
}
