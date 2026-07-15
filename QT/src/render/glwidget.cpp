#include "glwidget.h"
#include "world/world_3d.h"
#include <cmath>

GLWidget::GLWidget(QWidget* parent) : QOpenGLWidget(parent) {
    setMinimumSize(400, 300);
}

GLWidget::~GLWidget() {
    makeCurrent();
    point_vao_.destroy();
    line_vao_.destroy();
    point_vbo_.destroy();
    line_vbo_.destroy();
    delete point_shader_;
    delete line_shader_;
    doneCurrent();
}

QVector3D GLWidget::toQVec(const Eigen::Vector3f& v) const {
    return QVector3D(v.x(), v.y(), v.z());
}

Eigen::Vector3f GLWidget::toEigen(const QVector3D& v) const {
    return Eigen::Vector3f(v.x(), v.y(), v.z());
}

static const char* POINT_VERT = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 uModelView;
uniform mat4 uProjection;
uniform float uPointSize;
out vec3 vColor;
void main() {
    gl_Position = uProjection * uModelView * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
    vColor = aColor;
}
)";

static const char* POINT_FRAG = R"(
#version 330 core
in vec3 vColor;
out vec4 fragColor;
void main() {
    vec2 cxy = 2.0 * gl_PointCoord - 1.0;
    float r = dot(cxy, cxy);
    if (r > 1.0) discard;
    float alpha = smoothstep(1.0, 0.0, r);
    fragColor = vec4(vColor, alpha);
}
)";

static const char* LINE_VERT = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 uModelView;
uniform mat4 uProjection;
out vec3 vColor;
void main() {
    gl_Position = uProjection * uModelView * vec4(aPos, 1.0);
    vColor = aColor;
}
)";

static const char* LINE_FRAG = R"(
#version 330 core
in vec3 vColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(vColor, 1.0);
}
)";

void GLWidget::initializeGL() {
    initializeOpenGLFunctions();
    glClearColor(0.05f, 0.07f, 0.10f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto createShader = [&](const char* vert, const char* frag) -> QOpenGLShaderProgram* {
        auto* prog = new QOpenGLShaderProgram();
        if (!prog->addShaderFromSourceCode(QOpenGLShader::Vertex, vert) ||
            !prog->addShaderFromSourceCode(QOpenGLShader::Fragment, frag)) {
            qWarning("Shader error: %s", qPrintable(prog->log()));
            delete prog;
            return nullptr;
        }
        prog->link();
        return prog;
    };

    point_shader_ = createShader(POINT_VERT, POINT_FRAG);
    line_shader_  = createShader(LINE_VERT,  LINE_FRAG);

    point_vao_.create();
    point_vbo_.create();
    point_vao_.bind();
    point_vbo_.bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    point_vao_.release();

    line_vao_.create();
    line_vbo_.create();
    line_vao_.bind();
    line_vbo_.bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    line_vao_.release();
}

void GLWidget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

void GLWidget::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float aspect = (float)width() / (float)std::max(height(), 1);
    QMatrix4x4 proj;
    proj.perspective(45.0f, aspect, 0.1f, 500.0f);

    QMatrix4x4 view;
    float cx = camera_target_.x();
    float cy = camera_target_.y();
    float cz = 10.0f;
    float x = cx + camera_dist_ * std::sin(camera_theta_) * std::cos(camera_phi_);
    float y = cy + camera_dist_ * std::cos(camera_theta_) * std::cos(camera_phi_);
    float z = cz + camera_dist_ * std::sin(camera_phi_);
    view.lookAt(QVector3D(x, y, z), QVector3D(cx, cy, cz), QVector3D(0, 0, 1));

    QMatrix4x4 model;
    model.rotate(-90.0f, 1.0f, 0.0f, 0.0f);

    QMatrix4x4 mv = view * model;

    // Draw points
    if (points_dirty_ && !point_vertices_.empty()) {
        point_vbo_.bind();
        point_vbo_.allocate(point_vertices_.data(),
                           point_vertices_.size() * sizeof(GLVertex));
        point_vbo_.release();
        points_dirty_ = false;
    }

    if (point_shader_ && !point_vertices_.empty()) {
        point_shader_->bind();
        point_shader_->setUniformValue("uModelView", mv);
        point_shader_->setUniformValue("uProjection", proj);
        point_shader_->setUniformValue("uPointSize", 8.0f);
        point_vao_.bind();
        glDrawArrays(GL_POINTS, 0, (int)point_vertices_.size());
        point_vao_.release();
        point_shader_->release();
    }

    // Draw lines
    if (lines_dirty_ && !line_vertices_.empty()) {
        line_vbo_.bind();
        line_vbo_.allocate(line_vertices_.data(),
                          line_vertices_.size() * sizeof(GLVertex));
        line_vbo_.release();
        lines_dirty_ = false;
    }

    if (line_shader_ && !line_vertices_.empty()) {
        line_shader_->bind();
        line_shader_->setUniformValue("uModelView", mv);
        line_shader_->setUniformValue("uProjection", proj);
        line_vao_.bind();
        glDrawArrays(GL_LINE_STRIP, 0, (int)line_vertices_.size());
        line_vao_.release();
        line_shader_->release();
    }
}

void GLWidget::updateWorld(const VirtualWorld3D& world) {
    point_vertices_.clear();
    line_vertices_.clear();

    auto addPoint = [&](const Eigen::Vector3f& pos, float r, float g, float b) {
        GLVertex v;
        v.x = pos.x() - 25.0f;
        v.y = pos.y() - 25.0f;
        v.z = pos.z() - 10.0f;
        v.r = r; v.g = g; v.b = b;
        point_vertices_.push_back(v);
    };

    // Larva
    addPoint(world.insect_pos(), 0.0f, 1.0f, 1.0f);

    // Odor sources
    for (const auto& odor : world.odor_sources()) {
        float r = 0, g = 0, b = 0;
        switch (odor.type) {
            case 0: r=0; g=1; b=0; break;     // attractive = green
            case 1: r=1; g=0.5; b=0; break;   // aversive = orange
            default: r=0.5; g=0.5; b=0.5; break; // neutral = gray
        }
        addPoint(odor.pos, r, g, b);
    }

    // Food sources
    for (const auto& food : world.food_sources()) {
        if (food.consumed)
            addPoint(food.pos, 0.3f, 0.3f, 0.3f);
        else
            addPoint(food.pos, 1.0f, 0.84f, 0.0f);
    }

    // Threat zones
    for (const auto& threat : world.threat_zones())
        addPoint(threat.pos, 1.0f, 0.0f, 0.0f);

    // Obstacles
    for (const auto& obs : world.obstacles())
        addPoint(obs.pos, 0.8f, 0.8f, 0.8f);

    // Trajectory line
    const auto& traj = world.trajectory();
    for (size_t i = 0; i < traj.size(); i++) {
        GLVertex v;
        v.x = traj[i].x() - 25.0f;
        v.y = traj[i].y() - 25.0f;
        v.z = traj[i].z() - 10.0f;
        float t = (float)i / (float)std::max(traj.size(), size_t(1));
        v.r = 1.0f - t;
        v.g = t;
        v.b = 1.0f;
        line_vertices_.push_back(v);
    }

    points_dirty_ = true;
    lines_dirty_ = true;

    camera_target_ = QPointF(world.insect_pos().x(), world.insect_pos().y());

    update();
}

void GLWidget::setCameraCenter(const Eigen::Vector3f& center) {
    camera_target_ = QPointF(center.x(), center.y());
}

void GLWidget::mousePressEvent(QMouseEvent* e) {
    last_mouse_pos_ = e->pos();
    if (e->button() == Qt::LeftButton) orbiting_ = true;
    mouse_dragging_ = true;
}

void GLWidget::mouseMoveEvent(QMouseEvent* e) {
    if (!mouse_dragging_) return;
    float dx = (float)(e->pos().x() - last_mouse_pos_.x()) * 0.01f;
    float dy = (float)(e->pos().y() - last_mouse_pos_.y()) * 0.01f;
    if (orbiting_) {
        camera_theta_ -= dx;
        camera_phi_ = std::clamp(camera_phi_ + dy, -1.5f, 1.5f);
    } else {
        camera_target_ += QPointF(-dx * camera_dist_ * 0.01f, dy * camera_dist_ * 0.01f);
    }
    last_mouse_pos_ = e->pos();
    update();
}

void GLWidget::mouseReleaseEvent(QMouseEvent*) {
    mouse_dragging_ = false;
    orbiting_ = false;
}

void GLWidget::wheelEvent(QWheelEvent* e) {
    camera_dist_ *= (1.0f + e->angleDelta().y() * 0.001f);
    camera_dist_ = std::clamp(camera_dist_, 10.0f, 200.0f);
    update();
}
