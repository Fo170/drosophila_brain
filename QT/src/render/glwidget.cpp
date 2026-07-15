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

static const char* POINT_VERT = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 uView;
uniform mat4 uProj;
uniform float uPointSize;
out vec3 vColor;
void main() {
    gl_Position = uProj * uView * vec4(aPos, 1.0);
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
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vColor;
void main() {
    gl_Position = uProj * uView * vec4(aPos, 1.0);
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

QMatrix4x4 GLWidget::viewMatrix() const {
    QMatrix4x4 view;
    view.translate(-pan_.x(), -pan_.y(), 0.0f);
    return view;
}

void GLWidget::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float aspect = (float)width() / (float)std::max(height(), 1);
    float half_w = zoom_ * 0.5f;
    float half_h = half_w / aspect;

    QMatrix4x4 proj;
    proj.ortho(-half_w, half_w, -half_h, half_h, -100.0f, 100.0f);

    QMatrix4x4 view = viewMatrix();

    if (points_dirty_ && !point_vertices_.empty()) {
        point_vbo_.bind();
        point_vbo_.allocate(point_vertices_.data(),
                           point_vertices_.size() * sizeof(GLVertex));
        point_vbo_.release();
        points_dirty_ = false;
    }

    if (point_shader_ && !point_vertices_.empty()) {
        point_shader_->bind();
        point_shader_->setUniformValue("uView", view);
        point_shader_->setUniformValue("uProj", proj);
        point_shader_->setUniformValue("uPointSize", 10.0f);
        point_vao_.bind();
        glDrawArrays(GL_POINTS, 0, (int)point_vertices_.size());
        point_vao_.release();
        point_shader_->release();
    }

    if (lines_dirty_ && !line_vertices_.empty()) {
        line_vbo_.bind();
        line_vbo_.allocate(line_vertices_.data(),
                          line_vertices_.size() * sizeof(GLVertex));
        line_vbo_.release();
        lines_dirty_ = false;
    }

    if (line_shader_ && !line_vertices_.empty()) {
        line_shader_->bind();
        line_shader_->setUniformValue("uView", view);
        line_shader_->setUniformValue("uProj", proj);
        line_vao_.bind();
        glDrawArrays(GL_LINE_STRIP, 0, (int)line_vertices_.size());
        line_vao_.release();
        line_shader_->release();
    }

    // Draw world border
    glUseProgram(0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-half_w, half_w, -half_h, half_h, -100, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-pan_.x(), -pan_.y(), 0);

    glColor3f(0.2f, 0.2f, 0.3f);
    glBegin(GL_LINE_LOOP);
    glVertex3f(-25.0f, -25.0f, 0);
    glVertex3f( 25.0f, -25.0f, 0);
    glVertex3f( 25.0f,  25.0f, 0);
    glVertex3f(-25.0f,  25.0f, 0);
    glEnd();
}

void GLWidget::updateWorld(const VirtualWorld3D& world) {
    point_vertices_.clear();
    line_vertices_.clear();

    auto addPoint = [&](const Eigen::Vector3f& pos, float r, float g, float b) {
        GLVertex v;
        v.x = pos.x() - 25.0f;
        v.y = pos.y() - 25.0f;
        v.z = 0.0f;
        v.r = r; v.g = g; v.b = b;
        point_vertices_.push_back(v);
    };

    addPoint(world.insect_pos(), 0.0f, 1.0f, 1.0f);

    for (const auto& odor : world.odor_sources()) {
        float r, g, b;
        switch (odor.type) {
            case 0: r=0; g=1; b=0; break;
            case 1: r=1; g=0.5; b=0; break;
            default: r=0.5; g=0.5; b=0.5; break;
        }
        addPoint(odor.pos, r, g, b);
    }

    for (const auto& food : world.food_sources()) {
        addPoint(food.pos, food.consumed ? 0.3f : 1.0f, food.consumed ? 0.3f : 0.84f, food.consumed ? 0.3f : 0.0f);
    }

    for (const auto& threat : world.threat_zones())
        addPoint(threat.pos, 1.0f, 0.0f, 0.0f);

    for (const auto& obs : world.obstacles())
        addPoint(obs.pos, 0.8f, 0.8f, 0.8f);

    const auto& traj = world.trajectory();
    for (size_t i = 0; i < traj.size(); i++) {
        GLVertex v;
        v.x = traj[i].x() - 25.0f;
        v.y = traj[i].y() - 25.0f;
        v.z = 0.0f;
        v.r = 0.4f; v.g = 0.4f; v.b = 0.4f;
        line_vertices_.push_back(v);
    }

    points_dirty_ = true;
    lines_dirty_ = true;
    update();
}

void GLWidget::mousePressEvent(QMouseEvent* e) {
    last_mouse_pos_ = e->pos();
    dragging_ = true;
}

void GLWidget::mouseMoveEvent(QMouseEvent* e) {
    if (!dragging_) return;
    float aspect = (float)width() / (float)std::max(height(), 1);
    float scale = zoom_ / (float)std::max(height(), 1);
    float dx = (float)(e->pos().x() - last_mouse_pos_.x()) * scale;
    float dy = (float)(e->pos().y() - last_mouse_pos_.y()) * scale;
    pan_ -= QVector2D(dx, -dy);
    last_mouse_pos_ = e->pos();
    update();
}

void GLWidget::mouseReleaseEvent(QMouseEvent*) {
    dragging_ = false;
}

void GLWidget::wheelEvent(QWheelEvent* e) {
    zoom_ *= (1.0f - e->angleDelta().y() * 0.002f);
    zoom_ = std::clamp(zoom_, 10.0f, 200.0f);
    update();
}
