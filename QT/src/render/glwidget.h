#ifndef DROSOPHILA_GLWIDGET_H
#define DROSOPHILA_GLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QMatrix4x4>
#include <QVector2D>
#include <QMouseEvent>
#include <QWheelEvent>
#include <vector>
#include <Eigen/Core>

class VirtualWorld3D;

struct GLVertex {
    float x, y, z;
    float r, g, b;
};

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT
public:
    explicit GLWidget(QWidget* parent = nullptr);
    ~GLWidget();

    void updateWorld(const VirtualWorld3D& world);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;

private:
    QMatrix4x4 viewMatrix() const;

    QOpenGLShaderProgram* point_shader_ = nullptr;
    QOpenGLShaderProgram* line_shader_ = nullptr;

    QOpenGLBuffer point_vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer line_vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject point_vao_;
    QOpenGLVertexArrayObject line_vao_;

    std::vector<GLVertex> point_vertices_;
    std::vector<GLVertex> line_vertices_;
    bool points_dirty_ = true;
    bool lines_dirty_ = true;

    // Orthographic camera: top-down, fixed view
    float zoom_ = 60.0f;
    QVector2D pan_{0.0f, 0.0f};
    QPoint last_mouse_pos_;
    bool dragging_ = false;
};

#endif
