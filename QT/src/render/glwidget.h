#ifndef DROSOPHILA_GLWIDGET_H
#define DROSOPHILA_GLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QMatrix4x4>
#include <QVector3D>
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
    void setCameraCenter(const Eigen::Vector3f& center);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;

private:
    void buildPointBuffer(const std::vector<GLVertex>& verts);
    void buildLineBuffer(const std::vector<GLVertex>& verts);
    QVector3D toQVec(const Eigen::Vector3f& v) const;
    Eigen::Vector3f toEigen(const QVector3D& v) const;

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

    float camera_dist_ = 70.0f;
    float camera_theta_ = 0.5f;
    float camera_phi_ = 0.3f;
    QPointF camera_target_{25, 25};
    QPoint last_mouse_pos_;
    bool mouse_dragging_ = false;
    bool orbiting_ = false;
};

#endif
