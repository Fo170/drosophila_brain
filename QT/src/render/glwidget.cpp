// ===========================================================================
// glwidget.cpp — Rendu OpenGL de la vue 2D top-down du simulateur
//
//   Ce module gère l'affichage graphique de la simulation via OpenGL 3.3.
//   Il dessine deux types d'éléments :
//     - POINTS : larve, odeurs, nourriture, dangers, obstacles
//     - LIGNES : trajectoire de la larve, bord du monde
//
//   La caméra est orthographique 2D (vue de dessus) avec :
//     - Pan (déplacement) par glisser-souris
//     - Zoom par molette
//     - Le monde est centré sur (0,0) avec des entités dans [-25, 25]
//
//   Architecture OpenGL :
//     - Deux programmes shader (point + ligne) avec VAO/VBO séparés
//     - Les buffers sont mis à jour via le flag dirty (points_dirty_/lines_dirty_)
//     - Les points sont dessinés avec des cercles alpha (smoothstep dans le fragment)
//     - La bordure du monde utilise le pipeline fixe (glBegin/glEnd) par simplicité
// ===========================================================================

#include "glwidget.h"
#include "world/world_3d.h"
#include <cmath>

// ===========================================================================
// Constructeur / Destructeur
// ===========================================================================

GLWidget::GLWidget(QWidget* parent) : QOpenGLWidget(parent) {
    // Taille minimale pour éviter un affichage déformé
    setMinimumSize(400, 300);
}

GLWidget::~GLWidget() {
    // Nettoyage propre des ressources OpenGL
    makeCurrent();
    point_vao_.destroy();
    line_vao_.destroy();
    point_vbo_.destroy();
    line_vbo_.destroy();
    delete point_shader_;
    delete line_shader_;
    doneCurrent();
}

// ===========================================================================
// Shaders GLSL 330 core — Programmes de rendu
//
//   POINT_VERT / POINT_FRAG : Affiche des points ronds avec anti-aliasing
//     - Le fragment shader génère un cercle parfait (alpha doux sur les bords)
//     - Les points sont projetés en orthographique (vue de dessus)
//
//   LINE_VERT / LINE_FRAG : Affiche des lignes colorées
//     - Utilisé principalement pour la trajectoire de la larve
//     - Pas de transparence sur les lignes (alpha = 1.0)
// ===========================================================================

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
    // Coordonnées du fragment dans le point [-1, 1]
    vec2 cxy = 2.0 * gl_PointCoord - 1.0;
    float r = dot(cxy, cxy);
    // On rejette les fragments en dehors du cercle
    if (r > 1.0) discard;
    // Anti-aliasing doux sur le bord du cercle
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

// ===========================================================================
// initializeGL — Configuration initiale d'OpenGL
//
//   Initialise :
//     - Fond sombre (bleu-gris très foncé)
//     - Test de profondeur (ordre d'affichage)
//     - Taille variable des points (gl_PointSize dans le shader)
//     - Mélange alpha (transparence des points)
//     - Shaders GLSL
//     - VAO/VBO pour les points et les lignes
//
//   Les attributs de vertex sont :
//     - Location 0 : vec3 position (x, y, z)
//     - Location 1 : vec3 couleur (r, g, b)
// ===========================================================================
void GLWidget::initializeGL() {
    initializeOpenGLFunctions();
    glClearColor(0.05f, 0.07f, 0.10f, 1.0f); // Fond sombre (#0d121a)
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Fonction utilitaire pour compiler et linker un programme shader
    auto createShader = [&](const char* vert, const char* frag) -> QOpenGLShaderProgram* {
        auto* prog = new QOpenGLShaderProgram();
        if (!prog->addShaderFromSourceCode(QOpenGLShader::Vertex, vert) ||
            !prog->addShaderFromSourceCode(QOpenGLShader::Fragment, frag)) {
            qWarning("Erreur shader : %s", qPrintable(prog->log()));
            delete prog;
            return nullptr;
        }
        prog->link();
        return prog;
    };

    point_shader_ = createShader(POINT_VERT, POINT_FRAG);
    line_shader_  = createShader(LINE_VERT,  LINE_FRAG);

    // Configuration du VAO/VBO pour les POINTS
    // Attributs : position (3 floats) + couleur (3 floats) = 6 floats par vertex
    point_vao_.create();
    point_vbo_.create();
    point_vao_.bind();
    point_vbo_.bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    point_vao_.release();

    // Configuration du VAO/VBO pour les LIGNES
    // Même format de vertex que les points (position + couleur)
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

// ===========================================================================
// resizeGL — Ajuste le viewport lors du redimensionnement de la fenêtre
// ===========================================================================
void GLWidget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

// ===========================================================================
// viewMatrix — Matrice de vue (pan)
//   Applique le déplacement de la caméra (pan) pour naviguer dans le monde
// ===========================================================================
QMatrix4x4 GLWidget::viewMatrix() const {
    QMatrix4x4 view;
    view.translate(-pan_.x(), -pan_.y(), 0.0f);
    return view;
}

// ===========================================================================
// paintGL — Fonction de dessin principale
//
//   À chaque frame :
//     1. Efface le buffer couleur et profondeur
//     2. Calcule la matrice de projection orthographique
//     3. Met à jour les VBO si les données ont changé (dirty flag)
//     4. Dessine les POINTS (larve, entités, obstacles)
//     5. Dessine les LIGNES (trajectoire)
//     6. Dessine le bord du monde (pipeline fixe)
// ===========================================================================
void GLWidget::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Calcul de la projection orthographique
    // Le monde fait 50×50 unités, centré sur (0,0)
    float aspect = (float)width() / (float)std::max(height(), 1);
    float half_w = zoom_ * 0.5f;
    float half_h = half_w / aspect;

    QMatrix4x4 proj;
    proj.ortho(-half_w, half_w, -half_h, half_h, -100.0f, 100.0f);

    QMatrix4x4 view = viewMatrix();

    // --- Mise à jour du buffer de points (si nécessaire) ---
    if (points_dirty_ && !point_vertices_.empty()) {
        point_vbo_.bind();
        point_vbo_.allocate(point_vertices_.data(),
                           point_vertices_.size() * sizeof(GLVertex));
        point_vbo_.release();
        points_dirty_ = false;
    }

    // --- Dessin des POINTS ---
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

    // --- Mise à jour du buffer de lignes (si nécessaire) ---
    if (lines_dirty_ && !line_vertices_.empty()) {
        line_vbo_.bind();
        line_vbo_.allocate(line_vertices_.data(),
                          line_vertices_.size() * sizeof(GLVertex));
        line_vbo_.release();
        lines_dirty_ = false;
    }

    // --- Dessin des LIGNES (trajectoire) ---
    if (line_shader_ && !line_vertices_.empty()) {
        line_shader_->bind();
        line_shader_->setUniformValue("uView", view);
        line_shader_->setUniformValue("uProj", proj);
        line_vao_.bind();
        glDrawArrays(GL_LINE_STRIP, 0, (int)line_vertices_.size());
        line_vao_.release();
        line_shader_->release();
    }

    // --- Bordure du monde (pipeline fixe) ---
    // Utilisation du pipeline immédiat (glBegin/glEnd) pour la simplicité
    // de la bordure. Le monde fait 50×50 unités (de -25 à +25).
    glUseProgram(0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-half_w, half_w, -half_h, half_h, -100, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-pan_.x(), -pan_.y(), 0);

    glColor3f(0.2f, 0.2f, 0.3f); // Gris-bleu foncé pour la bordure
    glBegin(GL_LINE_LOOP);
    glVertex3f(-25.0f, -25.0f, 0);
    glVertex3f( 25.0f, -25.0f, 0);
    glVertex3f( 25.0f,  25.0f, 0);
    glVertex3f(-25.0f,  25.0f, 0);
    glEnd();
}

// ===========================================================================
// updateWorld — Met à jour les données de rendu depuis le monde virtuel
//
//   Appelée à chaque pas de simulation pour refléter l'état courant :
//     - Position de la larve (cyan)
//     - Sources d'odeur (vert=attractif, orange=aversif, gris=neutre)
//     - Sources de nourriture (jaune=disponible, marron=consumée)
//     - Zones de danger (rouge)
//     - Obstacles (gris clair)
//     - Trajectoire (gris moyen, en lignes continues)
//
//   Les coordonnées sont décalées de -25 pour centrer le monde sur (0,0)
//   dans l'espace OpenGL.
// ===========================================================================
void GLWidget::updateWorld(const VirtualWorld3D& world) {
    point_vertices_.clear();
    line_vertices_.clear();

    // Ajoute un point coloré à la liste des vertices
    auto addPoint = [&](const Eigen::Vector3f& pos, float r, float g, float b) {
        GLVertex v;
        v.x = pos.x() - 25.0f; // Centrage du monde sur (0,0)
        v.y = pos.y() - 25.0f;
        v.z = 0.0f;
        v.r = r; v.g = g; v.b = b;
        point_vertices_.push_back(v);
    };

    // Larve : point cyan
    addPoint(world.insect_pos(), 0.0f, 1.0f, 1.0f);

    // Odeurs : vert (attractif), orange (aversif), gris (neutre)
    for (const auto& odor : world.odor_sources()) {
        float r, g, b;
        switch (odor.type) {
            case 0: r=0; g=1; b=0; break;       // Attractive → vert
            case 1: r=1; g=0.5; b=0; break;     // Aversive → orange
            default: r=0.5; g=0.5; b=0.5; break; // Neutre → gris
        }
        addPoint(odor.pos, r, g, b);
    }

    // Nourriture : jaune si disponible, marron si consumée
    for (const auto& food : world.food_sources()) {
        addPoint(food.pos,
            food.consumed ? 0.3f : 1.0f,
            food.consumed ? 0.3f : 0.84f,
            food.consumed ? 0.3f : 0.0f);
    }

    // Zones de danger : rouge
    for (const auto& threat : world.threat_zones())
        addPoint(threat.pos, 1.0f, 0.0f, 0.0f);

    // Obstacles : gris clair
    for (const auto& obs : world.obstacles())
        addPoint(obs.pos, 0.8f, 0.8f, 0.8f);

    // Trajectoire : ligne gris moyen
    const auto& traj = world.trajectory();
    for (size_t i = 0; i < traj.size(); i++) {
        GLVertex v;
        v.x = traj[i].x() - 25.0f;
        v.y = traj[i].y() - 25.0f;
        v.z = 0.0f;
        v.r = 0.4f; v.g = 0.4f; v.b = 0.4f;
        line_vertices_.push_back(v);
    }

    // Marquage dirty pour mise à jour des VBO au prochain affichage
    points_dirty_ = true;
    lines_dirty_ = true;
    update(); // Déclenche paintGL()
}

// ===========================================================================
// Gestion des entrées souris — Navigation dans la vue 2D
//
//   mousePressEvent   : Début du glisser (pan)
//   mouseMoveEvent    : Déplacement de la vue (pan) par glisser
//   mouseReleaseEvent : Fin du glisser
//   wheelEvent        : Zoom avant/arrière (molette)
// ===========================================================================

void GLWidget::mousePressEvent(QMouseEvent* e) {
    last_mouse_pos_ = e->pos();
    dragging_ = true;
}

void GLWidget::mouseMoveEvent(QMouseEvent* e) {
    if (!dragging_) return;
    // Conversion du déplacement souris en déplacement monde
    float aspect = (float)width() / (float)std::max(height(), 1);
    float scale = zoom_ / (float)std::max(height(), 1);
    float dx = (float)(e->pos().x() - last_mouse_pos_.x()) * scale;
    float dy = (float)(e->pos().y() - last_mouse_pos_.y()) * scale;
    pan_ -= QVector2D(dx, -dy); // Y inversé (souris → monde)
    last_mouse_pos_ = e->pos();
    update(); // Redessine la vue
}

void GLWidget::mouseReleaseEvent(QMouseEvent*) {
    dragging_ = false;
}

void GLWidget::wheelEvent(QWheelEvent* e) {
    // Zoom exponentiel : chaque cran de molette multiplie/divid par ~1.002
    zoom_ *= (1.0f - e->angleDelta().y() * 0.002f);
    // Limites du zoom (10 = très zoomé, 200 = vue large)
    zoom_ = std::clamp(zoom_, 10.0f, 200.0f);
    update(); // Redessine la vue
}
