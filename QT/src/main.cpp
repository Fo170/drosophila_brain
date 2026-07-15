#include <QApplication>
#include <QSurfaceFormat>
#include "ui/mainwindow.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("Drosophila Brain Simulator");
    app.setApplicationVersion("1.0");

    QSurfaceFormat fmt;
    fmt.setDepthBufferSize(24);
    fmt.setSamples(4);
    fmt.setSwapInterval(1);
    QSurfaceFormat::setDefaultFormat(fmt);

    MainWindow w;
    w.show();

    return app.exec();
}
