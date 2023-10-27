/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actQFile_Open;
    QAction *actStream_Open;
    QAction *actQuit;
    QAction *actQFile_Save;
    QAction *actStream_Save;
    QAction *actPY_Open;
    QAction *actPY_Save;
    QWidget *centralWidget;
    QPlainTextEdit *textEdit;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(752, 471);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        actQFile_Open = new QAction(MainWindow);
        actQFile_Open->setObjectName(QStringLiteral("actQFile_Open"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/122.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actQFile_Open->setIcon(icon);
        actStream_Open = new QAction(MainWindow);
        actStream_Open->setObjectName(QStringLiteral("actStream_Open"));
        actStream_Open->setIcon(icon);
        actQuit = new QAction(MainWindow);
        actQuit->setObjectName(QStringLiteral("actQuit"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actQuit->setIcon(icon1);
        actQFile_Save = new QAction(MainWindow);
        actQFile_Save->setObjectName(QStringLiteral("actQFile_Save"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/104.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actQFile_Save->setIcon(icon2);
        actStream_Save = new QAction(MainWindow);
        actStream_Save->setObjectName(QStringLiteral("actStream_Save"));
        actStream_Save->setIcon(icon2);
        actPY_Open = new QAction(MainWindow);
        actPY_Open->setObjectName(QStringLiteral("actPY_Open"));
        actPY_Open->setIcon(icon);
        actPY_Save = new QAction(MainWindow);
        actPY_Save->setObjectName(QStringLiteral("actPY_Save"));
        actPY_Save->setIcon(icon2);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        textEdit = new QPlainTextEdit(centralWidget);
        textEdit->setObjectName(QStringLiteral("textEdit"));
        textEdit->setGeometry(QRect(125, 55, 391, 241));
        QFont font1;
        font1.setPointSize(12);
        textEdit->setFont(font1);
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 752, 23));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        mainToolBar->addAction(actQFile_Open);
        mainToolBar->addAction(actQFile_Save);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actStream_Open);
        mainToolBar->addAction(actStream_Save);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actPY_Open);
        mainToolBar->addAction(actPY_Save);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actQuit);

        retranslateUi(MainWindow);
        QObject::connect(actQuit, SIGNAL(triggered()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "\346\226\207\346\234\254\346\226\207\344\273\266\350\257\273\345\206\231", nullptr));
        actQFile_Open->setText(QApplication::translate("MainWindow", "QFile\346\211\223\345\274\200", nullptr));
#ifndef QT_NO_TOOLTIP
        actQFile_Open->setToolTip(QApplication::translate("MainWindow", "\347\224\250QFile\346\211\223\345\274\200\346\226\207\346\234\254\346\226\207\344\273\266", nullptr));
#endif // QT_NO_TOOLTIP
        actStream_Open->setText(QApplication::translate("MainWindow", "Stream\346\211\223\345\274\200", nullptr));
#ifndef QT_NO_TOOLTIP
        actStream_Open->setToolTip(QApplication::translate("MainWindow", "\347\224\250QtextStream\346\211\223\345\274\200\346\226\207\346\234\254\346\226\207\344\273\266", nullptr));
#endif // QT_NO_TOOLTIP
        actQuit->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actQuit->setToolTip(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#endif // QT_NO_TOOLTIP
        actQFile_Save->setText(QApplication::translate("MainWindow", "QFile\345\217\246\345\255\230", nullptr));
#ifndef QT_NO_TOOLTIP
        actQFile_Save->setToolTip(QApplication::translate("MainWindow", "\347\224\250QFile\347\233\264\346\216\245\345\217\246\345\255\230\346\226\207\344\273\266", nullptr));
#endif // QT_NO_TOOLTIP
        actStream_Save->setText(QApplication::translate("MainWindow", "Stream\345\217\246\345\255\230", nullptr));
#ifndef QT_NO_TOOLTIP
        actStream_Save->setToolTip(QApplication::translate("MainWindow", "\347\224\250QTextStream\345\217\246\345\255\230\346\226\207\344\273\266", nullptr));
#endif // QT_NO_TOOLTIP
        actPY_Open->setText(QApplication::translate("MainWindow", "Python\346\211\223\345\274\200", nullptr));
#ifndef QT_NO_TOOLTIP
        actPY_Open->setToolTip(QApplication::translate("MainWindow", "\344\275\277\347\224\250Python\345\206\205\347\275\256\345\212\237\350\203\275\346\211\223\345\274\200", nullptr));
#endif // QT_NO_TOOLTIP
        actPY_Save->setText(QApplication::translate("MainWindow", "Python\345\217\246\345\255\230", nullptr));
#ifndef QT_NO_TOOLTIP
        actPY_Save->setToolTip(QApplication::translate("MainWindow", "\351\207\207\347\224\250Python\345\206\205\347\275\256\345\212\237\350\203\275\345\217\246\345\255\230", nullptr));
#endif // QT_NO_TOOLTIP
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
