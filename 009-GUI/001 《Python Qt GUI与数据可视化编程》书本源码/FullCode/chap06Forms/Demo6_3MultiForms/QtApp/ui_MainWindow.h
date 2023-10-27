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
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actWindowInsite;
    QAction *actWidgetInsite;
    QAction *actQuit;
    QAction *actWindow;
    QAction *actWidget;
    QWidget *centralWidget;
    QTabWidget *tabWidget;
    QWidget *tab;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(827, 520);
        actWindowInsite = new QAction(MainWindow);
        actWindowInsite->setObjectName(QStringLiteral("actWindowInsite"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/808.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actWindowInsite->setIcon(icon);
        actWidgetInsite = new QAction(MainWindow);
        actWidgetInsite->setObjectName(QStringLiteral("actWidgetInsite"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/430.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actWidgetInsite->setIcon(icon1);
        actQuit = new QAction(MainWindow);
        actQuit->setObjectName(QStringLiteral("actQuit"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actQuit->setIcon(icon2);
        actWindow = new QAction(MainWindow);
        actWindow->setObjectName(QStringLiteral("actWindow"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/804.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actWindow->setIcon(icon3);
        actWidget = new QAction(MainWindow);
        actWidget->setObjectName(QStringLiteral("actWidget"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/806.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actWidget->setIcon(icon4);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(70, 30, 336, 196));
        tabWidget->setTabsClosable(true);
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        tabWidget->addTab(tab, QString());
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 827, 23));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setAutoFillBackground(false);
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        mainToolBar->addAction(actWidgetInsite);
        mainToolBar->addAction(actWidget);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actWindowInsite);
        mainToolBar->addAction(actWindow);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actQuit);

        retranslateUi(MainWindow);
        QObject::connect(actQuit, SIGNAL(triggered()), MainWindow, SLOT(close()));

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Demo6_3  \345\244\232\347\252\227\345\217\243\345\272\224\347\224\250\347\250\213\345\272\217", nullptr));
        actWindowInsite->setText(QApplication::translate("MainWindow", "\345\265\214\345\205\245\345\274\217MainWindow", nullptr));
#ifndef QT_NO_TOOLTIP
        actWindowInsite->setToolTip(QApplication::translate("MainWindow", "\345\265\214\345\205\245\345\274\217MainWindow", nullptr));
#endif // QT_NO_TOOLTIP
        actWidgetInsite->setText(QApplication::translate("MainWindow", "\345\265\214\345\205\245\345\274\217Widget", nullptr));
#ifndef QT_NO_TOOLTIP
        actWidgetInsite->setToolTip(QApplication::translate("MainWindow", "Widget\345\265\214\345\205\245\345\274\217\347\252\227\344\275\223", nullptr));
#endif // QT_NO_TOOLTIP
        actQuit->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actQuit->setToolTip(QApplication::translate("MainWindow", "\351\200\200\345\207\272\346\234\254\347\263\273\347\273\237", nullptr));
#endif // QT_NO_TOOLTIP
        actWindow->setText(QApplication::translate("MainWindow", "\347\213\254\347\253\213MainWindow\347\252\227\345\217\243", nullptr));
#ifndef QT_NO_TOOLTIP
        actWindow->setToolTip(QApplication::translate("MainWindow", "\347\213\254\347\253\213MainWindow\347\252\227\345\217\243", nullptr));
#endif // QT_NO_TOOLTIP
        actWidget->setText(QApplication::translate("MainWindow", "\347\213\254\347\253\213Widget\347\252\227\345\217\243", nullptr));
#ifndef QT_NO_TOOLTIP
        actWidget->setToolTip(QApplication::translate("MainWindow", "\346\226\260\345\273\272Widget\347\213\254\347\253\213\347\252\227\345\217\243", nullptr));
#endif // QT_NO_TOOLTIP
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("MainWindow", "Page", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
