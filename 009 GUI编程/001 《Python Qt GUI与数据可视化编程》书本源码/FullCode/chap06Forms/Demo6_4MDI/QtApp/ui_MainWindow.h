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
#include <QtWidgets/QMdiArea>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actDoc_New;
    QAction *actQuit;
    QAction *actDoc_Open;
    QAction *actEdit_Font;
    QAction *actEdit_Cut;
    QAction *actEdit_Copy;
    QAction *actEdit_Paste;
    QAction *actMDI_Mode;
    QAction *actMDI_Cascade;
    QAction *actMDI_Tile;
    QAction *actDoc_CloseALL;
    QWidget *centralWidget;
    QMdiArea *mdiArea;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(598, 376);
        actDoc_New = new QAction(MainWindow);
        actDoc_New->setObjectName(QStringLiteral("actDoc_New"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/100.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actDoc_New->setIcon(icon);
        actQuit = new QAction(MainWindow);
        actQuit->setObjectName(QStringLiteral("actQuit"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actQuit->setIcon(icon1);
        actDoc_Open = new QAction(MainWindow);
        actDoc_Open->setObjectName(QStringLiteral("actDoc_Open"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/122.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actDoc_Open->setIcon(icon2);
        actEdit_Font = new QAction(MainWindow);
        actEdit_Font->setObjectName(QStringLiteral("actEdit_Font"));
        actEdit_Font->setEnabled(false);
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/506.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Font->setIcon(icon3);
        actEdit_Cut = new QAction(MainWindow);
        actEdit_Cut->setObjectName(QStringLiteral("actEdit_Cut"));
        actEdit_Cut->setEnabled(false);
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/200.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Cut->setIcon(icon4);
        actEdit_Copy = new QAction(MainWindow);
        actEdit_Copy->setObjectName(QStringLiteral("actEdit_Copy"));
        actEdit_Copy->setEnabled(false);
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/icons/images/202.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Copy->setIcon(icon5);
        actEdit_Paste = new QAction(MainWindow);
        actEdit_Paste->setObjectName(QStringLiteral("actEdit_Paste"));
        actEdit_Paste->setEnabled(false);
        QIcon icon6;
        icon6.addFile(QStringLiteral(":/icons/images/204.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Paste->setIcon(icon6);
        actMDI_Mode = new QAction(MainWindow);
        actMDI_Mode->setObjectName(QStringLiteral("actMDI_Mode"));
        actMDI_Mode->setCheckable(true);
        QIcon icon7;
        icon7.addFile(QStringLiteral(":/icons/images/230.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actMDI_Mode->setIcon(icon7);
        actMDI_Cascade = new QAction(MainWindow);
        actMDI_Cascade->setObjectName(QStringLiteral("actMDI_Cascade"));
        QIcon icon8;
        icon8.addFile(QStringLiteral(":/icons/images/400.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actMDI_Cascade->setIcon(icon8);
        actMDI_Tile = new QAction(MainWindow);
        actMDI_Tile->setObjectName(QStringLiteral("actMDI_Tile"));
        QIcon icon9;
        icon9.addFile(QStringLiteral(":/icons/images/406.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actMDI_Tile->setIcon(icon9);
        actDoc_CloseALL = new QAction(MainWindow);
        actDoc_CloseALL->setObjectName(QStringLiteral("actDoc_CloseALL"));
        QIcon icon10;
        icon10.addFile(QStringLiteral(":/icons/images/128.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actDoc_CloseALL->setIcon(icon10);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        mdiArea = new QMdiArea(centralWidget);
        mdiArea->setObjectName(QStringLiteral("mdiArea"));
        mdiArea->setGeometry(QRect(65, 35, 366, 236));
        MainWindow->setCentralWidget(centralWidget);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setAutoFillBackground(true);
        mainToolBar->setToolButtonStyle(Qt::ToolButtonIconOnly);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        mainToolBar->addAction(actDoc_New);
        mainToolBar->addAction(actDoc_Open);
        mainToolBar->addAction(actDoc_CloseALL);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actEdit_Cut);
        mainToolBar->addAction(actEdit_Copy);
        mainToolBar->addAction(actEdit_Paste);
        mainToolBar->addAction(actEdit_Font);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actMDI_Mode);
        mainToolBar->addAction(actMDI_Cascade);
        mainToolBar->addAction(actMDI_Tile);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actQuit);

        retranslateUi(MainWindow);
        QObject::connect(actQuit, SIGNAL(triggered()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Demo5_4, MDI\345\272\224\347\224\250\347\250\213\345\272\217", nullptr));
        actDoc_New->setText(QApplication::translate("MainWindow", "\346\226\260\345\273\272\346\226\207\346\241\243", nullptr));
#ifndef QT_NO_TOOLTIP
        actDoc_New->setToolTip(QApplication::translate("MainWindow", "\346\226\260\345\273\272\346\226\207\346\241\243\347\252\227\345\217\243", nullptr));
#endif // QT_NO_TOOLTIP
        actQuit->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actQuit->setToolTip(QApplication::translate("MainWindow", "\351\200\200\345\207\272\346\234\254\347\263\273\347\273\237", nullptr));
#endif // QT_NO_TOOLTIP
        actDoc_Open->setText(QApplication::translate("MainWindow", "\346\211\223\345\274\200\346\226\207\346\241\243", nullptr));
#ifndef QT_NO_TOOLTIP
        actDoc_Open->setToolTip(QApplication::translate("MainWindow", "\346\211\223\345\274\200\346\226\207\346\241\243", nullptr));
#endif // QT_NO_TOOLTIP
        actEdit_Font->setText(QApplication::translate("MainWindow", "\345\255\227\344\275\223\350\256\276\347\275\256", nullptr));
#ifndef QT_NO_TOOLTIP
        actEdit_Font->setToolTip(QApplication::translate("MainWindow", "\345\255\227\344\275\223\350\256\276\347\275\256", nullptr));
#endif // QT_NO_TOOLTIP
        actEdit_Cut->setText(QApplication::translate("MainWindow", "\345\211\252\345\210\207", nullptr));
#ifndef QT_NO_TOOLTIP
        actEdit_Cut->setToolTip(QApplication::translate("MainWindow", "\345\211\252\345\210\207", nullptr));
#endif // QT_NO_TOOLTIP
        actEdit_Copy->setText(QApplication::translate("MainWindow", "\345\244\215\345\210\266", nullptr));
#ifndef QT_NO_TOOLTIP
        actEdit_Copy->setToolTip(QApplication::translate("MainWindow", "\345\244\215\345\210\266", nullptr));
#endif // QT_NO_TOOLTIP
        actEdit_Paste->setText(QApplication::translate("MainWindow", "\347\262\230\350\264\264", nullptr));
#ifndef QT_NO_TOOLTIP
        actEdit_Paste->setToolTip(QApplication::translate("MainWindow", "\347\262\230\350\264\264", nullptr));
#endif // QT_NO_TOOLTIP
        actMDI_Mode->setText(QApplication::translate("MainWindow", "MDI\346\250\241\345\274\217", nullptr));
#ifndef QT_NO_TOOLTIP
        actMDI_Mode->setToolTip(QApplication::translate("MainWindow", "\347\252\227\345\217\243\346\250\241\345\274\217\346\210\226\351\241\265\351\235\242\346\250\241\345\274\217", nullptr));
#endif // QT_NO_TOOLTIP
        actMDI_Cascade->setText(QApplication::translate("MainWindow", "\347\272\247\350\201\224\345\261\225\345\274\200", nullptr));
#ifndef QT_NO_TOOLTIP
        actMDI_Cascade->setToolTip(QApplication::translate("MainWindow", "\347\252\227\345\217\243\347\272\247\350\201\224\345\261\225\345\274\200", nullptr));
#endif // QT_NO_TOOLTIP
        actMDI_Tile->setText(QApplication::translate("MainWindow", "\345\271\263\351\223\272\345\261\225\345\274\200", nullptr));
#ifndef QT_NO_TOOLTIP
        actMDI_Tile->setToolTip(QApplication::translate("MainWindow", "\347\252\227\345\217\243\345\271\263\351\223\272\345\261\225\345\274\200", nullptr));
#endif // QT_NO_TOOLTIP
        actDoc_CloseALL->setText(QApplication::translate("MainWindow", "\345\205\263\351\227\255\345\205\250\351\203\250", nullptr));
#ifndef QT_NO_TOOLTIP
        actDoc_CloseALL->setToolTip(QApplication::translate("MainWindow", "\345\205\263\351\227\255\346\211\200\346\234\211\347\252\227\345\217\243", nullptr));
#endif // QT_NO_TOOLTIP
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
