/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.11.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actZoomReset;
    QAction *actQuit;
    QAction *actZoomIn;
    QAction *actZoomOut;
    QAction *actOpen;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_3;
    QSplitter *splitter;
    QFrame *frameSetup;
    QVBoxLayout *verticalLayout_2;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QRadioButton *radioFill_Pos;
    QRadioButton *radioFill_Neg;
    QRadioButton *radioFill_Both;
    QRadioButton *radioFill_None;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_3;
    QLabel *label_5;
    QDoubleSpinBox *spinY_Min;
    QPushButton *btnY_SetRange;
    QLabel *label_6;
    QDoubleSpinBox *spinY_Max;
    QLabel *label_3;
    QSpinBox *spinY_Ticks;
    QCheckBox *chkBoxY_GridLine;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout_4;
    QLabel *label_7;
    QDoubleSpinBox *spinX_Min;
    QPushButton *btnX_SetRange;
    QLabel *label_8;
    QDoubleSpinBox *spinX_Max;
    QLabel *label_4;
    QSpinBox *spinX_Ticks;
    QCheckBox *chkBoxX_GridLine;
    QSpacerItem *verticalSpacer;
    QFrame *frame_chart;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(859, 491);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        actZoomReset = new QAction(MainWindow);
        actZoomReset->setObjectName(QStringLiteral("actZoomReset"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/414.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actZoomReset->setIcon(icon);
        actQuit = new QAction(MainWindow);
        actQuit->setObjectName(QStringLiteral("actQuit"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actQuit->setIcon(icon1);
        actZoomIn = new QAction(MainWindow);
        actZoomIn->setObjectName(QStringLiteral("actZoomIn"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/418.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actZoomIn->setIcon(icon2);
        actZoomOut = new QAction(MainWindow);
        actZoomOut->setObjectName(QStringLiteral("actZoomOut"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/416.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actZoomOut->setIcon(icon3);
        actOpen = new QAction(MainWindow);
        actOpen->setObjectName(QStringLiteral("actOpen"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/001.GIF"), QSize(), QIcon::Normal, QIcon::Off);
        actOpen->setIcon(icon4);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout_3 = new QVBoxLayout(centralWidget);
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(2, 2, 2, 2);
        splitter = new QSplitter(centralWidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        frameSetup = new QFrame(splitter);
        frameSetup->setObjectName(QStringLiteral("frameSetup"));
        frameSetup->setEnabled(false);
        frameSetup->setMaximumSize(QSize(300, 16777215));
        frameSetup->setFrameShape(QFrame::Panel);
        frameSetup->setFrameShadow(QFrame::Raised);
        verticalLayout_2 = new QVBoxLayout(frameSetup);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        groupBox = new QGroupBox(frameSetup);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        radioFill_Pos = new QRadioButton(groupBox);
        radioFill_Pos->setObjectName(QStringLiteral("radioFill_Pos"));
        radioFill_Pos->setChecked(true);

        gridLayout->addWidget(radioFill_Pos, 0, 0, 1, 1);

        radioFill_Neg = new QRadioButton(groupBox);
        radioFill_Neg->setObjectName(QStringLiteral("radioFill_Neg"));
        radioFill_Neg->setChecked(false);

        gridLayout->addWidget(radioFill_Neg, 0, 1, 1, 1);

        radioFill_Both = new QRadioButton(groupBox);
        radioFill_Both->setObjectName(QStringLiteral("radioFill_Both"));
        radioFill_Both->setChecked(false);

        gridLayout->addWidget(radioFill_Both, 1, 0, 1, 1);

        radioFill_None = new QRadioButton(groupBox);
        radioFill_None->setObjectName(QStringLiteral("radioFill_None"));

        gridLayout->addWidget(radioFill_None, 1, 1, 1, 1);


        verticalLayout_2->addWidget(groupBox);

        groupBox_3 = new QGroupBox(frameSetup);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        gridLayout_3 = new QGridLayout(groupBox_3);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        gridLayout_3->setVerticalSpacing(10);
        label_5 = new QLabel(groupBox_3);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setAlignment(Qt::AlignCenter);

        gridLayout_3->addWidget(label_5, 0, 0, 1, 1);

        spinY_Min = new QDoubleSpinBox(groupBox_3);
        spinY_Min->setObjectName(QStringLiteral("spinY_Min"));
        spinY_Min->setMinimum(-50000);
        spinY_Min->setMaximum(50000);
        spinY_Min->setValue(-5);

        gridLayout_3->addWidget(spinY_Min, 0, 1, 1, 1);

        btnY_SetRange = new QPushButton(groupBox_3);
        btnY_SetRange->setObjectName(QStringLiteral("btnY_SetRange"));

        gridLayout_3->addWidget(btnY_SetRange, 0, 2, 1, 1);

        label_6 = new QLabel(groupBox_3);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setAlignment(Qt::AlignCenter);

        gridLayout_3->addWidget(label_6, 1, 0, 1, 1);

        spinY_Max = new QDoubleSpinBox(groupBox_3);
        spinY_Max->setObjectName(QStringLiteral("spinY_Max"));
        spinY_Max->setMaximum(1e+06);
        spinY_Max->setValue(5);

        gridLayout_3->addWidget(spinY_Max, 1, 1, 1, 1);

        label_3 = new QLabel(groupBox_3);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setAlignment(Qt::AlignCenter);

        gridLayout_3->addWidget(label_3, 2, 0, 1, 1);

        spinY_Ticks = new QSpinBox(groupBox_3);
        spinY_Ticks->setObjectName(QStringLiteral("spinY_Ticks"));
        spinY_Ticks->setMinimum(3);
        spinY_Ticks->setMaximum(1000);
        spinY_Ticks->setValue(5);
        spinY_Ticks->setDisplayIntegerBase(10);

        gridLayout_3->addWidget(spinY_Ticks, 2, 1, 1, 1);

        chkBoxY_GridLine = new QCheckBox(groupBox_3);
        chkBoxY_GridLine->setObjectName(QStringLiteral("chkBoxY_GridLine"));

        gridLayout_3->addWidget(chkBoxY_GridLine, 2, 2, 1, 1);


        verticalLayout_2->addWidget(groupBox_3);

        groupBox_4 = new QGroupBox(frameSetup);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        gridLayout_4 = new QGridLayout(groupBox_4);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setVerticalSpacing(10);
        label_7 = new QLabel(groupBox_4);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setAlignment(Qt::AlignCenter);

        gridLayout_4->addWidget(label_7, 0, 0, 1, 1);

        spinX_Min = new QDoubleSpinBox(groupBox_4);
        spinX_Min->setObjectName(QStringLiteral("spinX_Min"));
        spinX_Min->setMinimum(-50000);
        spinX_Min->setMaximum(50000);
        spinX_Min->setValue(-5);

        gridLayout_4->addWidget(spinX_Min, 0, 1, 1, 1);

        btnX_SetRange = new QPushButton(groupBox_4);
        btnX_SetRange->setObjectName(QStringLiteral("btnX_SetRange"));

        gridLayout_4->addWidget(btnX_SetRange, 0, 2, 1, 1);

        label_8 = new QLabel(groupBox_4);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setAlignment(Qt::AlignCenter);

        gridLayout_4->addWidget(label_8, 1, 0, 1, 1);

        spinX_Max = new QDoubleSpinBox(groupBox_4);
        spinX_Max->setObjectName(QStringLiteral("spinX_Max"));
        spinX_Max->setMaximum(1e+06);
        spinX_Max->setValue(5);

        gridLayout_4->addWidget(spinX_Max, 1, 1, 1, 1);

        label_4 = new QLabel(groupBox_4);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setAlignment(Qt::AlignCenter);

        gridLayout_4->addWidget(label_4, 2, 0, 1, 1);

        spinX_Ticks = new QSpinBox(groupBox_4);
        spinX_Ticks->setObjectName(QStringLiteral("spinX_Ticks"));
        spinX_Ticks->setMinimum(3);
        spinX_Ticks->setMaximum(1000);
        spinX_Ticks->setValue(5);
        spinX_Ticks->setDisplayIntegerBase(10);

        gridLayout_4->addWidget(spinX_Ticks, 2, 1, 1, 1);

        chkBoxX_GridLine = new QCheckBox(groupBox_4);
        chkBoxX_GridLine->setObjectName(QStringLiteral("chkBoxX_GridLine"));

        gridLayout_4->addWidget(chkBoxX_GridLine, 2, 2, 1, 1);


        verticalLayout_2->addWidget(groupBox_4);

        verticalSpacer = new QSpacerItem(20, 53, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);

        splitter->addWidget(frameSetup);
        frame_chart = new QFrame(splitter);
        frame_chart->setObjectName(QStringLiteral("frame_chart"));
        splitter->addWidget(frame_chart);

        verticalLayout_3->addWidget(splitter);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 859, 23));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        mainToolBar->addAction(actOpen);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actZoomIn);
        mainToolBar->addAction(actZoomOut);
        mainToolBar->addAction(actZoomReset);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actQuit);

        retranslateUi(MainWindow);
        QObject::connect(actQuit, SIGNAL(triggered()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "\351\235\242\347\247\257\345\241\253\345\205\205\345\233\276", nullptr));
        actZoomReset->setText(QApplication::translate("MainWindow", "\345\216\237\345\247\213\345\244\247\345\260\217", nullptr));
#ifndef QT_NO_TOOLTIP
        actZoomReset->setToolTip(QApplication::translate("MainWindow", "\345\216\237\345\247\213\345\244\247\345\260\217", nullptr));
#endif // QT_NO_TOOLTIP
        actQuit->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actQuit->setToolTip(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#endif // QT_NO_TOOLTIP
        actZoomIn->setText(QApplication::translate("MainWindow", "\346\224\276\345\244\247", nullptr));
#ifndef QT_NO_TOOLTIP
        actZoomIn->setToolTip(QApplication::translate("MainWindow", "\346\224\276\345\244\247", nullptr));
#endif // QT_NO_TOOLTIP
        actZoomOut->setText(QApplication::translate("MainWindow", "\347\274\251\345\260\217", nullptr));
#ifndef QT_NO_TOOLTIP
        actZoomOut->setToolTip(QApplication::translate("MainWindow", "\347\274\251\345\260\217", nullptr));
#endif // QT_NO_TOOLTIP
        actOpen->setText(QApplication::translate("MainWindow", "\346\211\223\345\274\200\346\226\207\344\273\266", nullptr));
#ifndef QT_NO_TOOLTIP
        actOpen->setToolTip(QApplication::translate("MainWindow", "\346\211\223\345\274\200\346\225\260\346\215\256\346\226\207\344\273\266", nullptr));
#endif // QT_NO_TOOLTIP
        groupBox->setTitle(QApplication::translate("MainWindow", "\345\241\253\345\205\205\347\261\273\345\236\213", nullptr));
        radioFill_Pos->setText(QApplication::translate("MainWindow", "\344\270\212\345\215\212\351\203\250\345\210\206\345\241\253\345\205\205", nullptr));
        radioFill_Neg->setText(QApplication::translate("MainWindow", "\344\270\213\345\215\212\351\203\250\345\210\206\345\241\253\345\205\205", nullptr));
        radioFill_Both->setText(QApplication::translate("MainWindow", "\344\270\212\343\200\201\344\270\213\345\235\207\345\241\253\345\205\205", nullptr));
        radioFill_None->setText(QApplication::translate("MainWindow", "\344\270\215\345\241\253\345\205\205", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "Y\350\275\264--QValueAxis", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "\346\234\200\345\260\217\345\200\274", nullptr));
        btnY_SetRange->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256\350\214\203\345\233\264", nullptr));
        label_6->setText(QApplication::translate("MainWindow", "\346\234\200\345\244\247\345\200\274", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "\345\210\206\345\272\246\344\270\252\346\225\260", nullptr));
        chkBoxY_GridLine->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\347\275\221\346\240\274\347\272\277", nullptr));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "X\350\275\264--QValueAxis", nullptr));
        label_7->setText(QApplication::translate("MainWindow", "\346\234\200\345\260\217\345\200\274", nullptr));
        btnX_SetRange->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256\350\214\203\345\233\264", nullptr));
        label_8->setText(QApplication::translate("MainWindow", "\346\234\200\345\244\247\345\200\274", nullptr));
        label_4->setText(QApplication::translate("MainWindow", "\345\210\206\345\272\246\344\270\252\346\225\260", nullptr));
        chkBoxX_GridLine->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\347\275\221\346\240\274\347\272\277", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
