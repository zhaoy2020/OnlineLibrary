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
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
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
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QComboBox *comboTheme;
    QCheckBox *chkBox_Legend;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_3;
    QLabel *label_5;
    QDoubleSpinBox *spinMag_Min;
    QPushButton *btnMag_SetRange;
    QLabel *label_6;
    QDoubleSpinBox *spinMag_Max;
    QLabel *label_3;
    QSpinBox *spinMag_Ticks;
    QCheckBox *chkBoxMag_Point;
    QGroupBox *groupBox_5;
    QGridLayout *gridLayout_4;
    QLabel *label_9;
    QDoubleSpinBox *spinPh_Min;
    QPushButton *btnPh_SetRange;
    QLabel *label_10;
    QDoubleSpinBox *spinPh_Max;
    QLabel *label_11;
    QSpinBox *spinPh_Ticks;
    QCheckBox *chkBoxPh_Point;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout;
    QDoubleSpinBox *spinX_Max;
    QLabel *label;
    QSpinBox *spinX_MinorTicks;
    QLabel *label_8;
    QDoubleSpinBox *spinX_Min;
    QLabel *label_7;
    QPushButton *btnX_SetRange;
    QLabel *label_4;
    QSpacerItem *verticalSpacer;
    QFrame *frameChart;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(760, 497);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        MainWindow->setStyleSheet(QLatin1String("QSpinBox, QDoubleSpinBox{\n"
"	max-width:100px;\n"
"	min-height:20px;\n"
"}"));
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
        frameSetup->setEnabled(true);
        frameSetup->setMaximumSize(QSize(300, 16777215));
        frameSetup->setFrameShape(QFrame::Panel);
        frameSetup->setFrameShadow(QFrame::Raised);
        verticalLayout_2 = new QVBoxLayout(frameSetup);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(5, 5, 5, 5);
        groupBox = new QGroupBox(frameSetup);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        horizontalLayout = new QHBoxLayout(groupBox);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 5, -1, 5);
        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_2->setMargin(0);
        label_2->setIndent(10);

        horizontalLayout->addWidget(label_2);

        comboTheme = new QComboBox(groupBox);
        comboTheme->addItem(QString());
        comboTheme->addItem(QString());
        comboTheme->addItem(QString());
        comboTheme->addItem(QString());
        comboTheme->addItem(QString());
        comboTheme->addItem(QString());
        comboTheme->addItem(QString());
        comboTheme->addItem(QString());
        comboTheme->setObjectName(QStringLiteral("comboTheme"));
        comboTheme->setMaximumSize(QSize(100, 16777215));

        horizontalLayout->addWidget(comboTheme);

        chkBox_Legend = new QCheckBox(groupBox);
        chkBox_Legend->setObjectName(QStringLiteral("chkBox_Legend"));
        chkBox_Legend->setChecked(true);

        horizontalLayout->addWidget(chkBox_Legend);


        verticalLayout_2->addWidget(groupBox);

        groupBox_3 = new QGroupBox(frameSetup);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        gridLayout_3 = new QGridLayout(groupBox_3);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        gridLayout_3->setVerticalSpacing(10);
        gridLayout_3->setContentsMargins(-1, 5, -1, 5);
        label_5 = new QLabel(groupBox_3);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_5->setIndent(10);

        gridLayout_3->addWidget(label_5, 0, 0, 1, 1);

        spinMag_Min = new QDoubleSpinBox(groupBox_3);
        spinMag_Min->setObjectName(QStringLiteral("spinMag_Min"));
        spinMag_Min->setMaximumSize(QSize(102, 16777215));
        spinMag_Min->setDecimals(1);
        spinMag_Min->setMinimum(-90000);
        spinMag_Min->setMaximum(90000);
        spinMag_Min->setSingleStep(10);
        spinMag_Min->setValue(-40);

        gridLayout_3->addWidget(spinMag_Min, 0, 1, 1, 1);

        btnMag_SetRange = new QPushButton(groupBox_3);
        btnMag_SetRange->setObjectName(QStringLiteral("btnMag_SetRange"));

        gridLayout_3->addWidget(btnMag_SetRange, 0, 2, 1, 1);

        label_6 = new QLabel(groupBox_3);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_6->setIndent(10);

        gridLayout_3->addWidget(label_6, 1, 0, 1, 1);

        spinMag_Max = new QDoubleSpinBox(groupBox_3);
        spinMag_Max->setObjectName(QStringLiteral("spinMag_Max"));
        spinMag_Max->setDecimals(1);
        spinMag_Max->setMinimum(-90000);
        spinMag_Max->setMaximum(90000);
        spinMag_Max->setSingleStep(10);
        spinMag_Max->setValue(10);

        gridLayout_3->addWidget(spinMag_Max, 1, 1, 1, 1);

        label_3 = new QLabel(groupBox_3);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_3->setIndent(10);

        gridLayout_3->addWidget(label_3, 2, 0, 1, 1);

        spinMag_Ticks = new QSpinBox(groupBox_3);
        spinMag_Ticks->setObjectName(QStringLiteral("spinMag_Ticks"));
        spinMag_Ticks->setMinimum(3);
        spinMag_Ticks->setMaximum(1000);
        spinMag_Ticks->setValue(6);
        spinMag_Ticks->setDisplayIntegerBase(10);

        gridLayout_3->addWidget(spinMag_Ticks, 2, 1, 1, 1);

        chkBoxMag_Point = new QCheckBox(groupBox_3);
        chkBoxMag_Point->setObjectName(QStringLiteral("chkBoxMag_Point"));
        chkBoxMag_Point->setChecked(true);

        gridLayout_3->addWidget(chkBoxMag_Point, 2, 2, 1, 1);


        verticalLayout_2->addWidget(groupBox_3);

        groupBox_5 = new QGroupBox(frameSetup);
        groupBox_5->setObjectName(QStringLiteral("groupBox_5"));
        gridLayout_4 = new QGridLayout(groupBox_5);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setVerticalSpacing(10);
        gridLayout_4->setContentsMargins(-1, 5, -1, 5);
        label_9 = new QLabel(groupBox_5);
        label_9->setObjectName(QStringLiteral("label_9"));
        label_9->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_9->setIndent(10);

        gridLayout_4->addWidget(label_9, 0, 0, 1, 1);

        spinPh_Min = new QDoubleSpinBox(groupBox_5);
        spinPh_Min->setObjectName(QStringLiteral("spinPh_Min"));
        spinPh_Min->setDecimals(1);
        spinPh_Min->setMinimum(-50000);
        spinPh_Min->setMaximum(50000);
        spinPh_Min->setSingleStep(10);
        spinPh_Min->setValue(-200);

        gridLayout_4->addWidget(spinPh_Min, 0, 1, 1, 1);

        btnPh_SetRange = new QPushButton(groupBox_5);
        btnPh_SetRange->setObjectName(QStringLiteral("btnPh_SetRange"));

        gridLayout_4->addWidget(btnPh_SetRange, 0, 2, 1, 1);

        label_10 = new QLabel(groupBox_5);
        label_10->setObjectName(QStringLiteral("label_10"));
        label_10->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_10->setIndent(10);

        gridLayout_4->addWidget(label_10, 1, 0, 1, 1);

        spinPh_Max = new QDoubleSpinBox(groupBox_5);
        spinPh_Max->setObjectName(QStringLiteral("spinPh_Max"));
        spinPh_Max->setDecimals(1);
        spinPh_Max->setMinimum(-50000);
        spinPh_Max->setMaximum(50000);
        spinPh_Max->setSingleStep(10);
        spinPh_Max->setValue(0);

        gridLayout_4->addWidget(spinPh_Max, 1, 1, 1, 1);

        label_11 = new QLabel(groupBox_5);
        label_11->setObjectName(QStringLiteral("label_11"));
        label_11->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_11->setIndent(10);

        gridLayout_4->addWidget(label_11, 2, 0, 1, 1);

        spinPh_Ticks = new QSpinBox(groupBox_5);
        spinPh_Ticks->setObjectName(QStringLiteral("spinPh_Ticks"));
        spinPh_Ticks->setMinimum(3);
        spinPh_Ticks->setMaximum(1000);
        spinPh_Ticks->setValue(6);
        spinPh_Ticks->setDisplayIntegerBase(10);

        gridLayout_4->addWidget(spinPh_Ticks, 2, 1, 1, 1);

        chkBoxPh_Point = new QCheckBox(groupBox_5);
        chkBoxPh_Point->setObjectName(QStringLiteral("chkBoxPh_Point"));
        chkBoxPh_Point->setChecked(true);

        gridLayout_4->addWidget(chkBoxPh_Point, 2, 2, 1, 1);


        verticalLayout_2->addWidget(groupBox_5);

        groupBox_4 = new QGroupBox(frameSetup);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        gridLayout = new QGridLayout(groupBox_4);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(-1, 5, -1, 5);
        spinX_Max = new QDoubleSpinBox(groupBox_4);
        spinX_Max->setObjectName(QStringLiteral("spinX_Max"));
        spinX_Max->setMinimum(0);
        spinX_Max->setMaximum(9e+06);
        spinX_Max->setSingleStep(10);
        spinX_Max->setValue(100);

        gridLayout->addWidget(spinX_Max, 1, 1, 1, 1);

        label = new QLabel(groupBox_4);
        label->setObjectName(QStringLiteral("label"));
        label->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label->setIndent(10);

        gridLayout->addWidget(label, 2, 0, 1, 1);

        spinX_MinorTicks = new QSpinBox(groupBox_4);
        spinX_MinorTicks->setObjectName(QStringLiteral("spinX_MinorTicks"));
        spinX_MinorTicks->setMinimum(1);
        spinX_MinorTicks->setMaximum(100);
        spinX_MinorTicks->setValue(9);

        gridLayout->addWidget(spinX_MinorTicks, 2, 1, 1, 1);

        label_8 = new QLabel(groupBox_4);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_8->setIndent(10);

        gridLayout->addWidget(label_8, 1, 0, 1, 1);

        spinX_Min = new QDoubleSpinBox(groupBox_4);
        spinX_Min->setObjectName(QStringLiteral("spinX_Min"));
        spinX_Min->setDecimals(3);
        spinX_Min->setMinimum(0.001);
        spinX_Min->setMaximum(50000);
        spinX_Min->setSingleStep(0.1);
        spinX_Min->setValue(0.1);

        gridLayout->addWidget(spinX_Min, 0, 1, 1, 1);

        label_7 = new QLabel(groupBox_4);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_7->setIndent(10);

        gridLayout->addWidget(label_7, 0, 0, 1, 1);

        btnX_SetRange = new QPushButton(groupBox_4);
        btnX_SetRange->setObjectName(QStringLiteral("btnX_SetRange"));

        gridLayout->addWidget(btnX_SetRange, 0, 2, 1, 1);

        label_4 = new QLabel(groupBox_4);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout->addWidget(label_4, 2, 2, 1, 1);


        verticalLayout_2->addWidget(groupBox_4);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);

        splitter->addWidget(frameSetup);
        frameChart = new QFrame(splitter);
        frameChart->setObjectName(QStringLiteral("frameChart"));
        frameChart->setFrameShape(QFrame::Panel);
        frameChart->setFrameShadow(QFrame::Raised);
        splitter->addWidget(frameChart);

        verticalLayout_3->addWidget(splitter);

        MainWindow->setCentralWidget(centralWidget);
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
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "\345\257\271\346\225\260\345\235\220\346\240\207\350\275\264\344\270\216\345\244\232\345\235\220\346\240\207\350\275\264", nullptr));
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
        groupBox->setTitle(QApplication::translate("MainWindow", "\345\233\276\350\241\250", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "\344\270\273\351\242\230", nullptr));
        comboTheme->setItemText(0, QApplication::translate("MainWindow", "Light", nullptr));
        comboTheme->setItemText(1, QApplication::translate("MainWindow", "BlueCerulean", nullptr));
        comboTheme->setItemText(2, QApplication::translate("MainWindow", "Dark", nullptr));
        comboTheme->setItemText(3, QApplication::translate("MainWindow", "BrownSand", nullptr));
        comboTheme->setItemText(4, QApplication::translate("MainWindow", "BlueNcs", nullptr));
        comboTheme->setItemText(5, QApplication::translate("MainWindow", "HighContrast", nullptr));
        comboTheme->setItemText(6, QApplication::translate("MainWindow", "BlueIcy", nullptr));
        comboTheme->setItemText(7, QApplication::translate("MainWindow", "Qt", nullptr));

#ifndef QT_NO_TOOLTIP
        comboTheme->setToolTip(QApplication::translate("MainWindow", "\350\256\276\347\275\256\345\233\276\350\241\250\344\270\273\351\242\230", nullptr));
#endif // QT_NO_TOOLTIP
        chkBox_Legend->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\345\233\276\344\276\213", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "\345\271\205\351\242\221\346\233\262\347\272\277", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "\346\234\200\345\260\217\345\200\274", nullptr));
        spinMag_Min->setSuffix(QApplication::translate("MainWindow", " dB", nullptr));
        btnMag_SetRange->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256\350\214\203\345\233\264", nullptr));
        label_6->setText(QApplication::translate("MainWindow", "\346\234\200\345\244\247\345\200\274", nullptr));
        spinMag_Max->setSuffix(QApplication::translate("MainWindow", " dB", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "\344\270\273\345\210\206\345\272\246\344\270\252\346\225\260", nullptr));
        chkBoxMag_Point->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\346\225\260\346\215\256\347\202\271", nullptr));
        groupBox_5->setTitle(QApplication::translate("MainWindow", "\347\233\270\351\242\221\346\233\262\347\272\277", nullptr));
        label_9->setText(QApplication::translate("MainWindow", "\346\234\200\345\260\217\345\200\274", nullptr));
        spinPh_Min->setSuffix(QApplication::translate("MainWindow", "\302\260", nullptr));
        btnPh_SetRange->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256\350\214\203\345\233\264", nullptr));
        label_10->setText(QApplication::translate("MainWindow", "\346\234\200\345\244\247\345\200\274", nullptr));
        spinPh_Max->setSuffix(QApplication::translate("MainWindow", "\302\260", nullptr));
        label_11->setText(QApplication::translate("MainWindow", "\344\270\273\345\210\206\345\272\246\344\270\252\346\225\260", nullptr));
        chkBoxPh_Point->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\346\225\260\346\215\256\347\202\271", nullptr));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "\351\242\221\347\216\207\345\235\220\346\240\207\350\275\264--QLogValueAxis", nullptr));
        label->setText(QApplication::translate("MainWindow", "\346\254\241\345\210\206\345\272\246\344\270\252\346\225\260", nullptr));
        label_8->setText(QApplication::translate("MainWindow", "\346\234\200\345\244\247\345\200\274", nullptr));
        label_7->setText(QApplication::translate("MainWindow", "\346\234\200\345\260\217\345\200\274", nullptr));
        btnX_SetRange->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256\350\214\203\345\233\264", nullptr));
        label_4->setText(QApplication::translate("MainWindow", "(\345\210\267\346\226\260\346\227\266\346\211\215\347\224\237\346\225\210)", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
