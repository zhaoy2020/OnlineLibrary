/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.11.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QChartView>
#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
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
    QAction *actRedraw;
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout;
    QFrame *frame;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout_4;
    QLabel *label_7;
    QComboBox *comboTheme;
    QCheckBox *chkBox_Legend;
    QCheckBox *chkBox_ShowPoints;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout;
    QLabel *label_3;
    QSpinBox *spinAngle_Min;
    QLabel *label_4;
    QSpinBox *spinAngle_Max;
    QLabel *label;
    QSpinBox *spinAngle_Ticks;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_2;
    QLabel *label_2;
    QSpinBox *spinRadial_Max;
    QLabel *label_5;
    QSpinBox *spinRadial_Ticks;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_3;
    QLabel *label_6;
    QSpinBox *spinRotate_Angle;
    QPushButton *btnRotate;
    QSpacerItem *verticalSpacer;
    QChartView *chartView;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(723, 491);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        actZoomReset = new QAction(MainWindow);
        actZoomReset->setObjectName(QStringLiteral("actZoomReset"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/images/images/414.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actZoomReset->setIcon(icon);
        actQuit = new QAction(MainWindow);
        actQuit->setObjectName(QStringLiteral("actQuit"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/images/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actQuit->setIcon(icon1);
        actZoomIn = new QAction(MainWindow);
        actZoomIn->setObjectName(QStringLiteral("actZoomIn"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/images/images/418.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actZoomIn->setIcon(icon2);
        actZoomOut = new QAction(MainWindow);
        actZoomOut->setObjectName(QStringLiteral("actZoomOut"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/images/images/416.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actZoomOut->setIcon(icon3);
        actRedraw = new QAction(MainWindow);
        actRedraw->setObjectName(QStringLiteral("actRedraw"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/images/images/828.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actRedraw->setIcon(icon4);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        horizontalLayout = new QHBoxLayout(centralWidget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(5, 5, 5, 5);
        frame = new QFrame(centralWidget);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setFrameShape(QFrame::Panel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frame);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(5, -1, -1, -1);
        groupBox_4 = new QGroupBox(frame);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        gridLayout_4 = new QGridLayout(groupBox_4);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setVerticalSpacing(10);
        gridLayout_4->setContentsMargins(-1, 5, -1, 5);
        label_7 = new QLabel(groupBox_4);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_7->setMargin(0);
        label_7->setIndent(10);

        gridLayout_4->addWidget(label_7, 0, 0, 1, 1);

        comboTheme = new QComboBox(groupBox_4);
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

        gridLayout_4->addWidget(comboTheme, 0, 1, 1, 1);

        chkBox_Legend = new QCheckBox(groupBox_4);
        chkBox_Legend->setObjectName(QStringLiteral("chkBox_Legend"));
        chkBox_Legend->setChecked(true);

        gridLayout_4->addWidget(chkBox_Legend, 1, 1, 1, 1);

        chkBox_ShowPoints = new QCheckBox(groupBox_4);
        chkBox_ShowPoints->setObjectName(QStringLiteral("chkBox_ShowPoints"));
        chkBox_ShowPoints->setChecked(true);

        gridLayout_4->addWidget(chkBox_ShowPoints, 2, 1, 1, 1);


        verticalLayout->addWidget(groupBox_4);

        groupBox_2 = new QGroupBox(frame);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        gridLayout = new QGridLayout(groupBox_2);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_3->setIndent(10);

        gridLayout->addWidget(label_3, 0, 0, 1, 1);

        spinAngle_Min = new QSpinBox(groupBox_2);
        spinAngle_Min->setObjectName(QStringLiteral("spinAngle_Min"));
        spinAngle_Min->setMinimum(-3600);
        spinAngle_Min->setMaximum(3600);
        spinAngle_Min->setSingleStep(10);

        gridLayout->addWidget(spinAngle_Min, 0, 1, 1, 1);

        label_4 = new QLabel(groupBox_2);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_4->setIndent(10);

        gridLayout->addWidget(label_4, 1, 0, 1, 1);

        spinAngle_Max = new QSpinBox(groupBox_2);
        spinAngle_Max->setObjectName(QStringLiteral("spinAngle_Max"));
        spinAngle_Max->setMinimum(-3600);
        spinAngle_Max->setMaximum(3600);
        spinAngle_Max->setSingleStep(10);
        spinAngle_Max->setValue(360);

        gridLayout->addWidget(spinAngle_Max, 1, 1, 1, 1);

        label = new QLabel(groupBox_2);
        label->setObjectName(QStringLiteral("label"));
        label->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label->setIndent(10);

        gridLayout->addWidget(label, 2, 0, 1, 1);

        spinAngle_Ticks = new QSpinBox(groupBox_2);
        spinAngle_Ticks->setObjectName(QStringLiteral("spinAngle_Ticks"));
        spinAngle_Ticks->setMinimum(4);
        spinAngle_Ticks->setMaximum(5000);
        spinAngle_Ticks->setValue(9);

        gridLayout->addWidget(spinAngle_Ticks, 2, 1, 1, 1);


        verticalLayout->addWidget(groupBox_2);

        groupBox_3 = new QGroupBox(frame);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        gridLayout_2 = new QGridLayout(groupBox_3);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        label_2 = new QLabel(groupBox_3);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_2->setIndent(10);

        gridLayout_2->addWidget(label_2, 0, 0, 1, 1);

        spinRadial_Max = new QSpinBox(groupBox_3);
        spinRadial_Max->setObjectName(QStringLiteral("spinRadial_Max"));
        spinRadial_Max->setMinimum(1);
        spinRadial_Max->setMaximum(500);
        spinRadial_Max->setValue(12);

        gridLayout_2->addWidget(spinRadial_Max, 0, 1, 1, 1);

        label_5 = new QLabel(groupBox_3);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_5->setIndent(10);

        gridLayout_2->addWidget(label_5, 1, 0, 1, 1);

        spinRadial_Ticks = new QSpinBox(groupBox_3);
        spinRadial_Ticks->setObjectName(QStringLiteral("spinRadial_Ticks"));
        spinRadial_Ticks->setMinimum(2);
        spinRadial_Ticks->setMaximum(100);
        spinRadial_Ticks->setValue(6);

        gridLayout_2->addWidget(spinRadial_Ticks, 1, 1, 1, 1);


        verticalLayout->addWidget(groupBox_3);

        groupBox = new QGroupBox(frame);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout_3 = new QGridLayout(groupBox);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        label_6 = new QLabel(groupBox);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_6->setIndent(10);

        gridLayout_3->addWidget(label_6, 0, 0, 1, 1);

        spinRotate_Angle = new QSpinBox(groupBox);
        spinRotate_Angle->setObjectName(QStringLiteral("spinRotate_Angle"));
        spinRotate_Angle->setMinimum(-180);
        spinRotate_Angle->setMaximum(180);
        spinRotate_Angle->setValue(90);

        gridLayout_3->addWidget(spinRotate_Angle, 0, 1, 1, 1);

        btnRotate = new QPushButton(groupBox);
        btnRotate->setObjectName(QStringLiteral("btnRotate"));
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/images/images/exec.ico"), QSize(), QIcon::Normal, QIcon::Off);
        btnRotate->setIcon(icon5);

        gridLayout_3->addWidget(btnRotate, 1, 1, 1, 1);


        verticalLayout->addWidget(groupBox);

        verticalSpacer = new QSpacerItem(20, 106, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        horizontalLayout->addWidget(frame);

        chartView = new QChartView(centralWidget);
        chartView->setObjectName(QStringLiteral("chartView"));
        chartView->setDragMode(QGraphicsView::RubberBandDrag);

        horizontalLayout->addWidget(chartView);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 723, 23));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        mainToolBar->addAction(actRedraw);
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
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "\346\236\201\345\235\220\346\240\207\345\233\276", nullptr));
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
        actRedraw->setText(QApplication::translate("MainWindow", "\351\207\215\347\273\230\345\216\237\345\247\213\345\233\276\345\275\242", nullptr));
#ifndef QT_NO_TOOLTIP
        actRedraw->setToolTip(QApplication::translate("MainWindow", "\351\207\215\347\273\230\345\216\237\345\247\213\345\233\276\345\275\242", nullptr));
#endif // QT_NO_TOOLTIP
        groupBox_4->setTitle(QApplication::translate("MainWindow", "\345\233\276\350\241\250", nullptr));
        label_7->setText(QApplication::translate("MainWindow", "\344\270\273\351\242\230", nullptr));
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
        chkBox_ShowPoints->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\346\225\260\346\215\256\347\202\271", nullptr));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "\345\234\206\345\221\250\345\235\220\346\240\207", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "\346\234\200\345\260\217\345\200\274", nullptr));
        spinAngle_Min->setSuffix(QApplication::translate("MainWindow", "\302\260", nullptr));
        label_4->setText(QApplication::translate("MainWindow", "\346\234\200\345\244\247\345\200\274", nullptr));
        spinAngle_Max->setSuffix(QApplication::translate("MainWindow", "\302\260", nullptr));
        label->setText(QApplication::translate("MainWindow", "\345\234\206\345\221\250\345\210\273\345\272\246\346\225\260", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "\345\276\204\345\220\221\345\235\220\346\240\207", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "\345\276\204\345\220\221\351\225\277\345\272\246", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "\345\276\204\345\220\221\345\210\273\345\272\246\346\225\260", nullptr));
        groupBox->setTitle(QApplication::translate("MainWindow", "\346\227\213\350\275\254\345\217\230\346\215\242", nullptr));
        label_6->setText(QApplication::translate("MainWindow", "\346\227\213\350\275\254\350\247\222\345\272\246", nullptr));
        spinRotate_Angle->setSuffix(QApplication::translate("MainWindow", "\302\260", nullptr));
        btnRotate->setText(QApplication::translate("MainWindow", "\346\227\213\350\275\254", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
