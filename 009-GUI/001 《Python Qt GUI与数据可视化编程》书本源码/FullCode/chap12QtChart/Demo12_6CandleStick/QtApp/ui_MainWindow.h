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
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTableView>
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
    QAction *actdrawChart;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_6;
    QSplitter *splitter;
    QTabWidget *tabWidget;
    QWidget *tab_Data;
    QVBoxLayout *verticalLayout_3;
    QFrame *frame;
    QVBoxLayout *verticalLayout;
    QTableView *tableView;
    QWidget *tab_Setup;
    QVBoxLayout *verticalLayout_2;
    QFrame *frameData;
    QVBoxLayout *verticalLayout_5;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_2;
    QLabel *label;
    QComboBox *comboTheme;
    QCheckBox *chkBox_Legend;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_7;
    QCheckBox *chkBox_Caps;
    QCheckBox *chkBox_Outline;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_3;
    QLabel *label_5;
    QDoubleSpinBox *spinY_Min;
    QPushButton *btnY_SetRange;
    QLabel *label_6;
    QDoubleSpinBox *spinY_Max;
    QLabel *label_3;
    QSpinBox *spinY_Ticks;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout;
    QLabel *label_4;
    QSpinBox *btnX_Ticks;
    QLabel *label_7;
    QComboBox *comboDateFormat;
    QPushButton *btnY_SetFormat;
    QSpacerItem *verticalSpacer;
    QFrame *frame_chart;
    QVBoxLayout *verticalLayout_4;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(729, 567);
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
        actOpen = new QAction(MainWindow);
        actOpen->setObjectName(QStringLiteral("actOpen"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/images/images/001.GIF"), QSize(), QIcon::Normal, QIcon::Off);
        actOpen->setIcon(icon4);
        actdrawChart = new QAction(MainWindow);
        actdrawChart->setObjectName(QStringLiteral("actdrawChart"));
        actdrawChart->setEnabled(false);
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/images/images/f14.ico"), QSize(), QIcon::Normal, QIcon::Off);
        actdrawChart->setIcon(icon5);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout_6 = new QVBoxLayout(centralWidget);
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setContentsMargins(11, 11, 11, 11);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(2, 2, 2, 2);
        splitter = new QSplitter(centralWidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        tabWidget = new QTabWidget(splitter);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setDocumentMode(true);
        tab_Data = new QWidget();
        tab_Data->setObjectName(QStringLiteral("tab_Data"));
        verticalLayout_3 = new QVBoxLayout(tab_Data);
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(tab_Data);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setFrameShape(QFrame::Panel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frame);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(5, 5, 5, 5);
        tableView = new QTableView(frame);
        tableView->setObjectName(QStringLiteral("tableView"));
        tableView->setAlternatingRowColors(true);
        tableView->horizontalHeader()->setDefaultSectionSize(50);
        tableView->verticalHeader()->setDefaultSectionSize(24);

        verticalLayout->addWidget(tableView);


        verticalLayout_3->addWidget(frame);

        tabWidget->addTab(tab_Data, QString());
        tab_Setup = new QWidget();
        tab_Setup->setObjectName(QStringLiteral("tab_Setup"));
        tab_Setup->setEnabled(false);
        verticalLayout_2 = new QVBoxLayout(tab_Setup);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        frameData = new QFrame(tab_Setup);
        frameData->setObjectName(QStringLiteral("frameData"));
        frameData->setFrameShape(QFrame::Panel);
        frameData->setFrameShadow(QFrame::Raised);
        verticalLayout_5 = new QVBoxLayout(frameData);
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        groupBox = new QGroupBox(frameData);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout_2 = new QGridLayout(groupBox);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setVerticalSpacing(10);
        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));
        label->setAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(label, 0, 0, 1, 1);

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
        comboTheme->setMinimumSize(QSize(120, 0));

        gridLayout_2->addWidget(comboTheme, 0, 1, 1, 1);

        chkBox_Legend = new QCheckBox(groupBox);
        chkBox_Legend->setObjectName(QStringLiteral("chkBox_Legend"));
        chkBox_Legend->setChecked(true);

        gridLayout_2->addWidget(chkBox_Legend, 1, 1, 1, 1);


        verticalLayout_5->addWidget(groupBox);

        groupBox_2 = new QGroupBox(frameData);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        verticalLayout_7 = new QVBoxLayout(groupBox_2);
        verticalLayout_7->setSpacing(6);
        verticalLayout_7->setContentsMargins(11, 11, 11, 11);
        verticalLayout_7->setObjectName(QStringLiteral("verticalLayout_7"));
        chkBox_Caps = new QCheckBox(groupBox_2);
        chkBox_Caps->setObjectName(QStringLiteral("chkBox_Caps"));

        verticalLayout_7->addWidget(chkBox_Caps);

        chkBox_Outline = new QCheckBox(groupBox_2);
        chkBox_Outline->setObjectName(QStringLiteral("chkBox_Outline"));

        verticalLayout_7->addWidget(chkBox_Outline);


        verticalLayout_5->addWidget(groupBox_2);

        groupBox_3 = new QGroupBox(frameData);
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
        spinY_Min->setMaximum(1e+06);

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
        spinY_Max->setValue(20);

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


        verticalLayout_5->addWidget(groupBox_3);

        groupBox_4 = new QGroupBox(frameData);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        gridLayout = new QGridLayout(groupBox_4);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setVerticalSpacing(10);
        label_4 = new QLabel(groupBox_4);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_4, 0, 0, 1, 1);

        btnX_Ticks = new QSpinBox(groupBox_4);
        btnX_Ticks->setObjectName(QStringLiteral("btnX_Ticks"));
        btnX_Ticks->setMinimum(3);
        btnX_Ticks->setMaximum(1000);
        btnX_Ticks->setValue(10);
        btnX_Ticks->setDisplayIntegerBase(10);

        gridLayout->addWidget(btnX_Ticks, 0, 1, 1, 1);

        label_7 = new QLabel(groupBox_4);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_7, 1, 0, 1, 1);

        comboDateFormat = new QComboBox(groupBox_4);
        comboDateFormat->addItem(QString());
        comboDateFormat->addItem(QString());
        comboDateFormat->addItem(QString());
        comboDateFormat->addItem(QString());
        comboDateFormat->setObjectName(QStringLiteral("comboDateFormat"));
        comboDateFormat->setEditable(true);

        gridLayout->addWidget(comboDateFormat, 1, 1, 1, 1);

        btnY_SetFormat = new QPushButton(groupBox_4);
        btnY_SetFormat->setObjectName(QStringLiteral("btnY_SetFormat"));

        gridLayout->addWidget(btnY_SetFormat, 2, 1, 1, 1);


        verticalLayout_5->addWidget(groupBox_4);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer);


        verticalLayout_2->addWidget(frameData);

        tabWidget->addTab(tab_Setup, QString());
        splitter->addWidget(tabWidget);
        frame_chart = new QFrame(splitter);
        frame_chart->setObjectName(QStringLiteral("frame_chart"));
        frame_chart->setFrameShape(QFrame::StyledPanel);
        frame_chart->setFrameShadow(QFrame::Raised);
        verticalLayout_4 = new QVBoxLayout(frame_chart);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(2, 2, 2, 2);
        splitter->addWidget(frame_chart);

        verticalLayout_6->addWidget(splitter);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 729, 23));
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

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "\350\234\241\347\203\233\345\233\276\357\274\214\346\227\266\351\227\264\346\227\245\346\234\237\345\235\220\346\240\207\350\275\264", nullptr));
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
        actdrawChart->setText(QApplication::translate("MainWindow", "\347\273\230\345\210\266\345\233\276\350\241\250", nullptr));
#ifndef QT_NO_TOOLTIP
        actdrawChart->setToolTip(QApplication::translate("MainWindow", "\347\273\230\345\210\266\345\233\276\350\241\250", nullptr));
#endif // QT_NO_TOOLTIP
        tabWidget->setTabText(tabWidget->indexOf(tab_Data), QApplication::translate("MainWindow", "\346\225\260\346\215\256", nullptr));
        groupBox->setTitle(QApplication::translate("MainWindow", "\345\233\276\350\241\250", nullptr));
        label->setText(QApplication::translate("MainWindow", "\345\233\276\350\241\250\344\270\273\351\242\230", nullptr));
        comboTheme->setItemText(0, QApplication::translate("MainWindow", "Light", nullptr));
        comboTheme->setItemText(1, QApplication::translate("MainWindow", "BlueCerulean", nullptr));
        comboTheme->setItemText(2, QApplication::translate("MainWindow", "Dark", nullptr));
        comboTheme->setItemText(3, QApplication::translate("MainWindow", "BrownSand", nullptr));
        comboTheme->setItemText(4, QApplication::translate("MainWindow", "BlueNcs", nullptr));
        comboTheme->setItemText(5, QApplication::translate("MainWindow", "HighContrast", nullptr));
        comboTheme->setItemText(6, QApplication::translate("MainWindow", "BlueIcy", nullptr));
        comboTheme->setItemText(7, QApplication::translate("MainWindow", "Qt", nullptr));

        chkBox_Legend->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\345\233\276\344\276\213", nullptr));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "\350\234\241\347\203\233\345\233\276", nullptr));
        chkBox_Caps->setText(QApplication::translate("MainWindow", "capsVisible", nullptr));
        chkBox_Outline->setText(QApplication::translate("MainWindow", "bodyOutlineVisible", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "Y\350\275\264--QValueAxis", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "\346\234\200\345\260\217\345\200\274", nullptr));
        btnY_SetRange->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256\350\214\203\345\233\264", nullptr));
        label_6->setText(QApplication::translate("MainWindow", "\346\234\200\345\244\247\345\200\274", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "\345\210\206\345\272\246\344\270\252\346\225\260", nullptr));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "X\350\275\264--QDateTimeAxis", nullptr));
        label_4->setText(QApplication::translate("MainWindow", "\345\210\206\345\272\246\344\270\252\346\225\260", nullptr));
        label_7->setText(QApplication::translate("MainWindow", "\346\227\245\346\234\237\346\240\274\345\274\217", nullptr));
        comboDateFormat->setItemText(0, QApplication::translate("MainWindow", "MM-dd", nullptr));
        comboDateFormat->setItemText(1, QApplication::translate("MainWindow", "M-d", nullptr));
        comboDateFormat->setItemText(2, QApplication::translate("MainWindow", "M\346\234\210d\346\227\245", nullptr));
        comboDateFormat->setItemText(3, QApplication::translate("MainWindow", "yyyy-MM-dd", nullptr));

        btnY_SetFormat->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256\346\227\245\346\234\237\346\240\274\345\274\217", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_Setup), QApplication::translate("MainWindow", "\345\233\276\350\241\250\350\256\276\347\275\256", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
