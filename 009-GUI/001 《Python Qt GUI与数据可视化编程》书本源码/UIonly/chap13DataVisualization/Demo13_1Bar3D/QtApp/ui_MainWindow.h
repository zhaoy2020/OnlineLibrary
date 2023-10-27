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
#include <QtGui/QIcon>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actBar_ChangeValue;
    QAction *actClose;
    QAction *actSeries_BaseColor;
    QAction *actData_Add;
    QAction *actData_Insert;
    QAction *actData_Delete;
    QWidget *centralWidget;
    QFrame *frameSetup;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_3;
    QLabel *label_5;
    QComboBox *comboCamera;
    QLabel *label_6;
    QSlider *sliderH;
    QFrame *frame;
    QGridLayout *gridLayout_4;
    QToolButton *btnMoveUp;
    QToolButton *btnMoveLeft;
    QToolButton *btnMoveRight;
    QToolButton *btnMoveDown;
    QToolButton *btnResetCamera;
    QLabel *label_7;
    QSlider *sliderV;
    QLabel *label_8;
    QSlider *sliderZoom;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout_2;
    QLabel *label;
    QComboBox *cBoxTheme;
    QLabel *label_4;
    QSpinBox *spinFontSize;
    QLabel *label_3;
    QCheckBox *chkBoxBackground;
    QCheckBox *chkBoxGrid;
    QCheckBox *chkBoxAxisTitle;
    QCheckBox *chkBoxAxisBackground;
    QCheckBox *chkBoxReflection;
    QCheckBox *chkBoxReverse;
    QComboBox *cBoxSelectionMode;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout;
    QLabel *label_2;
    QComboBox *cBoxBarStyle;
    QCheckBox *chkBoxSmooth;
    QCheckBox *chkBoxItemLabel;
    QSpacerItem *verticalSpacer;
    QStatusBar *statusBar;
    QToolBar *toolBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(697, 521);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        actBar_ChangeValue = new QAction(MainWindow);
        actBar_ChangeValue->setObjectName(QStringLiteral("actBar_ChangeValue"));
        actBar_ChangeValue->setEnabled(false);
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/810.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actBar_ChangeValue->setIcon(icon);
        actClose = new QAction(MainWindow);
        actClose->setObjectName(QStringLiteral("actClose"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actClose->setIcon(icon1);
        actSeries_BaseColor = new QAction(MainWindow);
        actSeries_BaseColor->setObjectName(QStringLiteral("actSeries_BaseColor"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/720.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actSeries_BaseColor->setIcon(icon2);
        actData_Add = new QAction(MainWindow);
        actData_Add->setObjectName(QStringLiteral("actData_Add"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/316.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actData_Add->setIcon(icon3);
        actData_Insert = new QAction(MainWindow);
        actData_Insert->setObjectName(QStringLiteral("actData_Insert"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/306.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actData_Insert->setIcon(icon4);
        actData_Delete = new QAction(MainWindow);
        actData_Delete->setObjectName(QStringLiteral("actData_Delete"));
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/icons/images/318.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actData_Delete->setIcon(icon5);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        frameSetup = new QFrame(centralWidget);
        frameSetup->setObjectName(QStringLiteral("frameSetup"));
        frameSetup->setGeometry(QRect(10, 10, 276, 446));
        frameSetup->setFrameShape(QFrame::Panel);
        frameSetup->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frameSetup);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(5, 5, 5, 5);
        groupBox_2 = new QGroupBox(frameSetup);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        gridLayout_3 = new QGridLayout(groupBox_2);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout_3->addWidget(label_5, 0, 0, 1, 1);

        comboCamera = new QComboBox(groupBox_2);
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->addItem(QString());
        comboCamera->setObjectName(QStringLiteral("comboCamera"));

        gridLayout_3->addWidget(comboCamera, 0, 1, 1, 2);

        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QStringLiteral("label_6"));

        gridLayout_3->addWidget(label_6, 1, 0, 1, 1);

        sliderH = new QSlider(groupBox_2);
        sliderH->setObjectName(QStringLiteral("sliderH"));
        sliderH->setMinimum(-180);
        sliderH->setMaximum(180);
        sliderH->setOrientation(Qt::Horizontal);

        gridLayout_3->addWidget(sliderH, 1, 1, 1, 1);

        frame = new QFrame(groupBox_2);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_4 = new QGridLayout(frame);
        gridLayout_4->setSpacing(5);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setContentsMargins(0, 0, 0, 0);
        btnMoveUp = new QToolButton(frame);
        btnMoveUp->setObjectName(QStringLiteral("btnMoveUp"));
        btnMoveUp->setArrowType(Qt::UpArrow);

        gridLayout_4->addWidget(btnMoveUp, 0, 1, 1, 1);

        btnMoveLeft = new QToolButton(frame);
        btnMoveLeft->setObjectName(QStringLiteral("btnMoveLeft"));
        btnMoveLeft->setArrowType(Qt::LeftArrow);

        gridLayout_4->addWidget(btnMoveLeft, 1, 0, 1, 1);

        btnMoveRight = new QToolButton(frame);
        btnMoveRight->setObjectName(QStringLiteral("btnMoveRight"));
        btnMoveRight->setArrowType(Qt::RightArrow);

        gridLayout_4->addWidget(btnMoveRight, 1, 2, 1, 1);

        btnMoveDown = new QToolButton(frame);
        btnMoveDown->setObjectName(QStringLiteral("btnMoveDown"));
        btnMoveDown->setArrowType(Qt::DownArrow);

        gridLayout_4->addWidget(btnMoveDown, 2, 1, 1, 1);

        btnResetCamera = new QToolButton(frame);
        btnResetCamera->setObjectName(QStringLiteral("btnResetCamera"));
        QIcon icon6;
        icon6.addFile(QStringLiteral(":/icons/images/326.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnResetCamera->setIcon(icon6);

        gridLayout_4->addWidget(btnResetCamera, 1, 1, 1, 1);


        gridLayout_3->addWidget(frame, 1, 2, 3, 1);

        label_7 = new QLabel(groupBox_2);
        label_7->setObjectName(QStringLiteral("label_7"));

        gridLayout_3->addWidget(label_7, 2, 0, 1, 1);

        sliderV = new QSlider(groupBox_2);
        sliderV->setObjectName(QStringLiteral("sliderV"));
        sliderV->setMinimum(-180);
        sliderV->setMaximum(180);
        sliderV->setOrientation(Qt::Horizontal);

        gridLayout_3->addWidget(sliderV, 2, 1, 1, 1);

        label_8 = new QLabel(groupBox_2);
        label_8->setObjectName(QStringLiteral("label_8"));

        gridLayout_3->addWidget(label_8, 3, 0, 1, 1);

        sliderZoom = new QSlider(groupBox_2);
        sliderZoom->setObjectName(QStringLiteral("sliderZoom"));
        sliderZoom->setMinimum(10);
        sliderZoom->setMaximum(500);
        sliderZoom->setValue(100);
        sliderZoom->setOrientation(Qt::Horizontal);

        gridLayout_3->addWidget(sliderZoom, 3, 1, 1, 1);


        verticalLayout->addWidget(groupBox_2);

        groupBox_4 = new QGroupBox(frameSetup);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        gridLayout_2 = new QGridLayout(groupBox_4);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setVerticalSpacing(10);
        label = new QLabel(groupBox_4);
        label->setObjectName(QStringLiteral("label"));
        label->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label->setIndent(10);

        gridLayout_2->addWidget(label, 0, 0, 1, 1);

        cBoxTheme = new QComboBox(groupBox_4);
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->setObjectName(QStringLiteral("cBoxTheme"));

        gridLayout_2->addWidget(cBoxTheme, 0, 1, 1, 2);

        label_4 = new QLabel(groupBox_4);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_4->setIndent(10);

        gridLayout_2->addWidget(label_4, 1, 0, 1, 1);

        spinFontSize = new QSpinBox(groupBox_4);
        spinFontSize->setObjectName(QStringLiteral("spinFontSize"));
        spinFontSize->setMinimum(5);
        spinFontSize->setValue(10);

        gridLayout_2->addWidget(spinFontSize, 1, 1, 1, 2);

        label_3 = new QLabel(groupBox_4);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_3->setIndent(10);

        gridLayout_2->addWidget(label_3, 2, 0, 1, 1);

        chkBoxBackground = new QCheckBox(groupBox_4);
        chkBoxBackground->setObjectName(QStringLiteral("chkBoxBackground"));
        chkBoxBackground->setChecked(true);

        gridLayout_2->addWidget(chkBoxBackground, 3, 0, 1, 1);

        chkBoxGrid = new QCheckBox(groupBox_4);
        chkBoxGrid->setObjectName(QStringLiteral("chkBoxGrid"));
        chkBoxGrid->setChecked(true);

        gridLayout_2->addWidget(chkBoxGrid, 3, 2, 1, 1);

        chkBoxAxisTitle = new QCheckBox(groupBox_4);
        chkBoxAxisTitle->setObjectName(QStringLiteral("chkBoxAxisTitle"));
        chkBoxAxisTitle->setChecked(true);

        gridLayout_2->addWidget(chkBoxAxisTitle, 6, 0, 1, 2);

        chkBoxAxisBackground = new QCheckBox(groupBox_4);
        chkBoxAxisBackground->setObjectName(QStringLiteral("chkBoxAxisBackground"));
        chkBoxAxisBackground->setChecked(false);

        gridLayout_2->addWidget(chkBoxAxisBackground, 6, 2, 1, 1);

        chkBoxReflection = new QCheckBox(groupBox_4);
        chkBoxReflection->setObjectName(QStringLiteral("chkBoxReflection"));

        gridLayout_2->addWidget(chkBoxReflection, 5, 0, 1, 1);

        chkBoxReverse = new QCheckBox(groupBox_4);
        chkBoxReverse->setObjectName(QStringLiteral("chkBoxReverse"));

        gridLayout_2->addWidget(chkBoxReverse, 5, 2, 1, 1);

        cBoxSelectionMode = new QComboBox(groupBox_4);
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->setObjectName(QStringLiteral("cBoxSelectionMode"));

        gridLayout_2->addWidget(cBoxSelectionMode, 2, 1, 1, 2);


        verticalLayout->addWidget(groupBox_4);

        groupBox_3 = new QGroupBox(frameSetup);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        gridLayout = new QGridLayout(groupBox_3);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setVerticalSpacing(10);
        label_2 = new QLabel(groupBox_3);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        label_2->setIndent(10);

        gridLayout->addWidget(label_2, 0, 0, 1, 1);

        cBoxBarStyle = new QComboBox(groupBox_3);
        cBoxBarStyle->addItem(QString());
        cBoxBarStyle->addItem(QString());
        cBoxBarStyle->addItem(QString());
        cBoxBarStyle->addItem(QString());
        cBoxBarStyle->addItem(QString());
        cBoxBarStyle->addItem(QString());
        cBoxBarStyle->addItem(QString());
        cBoxBarStyle->addItem(QString());
        cBoxBarStyle->addItem(QString());
        cBoxBarStyle->addItem(QString());
        cBoxBarStyle->addItem(QString());
        cBoxBarStyle->setObjectName(QStringLiteral("cBoxBarStyle"));

        gridLayout->addWidget(cBoxBarStyle, 0, 1, 1, 1);

        chkBoxSmooth = new QCheckBox(groupBox_3);
        chkBoxSmooth->setObjectName(QStringLiteral("chkBoxSmooth"));

        gridLayout->addWidget(chkBoxSmooth, 1, 0, 1, 1);

        chkBoxItemLabel = new QCheckBox(groupBox_3);
        chkBoxItemLabel->setObjectName(QStringLiteral("chkBoxItemLabel"));
        chkBoxItemLabel->setChecked(true);

        gridLayout->addWidget(chkBoxItemLabel, 1, 1, 1, 1);


        verticalLayout->addWidget(groupBox_3);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);

        MainWindow->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);
        toolBar = new QToolBar(MainWindow);
        toolBar->setObjectName(QStringLiteral("toolBar"));
        toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, toolBar);
#ifndef QT_NO_SHORTCUT
        label_5->setBuddy(comboCamera);
        label_6->setBuddy(sliderH);
        label_7->setBuddy(sliderV);
        label_8->setBuddy(sliderZoom);
        label->setBuddy(cBoxTheme);
        label_4->setBuddy(spinFontSize);
        label_2->setBuddy(cBoxBarStyle);
#endif // QT_NO_SHORTCUT

        toolBar->addAction(actSeries_BaseColor);
        toolBar->addAction(actBar_ChangeValue);
        toolBar->addSeparator();
        toolBar->addAction(actData_Add);
        toolBar->addAction(actData_Insert);
        toolBar->addAction(actData_Delete);
        toolBar->addSeparator();
        toolBar->addAction(actClose);

        retranslateUi(MainWindow);
        QObject::connect(actClose, SIGNAL(triggered()), MainWindow, SLOT(close()));

        comboCamera->setCurrentIndex(2);
        cBoxSelectionMode->setCurrentIndex(1);
        cBoxBarStyle->setCurrentIndex(4);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Q3DBars\347\273\230\345\233\276", nullptr));
        actBar_ChangeValue->setText(QApplication::translate("MainWindow", "\344\277\256\346\224\271\346\225\260\345\200\274", nullptr));
#ifndef QT_NO_TOOLTIP
        actBar_ChangeValue->setToolTip(QApplication::translate("MainWindow", "\344\277\256\346\224\271\351\200\211\344\270\255\346\243\222\346\237\261\347\232\204\346\225\260\345\200\274(F2)", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actBar_ChangeValue->setShortcut(QApplication::translate("MainWindow", "F2", nullptr));
#endif // QT_NO_SHORTCUT
        actClose->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actClose->setToolTip(QApplication::translate("MainWindow", "\351\200\200\345\207\272\346\234\254\347\250\213\345\272\217", nullptr));
#endif // QT_NO_TOOLTIP
        actSeries_BaseColor->setText(QApplication::translate("MainWindow", "\345\272\217\345\210\227\345\237\272\346\234\254\351\242\234\350\211\262", nullptr));
#ifndef QT_NO_TOOLTIP
        actSeries_BaseColor->setToolTip(QApplication::translate("MainWindow", "\350\256\276\347\275\256\345\272\217\345\210\227\345\237\272\346\234\254\351\242\234\350\211\262", nullptr));
#endif // QT_NO_TOOLTIP
        actData_Add->setText(QApplication::translate("MainWindow", "\346\267\273\345\212\240\350\241\214", nullptr));
#ifndef QT_NO_TOOLTIP
        actData_Add->setToolTip(QApplication::translate("MainWindow", "\346\267\273\345\212\240\344\270\200\350\241\214", nullptr));
#endif // QT_NO_TOOLTIP
        actData_Insert->setText(QApplication::translate("MainWindow", "\346\217\222\345\205\245\350\241\214", nullptr));
#ifndef QT_NO_TOOLTIP
        actData_Insert->setToolTip(QApplication::translate("MainWindow", "\346\217\222\345\205\245\344\270\200\350\241\214", nullptr));
#endif // QT_NO_TOOLTIP
        actData_Delete->setText(QApplication::translate("MainWindow", "\345\210\240\351\231\244\350\241\214", nullptr));
#ifndef QT_NO_TOOLTIP
        actData_Delete->setToolTip(QApplication::translate("MainWindow", "\345\210\240\351\231\244\345\275\223\345\211\215\350\241\214", nullptr));
#endif // QT_NO_TOOLTIP
        groupBox_2->setTitle(QApplication::translate("MainWindow", "\346\227\213\350\275\254\345\222\214\345\271\263\347\247\273", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "\351\242\204\350\256\276\350\247\206\350\247\222", nullptr));
        comboCamera->setItemText(0, QApplication::translate("MainWindow", "FrontLow", nullptr));
        comboCamera->setItemText(1, QApplication::translate("MainWindow", "Front", nullptr));
        comboCamera->setItemText(2, QApplication::translate("MainWindow", "FrontHigh", nullptr));
        comboCamera->setItemText(3, QApplication::translate("MainWindow", "LeftLow", nullptr));
        comboCamera->setItemText(4, QApplication::translate("MainWindow", "Left", nullptr));
        comboCamera->setItemText(5, QApplication::translate("MainWindow", "LeftHigh", nullptr));
        comboCamera->setItemText(6, QApplication::translate("MainWindow", "RightLow", nullptr));
        comboCamera->setItemText(7, QApplication::translate("MainWindow", "Right", nullptr));
        comboCamera->setItemText(8, QApplication::translate("MainWindow", "RightHigh", nullptr));
        comboCamera->setItemText(9, QApplication::translate("MainWindow", "BehindLow", nullptr));
        comboCamera->setItemText(10, QApplication::translate("MainWindow", "Behind", nullptr));
        comboCamera->setItemText(11, QApplication::translate("MainWindow", "BehindHigh", nullptr));
        comboCamera->setItemText(12, QApplication::translate("MainWindow", "IsometricLeft", nullptr));
        comboCamera->setItemText(13, QApplication::translate("MainWindow", "IsometricLeftHigh", nullptr));
        comboCamera->setItemText(14, QApplication::translate("MainWindow", "IsometricRight", nullptr));
        comboCamera->setItemText(15, QApplication::translate("MainWindow", "IsometricRightHigh", nullptr));
        comboCamera->setItemText(16, QApplication::translate("MainWindow", "DirectlyAbove", nullptr));
        comboCamera->setItemText(17, QApplication::translate("MainWindow", "DirectlyAboveCW45", nullptr));
        comboCamera->setItemText(18, QApplication::translate("MainWindow", "DirectlyAboveCCW45", nullptr));
        comboCamera->setItemText(19, QApplication::translate("MainWindow", "FrontBelow", nullptr));
        comboCamera->setItemText(20, QApplication::translate("MainWindow", "LeftBelow", nullptr));
        comboCamera->setItemText(21, QApplication::translate("MainWindow", "RightBelow", nullptr));
        comboCamera->setItemText(22, QApplication::translate("MainWindow", "BehindBelow", nullptr));
        comboCamera->setItemText(23, QApplication::translate("MainWindow", "DirectlyBelow", nullptr));

        label_6->setText(QApplication::translate("MainWindow", "\346\260\264\345\271\263\346\227\213\350\275\254", nullptr));
        btnMoveUp->setText(QString());
        btnMoveLeft->setText(QString());
        btnMoveRight->setText(QString());
        btnMoveDown->setText(QString());
#ifndef QT_NO_TOOLTIP
        btnResetCamera->setToolTip(QApplication::translate("MainWindow", "\345\244\215\344\275\215\345\210\260FrontHigh\350\247\206\350\247\222", nullptr));
#endif // QT_NO_TOOLTIP
        btnResetCamera->setText(QString());
        label_7->setText(QApplication::translate("MainWindow", "\345\236\202\347\233\264\346\227\213\350\275\254", nullptr));
        label_8->setText(QApplication::translate("MainWindow", "\347\274\251  \346\224\276", nullptr));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "\345\233\276\350\241\250\346\200\273\344\275\223", nullptr));
        label->setText(QApplication::translate("MainWindow", "\345\233\276\350\241\250\344\270\273\351\242\230", nullptr));
        cBoxTheme->setItemText(0, QApplication::translate("MainWindow", "Qt", nullptr));
        cBoxTheme->setItemText(1, QApplication::translate("MainWindow", "Primary Colors", nullptr));
        cBoxTheme->setItemText(2, QApplication::translate("MainWindow", "Digia", nullptr));
        cBoxTheme->setItemText(3, QApplication::translate("MainWindow", "StoneMoss", nullptr));
        cBoxTheme->setItemText(4, QApplication::translate("MainWindow", "ArmyBlue", nullptr));
        cBoxTheme->setItemText(5, QApplication::translate("MainWindow", "Retro", nullptr));
        cBoxTheme->setItemText(6, QApplication::translate("MainWindow", "Ebony", nullptr));
        cBoxTheme->setItemText(7, QApplication::translate("MainWindow", "Isabelle", nullptr));

        label_4->setText(QApplication::translate("MainWindow", "\345\255\227\344\275\223\345\244\247\345\260\217", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "\351\200\211\346\213\251\346\250\241\345\274\217", nullptr));
        chkBoxBackground->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\350\203\214\346\231\257", nullptr));
        chkBoxGrid->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\350\203\214\346\231\257\347\275\221\346\240\274", nullptr));
        chkBoxAxisTitle->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\350\275\264\346\240\207\351\242\230", nullptr));
        chkBoxAxisBackground->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\350\275\264\346\240\207\347\255\276\350\203\214\346\231\257", nullptr));
        chkBoxReflection->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\345\200\222\345\275\261", nullptr));
        chkBoxReverse->setText(QApplication::translate("MainWindow", "\346\225\260\345\200\274\345\235\220\346\240\207\350\275\264\345\217\215\345\220\221", nullptr));
        cBoxSelectionMode->setItemText(0, QApplication::translate("MainWindow", "None", nullptr));
        cBoxSelectionMode->setItemText(1, QApplication::translate("MainWindow", "Item", nullptr));
        cBoxSelectionMode->setItemText(2, QApplication::translate("MainWindow", "Row", nullptr));
        cBoxSelectionMode->setItemText(3, QApplication::translate("MainWindow", "Item And Row", nullptr));
        cBoxSelectionMode->setItemText(4, QApplication::translate("MainWindow", "Column", nullptr));
        cBoxSelectionMode->setItemText(5, QApplication::translate("MainWindow", "Item And Column", nullptr));
        cBoxSelectionMode->setItemText(6, QApplication::translate("MainWindow", "Row And Column", nullptr));
        cBoxSelectionMode->setItemText(7, QApplication::translate("MainWindow", "Item Row And Column", nullptr));
        cBoxSelectionMode->setItemText(8, QApplication::translate("MainWindow", "Row Slice", nullptr));
        cBoxSelectionMode->setItemText(9, QApplication::translate("MainWindow", "Column Slice", nullptr));

        groupBox_3->setTitle(QApplication::translate("MainWindow", "\345\272\217\345\210\227\350\256\276\347\275\256", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "\346\243\222\346\237\261\346\240\267\345\274\217", nullptr));
        cBoxBarStyle->setItemText(0, QApplication::translate("MainWindow", "Bar", nullptr));
        cBoxBarStyle->setItemText(1, QApplication::translate("MainWindow", "Cube", nullptr));
        cBoxBarStyle->setItemText(2, QApplication::translate("MainWindow", "Pyramid", nullptr));
        cBoxBarStyle->setItemText(3, QApplication::translate("MainWindow", "Cone", nullptr));
        cBoxBarStyle->setItemText(4, QApplication::translate("MainWindow", "Cylinder", nullptr));
        cBoxBarStyle->setItemText(5, QApplication::translate("MainWindow", "BevelBar", nullptr));
        cBoxBarStyle->setItemText(6, QApplication::translate("MainWindow", "BevelCube", nullptr));
        cBoxBarStyle->setItemText(7, QApplication::translate("MainWindow", "Sphere", nullptr));
        cBoxBarStyle->setItemText(8, QApplication::translate("MainWindow", "Minimal", nullptr));
        cBoxBarStyle->setItemText(9, QApplication::translate("MainWindow", "Arrow", nullptr));
        cBoxBarStyle->setItemText(10, QApplication::translate("MainWindow", "Point", nullptr));

        chkBoxSmooth->setText(QApplication::translate("MainWindow", "\345\205\211\346\273\221\346\225\210\346\236\234", nullptr));
        chkBoxItemLabel->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\351\200\211\344\270\255\346\243\222\346\237\261\347\232\204\346\240\207\347\255\276", nullptr));
        toolBar->setWindowTitle(QApplication::translate("MainWindow", "toolBar", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
