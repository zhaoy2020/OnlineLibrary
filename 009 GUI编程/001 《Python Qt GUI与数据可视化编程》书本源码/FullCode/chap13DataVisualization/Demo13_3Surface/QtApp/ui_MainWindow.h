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
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actSurf_Color;
    QAction *actSurf_GradColor1;
    QAction *actSurf_GradColor2;
    QAction *actSurf_Texture;
    QAction *actSurf_ClearTexture;
    QAction *actClose;
    QWidget *centralWidget;
    QFrame *frameSetup;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout;
    QLabel *label_6;
    QComboBox *comboCamera;
    QLabel *label_7;
    QSlider *sliderH;
    QLabel *label_8;
    QSlider *sliderV;
    QLabel *label_12;
    QSlider *sliderZoom;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_3;
    QLabel *label;
    QComboBox *cBoxTheme;
    QLabel *label_4;
    QSpinBox *spinFontSize;
    QCheckBox *chkBoxBackground;
    QCheckBox *chkBoxGrid;
    QCheckBox *chkBoxShadow;
    QCheckBox *chkBoxAxisTitle;
    QCheckBox *chkBoxReverse;
    QCheckBox *chkBoxAxisBackground;
    QComboBox *cBoxSelectionMode;
    QLabel *label_3;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout_2;
    QLabel *label_5;
    QComboBox *comboDrawMode;
    QLabel *label_2;
    QComboBox *cBoxBarStyle;
    QCheckBox *chkBoxSmooth;
    QCheckBox *chkBoxFlatShading;
    QCheckBox *chkBoxItemLabel;
    QSpacerItem *verticalSpacer;
    QStatusBar *statusBar;
    QToolBar *toolBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(866, 557);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        actSurf_Color = new QAction(MainWindow);
        actSurf_Color->setObjectName(QStringLiteral("actSurf_Color"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/211.JPG"), QSize(), QIcon::Normal, QIcon::Off);
        actSurf_Color->setIcon(icon);
        actSurf_GradColor1 = new QAction(MainWindow);
        actSurf_GradColor1->setObjectName(QStringLiteral("actSurf_GradColor1"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/720.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actSurf_GradColor1->setIcon(icon1);
        actSurf_GradColor2 = new QAction(MainWindow);
        actSurf_GradColor2->setObjectName(QStringLiteral("actSurf_GradColor2"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/726.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actSurf_GradColor2->setIcon(icon2);
        actSurf_Texture = new QAction(MainWindow);
        actSurf_Texture->setObjectName(QStringLiteral("actSurf_Texture"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/156.GIF"), QSize(), QIcon::Normal, QIcon::Off);
        actSurf_Texture->setIcon(icon3);
        actSurf_ClearTexture = new QAction(MainWindow);
        actSurf_ClearTexture->setObjectName(QStringLiteral("actSurf_ClearTexture"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/212.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actSurf_ClearTexture->setIcon(icon4);
        actClose = new QAction(MainWindow);
        actClose->setObjectName(QStringLiteral("actClose"));
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actClose->setIcon(icon5);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        frameSetup = new QFrame(centralWidget);
        frameSetup->setObjectName(QStringLiteral("frameSetup"));
        frameSetup->setGeometry(QRect(10, 10, 281, 496));
        frameSetup->setFrameShape(QFrame::Panel);
        frameSetup->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frameSetup);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(5, 5, 5, 5);
        groupBox_2 = new QGroupBox(frameSetup);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        gridLayout = new QGridLayout(groupBox_2);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setHorizontalSpacing(8);
        gridLayout->setVerticalSpacing(10);
        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QStringLiteral("label_6"));

        gridLayout->addWidget(label_6, 0, 0, 1, 1);

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

        gridLayout->addWidget(comboCamera, 0, 1, 1, 1);

        label_7 = new QLabel(groupBox_2);
        label_7->setObjectName(QStringLiteral("label_7"));

        gridLayout->addWidget(label_7, 1, 0, 1, 1);

        sliderH = new QSlider(groupBox_2);
        sliderH->setObjectName(QStringLiteral("sliderH"));
        sliderH->setMinimum(-180);
        sliderH->setMaximum(180);
        sliderH->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(sliderH, 1, 1, 1, 1);

        label_8 = new QLabel(groupBox_2);
        label_8->setObjectName(QStringLiteral("label_8"));

        gridLayout->addWidget(label_8, 2, 0, 1, 1);

        sliderV = new QSlider(groupBox_2);
        sliderV->setObjectName(QStringLiteral("sliderV"));
        sliderV->setMinimum(-180);
        sliderV->setMaximum(180);
        sliderV->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(sliderV, 2, 1, 1, 1);

        label_12 = new QLabel(groupBox_2);
        label_12->setObjectName(QStringLiteral("label_12"));

        gridLayout->addWidget(label_12, 3, 0, 1, 1);

        sliderZoom = new QSlider(groupBox_2);
        sliderZoom->setObjectName(QStringLiteral("sliderZoom"));
        sliderZoom->setMinimum(10);
        sliderZoom->setMaximum(500);
        sliderZoom->setValue(100);
        sliderZoom->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(sliderZoom, 3, 1, 1, 1);


        verticalLayout->addWidget(groupBox_2);

        groupBox_3 = new QGroupBox(frameSetup);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        gridLayout_3 = new QGridLayout(groupBox_3);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        label = new QLabel(groupBox_3);
        label->setObjectName(QStringLiteral("label"));

        gridLayout_3->addWidget(label, 0, 0, 1, 1);

        cBoxTheme = new QComboBox(groupBox_3);
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->addItem(QString());
        cBoxTheme->setObjectName(QStringLiteral("cBoxTheme"));

        gridLayout_3->addWidget(cBoxTheme, 0, 1, 1, 2);

        label_4 = new QLabel(groupBox_3);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout_3->addWidget(label_4, 1, 0, 1, 1);

        spinFontSize = new QSpinBox(groupBox_3);
        spinFontSize->setObjectName(QStringLiteral("spinFontSize"));
        spinFontSize->setMinimum(5);
        spinFontSize->setValue(10);

        gridLayout_3->addWidget(spinFontSize, 1, 1, 1, 2);

        chkBoxBackground = new QCheckBox(groupBox_3);
        chkBoxBackground->setObjectName(QStringLiteral("chkBoxBackground"));
        chkBoxBackground->setChecked(true);

        gridLayout_3->addWidget(chkBoxBackground, 3, 0, 1, 1);

        chkBoxGrid = new QCheckBox(groupBox_3);
        chkBoxGrid->setObjectName(QStringLiteral("chkBoxGrid"));
        chkBoxGrid->setChecked(true);

        gridLayout_3->addWidget(chkBoxGrid, 3, 2, 1, 1);

        chkBoxShadow = new QCheckBox(groupBox_3);
        chkBoxShadow->setObjectName(QStringLiteral("chkBoxShadow"));
        chkBoxShadow->setChecked(true);

        gridLayout_3->addWidget(chkBoxShadow, 4, 0, 1, 1);

        chkBoxAxisTitle = new QCheckBox(groupBox_3);
        chkBoxAxisTitle->setObjectName(QStringLiteral("chkBoxAxisTitle"));
        chkBoxAxisTitle->setChecked(true);

        gridLayout_3->addWidget(chkBoxAxisTitle, 5, 0, 1, 2);

        chkBoxReverse = new QCheckBox(groupBox_3);
        chkBoxReverse->setObjectName(QStringLiteral("chkBoxReverse"));

        gridLayout_3->addWidget(chkBoxReverse, 4, 2, 1, 1);

        chkBoxAxisBackground = new QCheckBox(groupBox_3);
        chkBoxAxisBackground->setObjectName(QStringLiteral("chkBoxAxisBackground"));

        gridLayout_3->addWidget(chkBoxAxisBackground, 5, 2, 1, 1);

        cBoxSelectionMode = new QComboBox(groupBox_3);
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->addItem(QString());
        cBoxSelectionMode->setObjectName(QStringLiteral("cBoxSelectionMode"));

        gridLayout_3->addWidget(cBoxSelectionMode, 2, 1, 1, 2);

        label_3 = new QLabel(groupBox_3);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout_3->addWidget(label_3, 2, 0, 1, 1);


        verticalLayout->addWidget(groupBox_3);

        groupBox_4 = new QGroupBox(frameSetup);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        gridLayout_2 = new QGridLayout(groupBox_4);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        label_5 = new QLabel(groupBox_4);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout_2->addWidget(label_5, 0, 0, 1, 1);

        comboDrawMode = new QComboBox(groupBox_4);
        comboDrawMode->addItem(QString());
        comboDrawMode->addItem(QString());
        comboDrawMode->addItem(QString());
        comboDrawMode->setObjectName(QStringLiteral("comboDrawMode"));

        gridLayout_2->addWidget(comboDrawMode, 0, 1, 1, 2);

        label_2 = new QLabel(groupBox_4);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout_2->addWidget(label_2, 1, 0, 1, 1);

        cBoxBarStyle = new QComboBox(groupBox_4);
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

        gridLayout_2->addWidget(cBoxBarStyle, 1, 1, 1, 2);

        chkBoxSmooth = new QCheckBox(groupBox_4);
        chkBoxSmooth->setObjectName(QStringLiteral("chkBoxSmooth"));
        chkBoxSmooth->setChecked(true);

        gridLayout_2->addWidget(chkBoxSmooth, 3, 0, 1, 2);

        chkBoxFlatShading = new QCheckBox(groupBox_4);
        chkBoxFlatShading->setObjectName(QStringLiteral("chkBoxFlatShading"));
        chkBoxFlatShading->setChecked(true);

        gridLayout_2->addWidget(chkBoxFlatShading, 5, 0, 1, 2);

        chkBoxItemLabel = new QCheckBox(groupBox_4);
        chkBoxItemLabel->setObjectName(QStringLiteral("chkBoxItemLabel"));
        chkBoxItemLabel->setChecked(true);

        gridLayout_2->addWidget(chkBoxItemLabel, 3, 2, 1, 1);


        verticalLayout->addWidget(groupBox_4);

        verticalSpacer = new QSpacerItem(20, 22, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);

        MainWindow->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);
        toolBar = new QToolBar(MainWindow);
        toolBar->setObjectName(QStringLiteral("toolBar"));
        toolBar->setIconSize(QSize(16, 16));
        toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, toolBar);
#ifndef QT_NO_SHORTCUT
        label_6->setBuddy(comboCamera);
        label_7->setBuddy(sliderH);
        label_8->setBuddy(sliderV);
        label->setBuddy(cBoxTheme);
        label_4->setBuddy(spinFontSize);
        label_3->setBuddy(cBoxSelectionMode);
        label_2->setBuddy(cBoxBarStyle);
#endif // QT_NO_SHORTCUT

        toolBar->addAction(actSurf_Color);
        toolBar->addAction(actSurf_GradColor1);
        toolBar->addAction(actSurf_GradColor2);
        toolBar->addSeparator();
        toolBar->addAction(actSurf_Texture);
        toolBar->addAction(actSurf_ClearTexture);
        toolBar->addSeparator();
        toolBar->addAction(actClose);

        retranslateUi(MainWindow);
        QObject::connect(actClose, SIGNAL(triggered()), MainWindow, SLOT(close()));

        comboCamera->setCurrentIndex(0);
        cBoxSelectionMode->setCurrentIndex(1);
        comboDrawMode->setCurrentIndex(2);
        cBoxBarStyle->setCurrentIndex(7);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Q3DSurface\347\273\230\345\233\276", nullptr));
        actSurf_Color->setText(QApplication::translate("MainWindow", "\346\233\262\351\235\242\351\242\234\350\211\262", nullptr));
#ifndef QT_NO_TOOLTIP
        actSurf_Color->setToolTip(QApplication::translate("MainWindow", "\350\256\276\347\275\256\346\233\262\351\235\242\351\242\234\350\211\262", nullptr));
#endif // QT_NO_TOOLTIP
        actSurf_GradColor1->setText(QApplication::translate("MainWindow", "\346\270\220\345\217\230\350\211\262\344\270\200", nullptr));
#ifndef QT_NO_TOOLTIP
        actSurf_GradColor1->setToolTip(QApplication::translate("MainWindow", "\344\275\277\347\224\250\346\270\220\345\217\230\350\211\262\344\270\200", nullptr));
#endif // QT_NO_TOOLTIP
        actSurf_GradColor2->setText(QApplication::translate("MainWindow", "\346\270\220\345\217\230\350\211\262\344\272\214", nullptr));
#ifndef QT_NO_TOOLTIP
        actSurf_GradColor2->setToolTip(QApplication::translate("MainWindow", "\344\275\277\347\224\250\346\270\220\345\217\230\350\211\262\344\272\214", nullptr));
#endif // QT_NO_TOOLTIP
        actSurf_Texture->setText(QApplication::translate("MainWindow", "\344\275\277\347\224\250\346\235\220\350\264\250\345\233\276\347\211\207", nullptr));
#ifndef QT_NO_TOOLTIP
        actSurf_Texture->setToolTip(QApplication::translate("MainWindow", "\344\275\277\347\224\250\346\235\220\350\264\250\345\233\276\347\211\207", nullptr));
#endif // QT_NO_TOOLTIP
        actSurf_ClearTexture->setText(QApplication::translate("MainWindow", "\346\270\205\351\231\244\346\235\220\350\264\250", nullptr));
#ifndef QT_NO_TOOLTIP
        actSurf_ClearTexture->setToolTip(QApplication::translate("MainWindow", "\346\270\205\351\231\244\346\235\220\350\264\250", nullptr));
#endif // QT_NO_TOOLTIP
        actClose->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actClose->setToolTip(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#endif // QT_NO_TOOLTIP
        groupBox_2->setTitle(QApplication::translate("MainWindow", "\344\270\211\347\273\264\346\227\213\350\275\254", nullptr));
        label_6->setText(QApplication::translate("MainWindow", "\351\242\204\350\256\276\350\247\206\350\247\222", nullptr));
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

        label_7->setText(QApplication::translate("MainWindow", "\346\260\264\345\271\263\346\227\213\350\275\254", nullptr));
        label_8->setText(QApplication::translate("MainWindow", "\345\236\202\347\233\264\346\227\213\350\275\254", nullptr));
        label_12->setText(QApplication::translate("MainWindow", "\347\274\251  \346\224\276", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "\345\233\276\350\241\250\346\200\273\344\275\223", nullptr));
        label->setText(QApplication::translate("MainWindow", "\344\270\273  \351\242\230", nullptr));
        cBoxTheme->setItemText(0, QApplication::translate("MainWindow", "Qt", nullptr));
        cBoxTheme->setItemText(1, QApplication::translate("MainWindow", "Primary Colors", nullptr));
        cBoxTheme->setItemText(2, QApplication::translate("MainWindow", "Digia", nullptr));
        cBoxTheme->setItemText(3, QApplication::translate("MainWindow", "StoneMoss", nullptr));
        cBoxTheme->setItemText(4, QApplication::translate("MainWindow", "ArmyBlue", nullptr));
        cBoxTheme->setItemText(5, QApplication::translate("MainWindow", "Retro", nullptr));
        cBoxTheme->setItemText(6, QApplication::translate("MainWindow", "Ebony", nullptr));
        cBoxTheme->setItemText(7, QApplication::translate("MainWindow", "Isabelle", nullptr));

        label_4->setText(QApplication::translate("MainWindow", "\350\275\264\345\255\227\344\275\223\345\244\247\345\260\217", nullptr));
        chkBoxBackground->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\350\203\214\346\231\257", nullptr));
        chkBoxGrid->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\350\203\214\346\231\257\347\275\221\346\240\274", nullptr));
        chkBoxShadow->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\351\230\264\345\275\261", nullptr));
        chkBoxAxisTitle->setText(QApplication::translate("MainWindow", "\350\275\264\346\240\207\351\242\230\345\217\257\350\247\201", nullptr));
        chkBoxReverse->setText(QApplication::translate("MainWindow", "Y\350\275\264(\345\236\202\347\233\264\346\226\271\345\220\221)\345\217\215\345\220\221", nullptr));
        chkBoxAxisBackground->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\350\275\264\346\240\207\347\255\276\350\203\214\346\231\257", nullptr));
        cBoxSelectionMode->setItemText(0, QApplication::translate("MainWindow", "None", nullptr));
        cBoxSelectionMode->setItemText(1, QApplication::translate("MainWindow", "Item", nullptr));
        cBoxSelectionMode->setItemText(2, QApplication::translate("MainWindow", "Row Slice", nullptr));
        cBoxSelectionMode->setItemText(3, QApplication::translate("MainWindow", "Column Slice", nullptr));

        label_3->setText(QApplication::translate("MainWindow", "\351\200\211\346\213\251\346\250\241\345\274\217", nullptr));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "\346\233\262\351\235\242\345\272\217\345\210\227", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "\346\233\262\351\235\242\346\240\267\345\274\217", nullptr));
        comboDrawMode->setItemText(0, QApplication::translate("MainWindow", "Wireframe", nullptr));
        comboDrawMode->setItemText(1, QApplication::translate("MainWindow", "Surface", nullptr));
        comboDrawMode->setItemText(2, QApplication::translate("MainWindow", "Surface And Wireframe", nullptr));

        label_2->setText(QApplication::translate("MainWindow", "\345\215\225\347\202\271\345\275\242\347\212\266", nullptr));
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

        chkBoxSmooth->setText(QApplication::translate("MainWindow", "\345\215\225\347\202\271\345\205\211\346\273\221\346\225\210\346\236\234", nullptr));
        chkBoxFlatShading->setText(QApplication::translate("MainWindow", "FlatShading", nullptr));
        chkBoxItemLabel->setText(QApplication::translate("MainWindow", "\345\215\225\347\202\271\346\240\207\347\255\276\345\217\257\350\247\201", nullptr));
        toolBar->setWindowTitle(QApplication::translate("MainWindow", "toolBar", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
