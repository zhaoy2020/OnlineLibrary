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
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
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
    QLabel *label_9;
    QSlider *sliderZoom;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_2;
    QLabel *label;
    QComboBox *cBoxTheme;
    QLabel *label_4;
    QSpinBox *spinFontSize;
    QLabel *label_3;
    QComboBox *cBoxSelectionMode;
    QCheckBox *chkBoxBackground;
    QCheckBox *chkBoxGrid;
    QCheckBox *chkBoxShadow;
    QCheckBox *chkBoxAxisBackground;
    QCheckBox *chkBoxReflection;
    QCheckBox *chkBoxAxisTitle;
    QCheckBox *chkBoxReverse;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout_3;
    QLabel *label_2;
    QComboBox *cBoxBarStyle;
    QLabel *label_5;
    QDoubleSpinBox *spinItemSize;
    QPushButton *btnBarColor;
    QCheckBox *chkBoxSmooth;
    QCheckBox *chkBoxItemLabel;
    QSpacerItem *verticalSpacer;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(944, 626);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        frameSetup = new QFrame(centralWidget);
        frameSetup->setObjectName(QStringLiteral("frameSetup"));
        frameSetup->setGeometry(QRect(10, 15, 304, 577));
        frameSetup->setFrameShape(QFrame::Panel);
        frameSetup->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frameSetup);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        groupBox_2 = new QGroupBox(frameSetup);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        gridLayout = new QGridLayout(groupBox_2);
        gridLayout->setSpacing(10);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
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

        label_9 = new QLabel(groupBox_2);
        label_9->setObjectName(QStringLiteral("label_9"));

        gridLayout->addWidget(label_9, 3, 0, 1, 1);

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
        gridLayout_2 = new QGridLayout(groupBox_3);
        gridLayout_2->setSpacing(10);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        label = new QLabel(groupBox_3);
        label->setObjectName(QStringLiteral("label"));

        gridLayout_2->addWidget(label, 0, 0, 1, 1);

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

        gridLayout_2->addWidget(cBoxTheme, 0, 1, 1, 1);

        label_4 = new QLabel(groupBox_3);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout_2->addWidget(label_4, 1, 0, 1, 1);

        spinFontSize = new QSpinBox(groupBox_3);
        spinFontSize->setObjectName(QStringLiteral("spinFontSize"));
        spinFontSize->setMinimum(5);
        spinFontSize->setValue(10);

        gridLayout_2->addWidget(spinFontSize, 1, 1, 1, 1);

        label_3 = new QLabel(groupBox_3);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout_2->addWidget(label_3, 2, 0, 1, 1);

        cBoxSelectionMode = new QComboBox(groupBox_3);
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

        gridLayout_2->addWidget(cBoxSelectionMode, 2, 1, 1, 1);

        chkBoxBackground = new QCheckBox(groupBox_3);
        chkBoxBackground->setObjectName(QStringLiteral("chkBoxBackground"));

        gridLayout_2->addWidget(chkBoxBackground, 3, 0, 1, 1);

        chkBoxGrid = new QCheckBox(groupBox_3);
        chkBoxGrid->setObjectName(QStringLiteral("chkBoxGrid"));

        gridLayout_2->addWidget(chkBoxGrid, 3, 1, 1, 1);

        chkBoxShadow = new QCheckBox(groupBox_3);
        chkBoxShadow->setObjectName(QStringLiteral("chkBoxShadow"));

        gridLayout_2->addWidget(chkBoxShadow, 4, 0, 1, 1);

        chkBoxAxisBackground = new QCheckBox(groupBox_3);
        chkBoxAxisBackground->setObjectName(QStringLiteral("chkBoxAxisBackground"));

        gridLayout_2->addWidget(chkBoxAxisBackground, 4, 1, 1, 1);

        chkBoxReflection = new QCheckBox(groupBox_3);
        chkBoxReflection->setObjectName(QStringLiteral("chkBoxReflection"));

        gridLayout_2->addWidget(chkBoxReflection, 5, 0, 1, 1);

        chkBoxAxisTitle = new QCheckBox(groupBox_3);
        chkBoxAxisTitle->setObjectName(QStringLiteral("chkBoxAxisTitle"));

        gridLayout_2->addWidget(chkBoxAxisTitle, 5, 1, 1, 1);

        chkBoxReverse = new QCheckBox(groupBox_3);
        chkBoxReverse->setObjectName(QStringLiteral("chkBoxReverse"));

        gridLayout_2->addWidget(chkBoxReverse, 6, 1, 1, 1);


        verticalLayout->addWidget(groupBox_3);

        groupBox_4 = new QGroupBox(frameSetup);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        gridLayout_3 = new QGridLayout(groupBox_4);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        label_2 = new QLabel(groupBox_4);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout_3->addWidget(label_2, 0, 0, 1, 1);

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

        gridLayout_3->addWidget(cBoxBarStyle, 0, 1, 1, 2);

        label_5 = new QLabel(groupBox_4);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout_3->addWidget(label_5, 1, 0, 1, 1);

        spinItemSize = new QDoubleSpinBox(groupBox_4);
        spinItemSize->setObjectName(QStringLiteral("spinItemSize"));
        spinItemSize->setMaximum(1);
        spinItemSize->setSingleStep(0.02);

        gridLayout_3->addWidget(spinItemSize, 1, 1, 1, 2);

        btnBarColor = new QPushButton(groupBox_4);
        btnBarColor->setObjectName(QStringLiteral("btnBarColor"));

        gridLayout_3->addWidget(btnBarColor, 2, 1, 1, 2);

        chkBoxSmooth = new QCheckBox(groupBox_4);
        chkBoxSmooth->setObjectName(QStringLiteral("chkBoxSmooth"));

        gridLayout_3->addWidget(chkBoxSmooth, 3, 0, 1, 2);

        chkBoxItemLabel = new QCheckBox(groupBox_4);
        chkBoxItemLabel->setObjectName(QStringLiteral("chkBoxItemLabel"));

        gridLayout_3->addWidget(chkBoxItemLabel, 3, 2, 1, 1);


        verticalLayout->addWidget(groupBox_4);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);

        MainWindow->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);
#ifndef QT_NO_SHORTCUT
        label_6->setBuddy(comboCamera);
        label_7->setBuddy(sliderH);
        label_8->setBuddy(sliderV);
        label_9->setBuddy(sliderZoom);
        label->setBuddy(cBoxTheme);
        label_4->setBuddy(spinFontSize);
        label_3->setBuddy(cBoxSelectionMode);
        label_2->setBuddy(cBoxBarStyle);
        label_5->setBuddy(spinItemSize);
#endif // QT_NO_SHORTCUT

        retranslateUi(MainWindow);

        comboCamera->setCurrentIndex(2);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Q3DScatter\347\273\230\345\233\276", nullptr));
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
        label_9->setText(QApplication::translate("MainWindow", "\347\274\251  \346\224\276", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "\345\233\276\350\241\250\346\200\273\344\275\223", nullptr));
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
        cBoxSelectionMode->setItemText(0, QApplication::translate("MainWindow", "None", nullptr));
        cBoxSelectionMode->setItemText(1, QApplication::translate("MainWindow", "Item", nullptr));
        cBoxSelectionMode->setItemText(2, QApplication::translate("MainWindow", "Row", nullptr));
        cBoxSelectionMode->setItemText(3, QApplication::translate("MainWindow", "Item And Row", nullptr));
        cBoxSelectionMode->setItemText(4, QApplication::translate("MainWindow", "Column", nullptr));
        cBoxSelectionMode->setItemText(5, QApplication::translate("MainWindow", "Item And Column", nullptr));
        cBoxSelectionMode->setItemText(6, QApplication::translate("MainWindow", "Row And Column", nullptr));
        cBoxSelectionMode->setItemText(7, QApplication::translate("MainWindow", "Item Row And Column", nullptr));
        cBoxSelectionMode->setItemText(8, QApplication::translate("MainWindow", "Slice", nullptr));

        chkBoxBackground->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\350\203\214\346\231\257", nullptr));
        chkBoxGrid->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\350\203\214\346\231\257\347\275\221\346\240\274", nullptr));
        chkBoxShadow->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\351\230\264\345\275\261", nullptr));
        chkBoxAxisBackground->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\350\275\264\346\240\207\347\255\276\350\203\214\346\231\257", nullptr));
        chkBoxReflection->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\345\200\222\345\275\261", nullptr));
        chkBoxAxisTitle->setText(QApplication::translate("MainWindow", "\350\275\264\346\240\207\351\242\230\345\217\257\350\247\201", nullptr));
        chkBoxReverse->setText(QApplication::translate("MainWindow", "\345\236\202\347\233\264\345\235\220\346\240\207\350\275\264\345\217\215\345\220\221", nullptr));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "\345\272\217\345\210\227\350\256\276\347\275\256", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "\346\225\243\347\202\271\345\275\242\347\212\266", nullptr));
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

        label_5->setText(QApplication::translate("MainWindow", "\346\225\243\347\202\271\345\244\247\345\260\217", nullptr));
        btnBarColor->setText(QApplication::translate("MainWindow", "\346\225\243\347\202\271\347\232\204\351\242\234\350\211\262", nullptr));
        chkBoxSmooth->setText(QApplication::translate("MainWindow", "\345\205\211\346\273\221\346\225\210\346\236\234", nullptr));
        chkBoxItemLabel->setText(QApplication::translate("MainWindow", "\351\241\271\346\240\207\347\255\276\345\217\257\350\247\201", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
