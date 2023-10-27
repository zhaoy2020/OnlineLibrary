/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.11.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QCameraViewfinder>
#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actStartCamera;
    QAction *actStopCamera;
    QAction *actCapture;
    QAction *actQuit;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_3;
    QGroupBox *groupBox_2;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QComboBox *comboCamera;
    QCheckBox *checkStillImage;
    QCheckBox *checkVideo;
    QCheckBox *checkExposure;
    QCheckBox *checkFocus;
    QSpacerItem *horizontalSpacer;
    QSplitter *splitter;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout;
    QCameraViewfinder *viewFinder;
    QGroupBox *groupBox_3;
    QVBoxLayout *verticalLayout_2;
    QLabel *LabImage;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(611, 385);
        actStartCamera = new QAction(MainWindow);
        actStartCamera->setObjectName(QStringLiteral("actStartCamera"));
        actStartCamera->setEnabled(false);
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/video_chat.ico"), QSize(), QIcon::Normal, QIcon::Off);
        actStartCamera->setIcon(icon);
        actStopCamera = new QAction(MainWindow);
        actStopCamera->setObjectName(QStringLiteral("actStopCamera"));
        actStopCamera->setEnabled(false);
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/remove_user.ico"), QSize(), QIcon::Normal, QIcon::Off);
        actStopCamera->setIcon(icon1);
        actCapture = new QAction(MainWindow);
        actCapture->setObjectName(QStringLiteral("actCapture"));
        actCapture->setEnabled(false);
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/picture.ico"), QSize(), QIcon::Normal, QIcon::Off);
        actCapture->setIcon(icon2);
        actQuit = new QAction(MainWindow);
        actQuit->setObjectName(QStringLiteral("actQuit"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/exit_24.ico"), QSize(), QIcon::Normal, QIcon::Off);
        actQuit->setIcon(icon3);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout_3 = new QVBoxLayout(centralWidget);
        verticalLayout_3->setSpacing(3);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(3, 3, 3, 3);
        groupBox_2 = new QGroupBox(centralWidget);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        groupBox_2->setMaximumSize(QSize(16777215, 40));
        horizontalLayout = new QHBoxLayout(groupBox_2);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label = new QLabel(groupBox_2);
        label->setObjectName(QStringLiteral("label"));
        label->setMaximumSize(QSize(80, 16777215));

        horizontalLayout->addWidget(label);

        comboCamera = new QComboBox(groupBox_2);
        comboCamera->setObjectName(QStringLiteral("comboCamera"));
        comboCamera->setMinimumSize(QSize(150, 20));

        horizontalLayout->addWidget(comboCamera);

        checkStillImage = new QCheckBox(groupBox_2);
        checkStillImage->setObjectName(QStringLiteral("checkStillImage"));
        checkStillImage->setEnabled(false);

        horizontalLayout->addWidget(checkStillImage);

        checkVideo = new QCheckBox(groupBox_2);
        checkVideo->setObjectName(QStringLiteral("checkVideo"));
        checkVideo->setEnabled(false);

        horizontalLayout->addWidget(checkVideo);

        checkExposure = new QCheckBox(groupBox_2);
        checkExposure->setObjectName(QStringLiteral("checkExposure"));
        checkExposure->setEnabled(false);

        horizontalLayout->addWidget(checkExposure);

        checkFocus = new QCheckBox(groupBox_2);
        checkFocus->setObjectName(QStringLiteral("checkFocus"));
        checkFocus->setEnabled(false);

        horizontalLayout->addWidget(checkFocus);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        verticalLayout_3->addWidget(groupBox_2);

        splitter = new QSplitter(centralWidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        groupBox = new QGroupBox(splitter);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        verticalLayout = new QVBoxLayout(groupBox);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        viewFinder = new QCameraViewfinder(groupBox);
        viewFinder->setObjectName(QStringLiteral("viewFinder"));
        viewFinder->setMinimumSize(QSize(150, 0));

        verticalLayout->addWidget(viewFinder);

        splitter->addWidget(groupBox);
        groupBox_3 = new QGroupBox(splitter);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        verticalLayout_2 = new QVBoxLayout(groupBox_3);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        LabImage = new QLabel(groupBox_3);
        LabImage->setObjectName(QStringLiteral("LabImage"));
        LabImage->setAlignment(Qt::AlignCenter);

        verticalLayout_2->addWidget(LabImage);

        splitter->addWidget(groupBox_3);

        verticalLayout_3->addWidget(splitter);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 611, 23));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        mainToolBar->addAction(actStartCamera);
        mainToolBar->addAction(actStopCamera);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actCapture);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actQuit);

        retranslateUi(MainWindow);
        QObject::connect(actQuit, SIGNAL(triggered()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Demo10_7, Camera\346\213\215\347\205\247", nullptr));
        actStartCamera->setText(QApplication::translate("MainWindow", "\345\274\200\345\220\257\346\221\204\345\203\217\345\244\264", nullptr));
#ifndef QT_NO_TOOLTIP
        actStartCamera->setToolTip(QApplication::translate("MainWindow", "\345\274\200\345\220\257\346\221\204\345\203\217\345\244\264", nullptr));
#endif // QT_NO_TOOLTIP
        actStopCamera->setText(QApplication::translate("MainWindow", "\345\205\263\351\227\255\346\221\204\345\203\217\345\244\264", nullptr));
#ifndef QT_NO_TOOLTIP
        actStopCamera->setToolTip(QApplication::translate("MainWindow", "\345\205\263\351\227\255\346\221\204\345\203\217\345\244\264", nullptr));
#endif // QT_NO_TOOLTIP
        actCapture->setText(QApplication::translate("MainWindow", "\346\221\204\345\203\217\345\244\264\346\213\215\347\205\247", nullptr));
#ifndef QT_NO_TOOLTIP
        actCapture->setToolTip(QApplication::translate("MainWindow", "\346\221\204\345\203\217\345\244\264\346\213\215\347\205\247", nullptr));
#endif // QT_NO_TOOLTIP
        actQuit->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actQuit->setToolTip(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#endif // QT_NO_TOOLTIP
        groupBox_2->setTitle(QString());
        label->setText(QApplication::translate("MainWindow", "\346\221\204\345\203\217\345\244\264\350\256\276\345\244\207", nullptr));
        checkStillImage->setText(QApplication::translate("MainWindow", "\346\224\257\346\214\201\346\213\215\347\205\247", nullptr));
        checkVideo->setText(QApplication::translate("MainWindow", "\346\224\257\346\214\201\345\275\225\345\203\217", nullptr));
        checkExposure->setText(QApplication::translate("MainWindow", "\346\224\257\346\214\201\346\233\235\345\205\211\346\216\247\345\210\266", nullptr));
        checkFocus->setText(QApplication::translate("MainWindow", "\346\224\257\346\214\201\345\217\230\347\204\246", nullptr));
        groupBox->setTitle(QApplication::translate("MainWindow", "\346\221\204\345\203\217\345\244\264\351\242\204\350\247\210", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "\346\212\223\345\217\226\347\232\204\345\233\276\347\211\207", nullptr));
        LabImage->setText(QApplication::translate("MainWindow", "\346\212\223\345\217\226\347\232\204\345\233\276\347\211\207", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
