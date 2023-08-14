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
#include <QtWidgets/QFrame>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_2;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout;
    QGraphicsView *graphicsView;
    QWidget *widget;
    QHBoxLayout *horizontalLayout;
    QPushButton *btnOpen;
    QPushButton *btnPlay;
    QPushButton *btnPause;
    QPushButton *btnStop;
    QSpacerItem *horizontalSpacer;
    QPushButton *btnSound;
    QSlider *sliderVolumn;
    QPushButton *btnText;
    QPushButton *btnZoomIn;
    QPushButton *btnZoomOut;
    QPushButton *btnClose;
    QFrame *line;
    QWidget *widget_2;
    QHBoxLayout *horizontalLayout_2;
    QLabel *LabCurMedia;
    QSlider *sliderPosition;
    QLabel *LabRatio;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(478, 357);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout_2 = new QVBoxLayout(centralWidget);
        verticalLayout_2->setSpacing(1);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(1, 1, 1, 1);
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        verticalLayout = new QVBoxLayout(groupBox);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(2, 2, 2, 2);
        graphicsView = new QGraphicsView(groupBox);
        graphicsView->setObjectName(QStringLiteral("graphicsView"));

        verticalLayout->addWidget(graphicsView);


        verticalLayout_2->addWidget(groupBox);

        widget = new QWidget(centralWidget);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setMinimumSize(QSize(0, 35));
        widget->setMaximumSize(QSize(16777215, 25));
        horizontalLayout = new QHBoxLayout(widget);
        horizontalLayout->setSpacing(9);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(1, 1, 1, 1);
        btnOpen = new QPushButton(widget);
        btnOpen->setObjectName(QStringLiteral("btnOpen"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/001.GIF"), QSize(), QIcon::Normal, QIcon::Off);
        btnOpen->setIcon(icon);

        horizontalLayout->addWidget(btnOpen);

        btnPlay = new QPushButton(widget);
        btnPlay->setObjectName(QStringLiteral("btnPlay"));
        btnPlay->setEnabled(false);
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/620.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnPlay->setIcon(icon1);

        horizontalLayout->addWidget(btnPlay);

        btnPause = new QPushButton(widget);
        btnPause->setObjectName(QStringLiteral("btnPause"));
        btnPause->setEnabled(false);
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/622.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnPause->setIcon(icon2);

        horizontalLayout->addWidget(btnPause);

        btnStop = new QPushButton(widget);
        btnStop->setObjectName(QStringLiteral("btnStop"));
        btnStop->setEnabled(false);
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/624.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnStop->setIcon(icon3);

        horizontalLayout->addWidget(btnStop);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        btnSound = new QPushButton(widget);
        btnSound->setObjectName(QStringLiteral("btnSound"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/volumn.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnSound->setIcon(icon4);
        btnSound->setFlat(true);

        horizontalLayout->addWidget(btnSound);

        sliderVolumn = new QSlider(widget);
        sliderVolumn->setObjectName(QStringLiteral("sliderVolumn"));
        sliderVolumn->setMaximum(100);
        sliderVolumn->setValue(100);
        sliderVolumn->setOrientation(Qt::Horizontal);

        horizontalLayout->addWidget(sliderVolumn);

        btnText = new QPushButton(widget);
        btnText->setObjectName(QStringLiteral("btnText"));
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/icons/images/165.GIF"), QSize(), QIcon::Normal, QIcon::Off);
        btnText->setIcon(icon5);
        btnText->setCheckable(true);
        btnText->setChecked(true);

        horizontalLayout->addWidget(btnText);

        btnZoomIn = new QPushButton(widget);
        btnZoomIn->setObjectName(QStringLiteral("btnZoomIn"));
        QIcon icon6;
        icon6.addFile(QStringLiteral(":/icons/images/418.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnZoomIn->setIcon(icon6);

        horizontalLayout->addWidget(btnZoomIn);

        btnZoomOut = new QPushButton(widget);
        btnZoomOut->setObjectName(QStringLiteral("btnZoomOut"));
        QIcon icon7;
        icon7.addFile(QStringLiteral(":/icons/images/416.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnZoomOut->setIcon(icon7);

        horizontalLayout->addWidget(btnZoomOut);

        btnClose = new QPushButton(widget);
        btnClose->setObjectName(QStringLiteral("btnClose"));
        QIcon icon8;
        icon8.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        btnClose->setIcon(icon8);

        horizontalLayout->addWidget(btnClose);


        verticalLayout_2->addWidget(widget);

        line = new QFrame(centralWidget);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShadow(QFrame::Plain);
        line->setFrameShape(QFrame::HLine);

        verticalLayout_2->addWidget(line);

        widget_2 = new QWidget(centralWidget);
        widget_2->setObjectName(QStringLiteral("widget_2"));
        widget_2->setMaximumSize(QSize(16777215, 25));
        horizontalLayout_2 = new QHBoxLayout(widget_2);
        horizontalLayout_2->setSpacing(9);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(1, 1, 1, 1);
        LabCurMedia = new QLabel(widget_2);
        LabCurMedia->setObjectName(QStringLiteral("LabCurMedia"));
        LabCurMedia->setMinimumSize(QSize(100, 0));

        horizontalLayout_2->addWidget(LabCurMedia);

        sliderPosition = new QSlider(widget_2);
        sliderPosition->setObjectName(QStringLiteral("sliderPosition"));
        sliderPosition->setTracking(false);
        sliderPosition->setOrientation(Qt::Horizontal);

        horizontalLayout_2->addWidget(sliderPosition);

        LabRatio = new QLabel(widget_2);
        LabRatio->setObjectName(QStringLiteral("LabRatio"));
        LabRatio->setMinimumSize(QSize(80, 0));

        horizontalLayout_2->addWidget(LabRatio);


        verticalLayout_2->addWidget(widget_2);

        MainWindow->setCentralWidget(centralWidget);

        retranslateUi(MainWindow);
        QObject::connect(btnClose, SIGNAL(clicked()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "QGraphicsVideoItem \350\247\206\351\242\221\346\222\255\346\224\276", nullptr));
        groupBox->setTitle(QString());
#ifndef QT_NO_TOOLTIP
        btnOpen->setToolTip(QApplication::translate("MainWindow", "\346\211\223\345\274\200\350\247\206\351\242\221\346\226\207\344\273\266", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        btnPlay->setToolTip(QApplication::translate("MainWindow", "\346\222\255\346\224\276", nullptr));
#endif // QT_NO_TOOLTIP
        btnPlay->setText(QString());
#ifndef QT_NO_TOOLTIP
        btnPause->setToolTip(QApplication::translate("MainWindow", "\346\232\202\345\201\234", nullptr));
#endif // QT_NO_TOOLTIP
        btnPause->setText(QString());
#ifndef QT_NO_TOOLTIP
        btnStop->setToolTip(QApplication::translate("MainWindow", "\345\201\234\346\255\242", nullptr));
#endif // QT_NO_TOOLTIP
        btnStop->setText(QString());
        btnSound->setText(QString());
#ifndef QT_NO_TOOLTIP
        btnText->setToolTip(QApplication::translate("MainWindow", "\345\274\271\345\271\225\346\226\207\345\255\227", nullptr));
#endif // QT_NO_TOOLTIP
        btnText->setText(QString());
#ifndef QT_NO_TOOLTIP
        btnZoomIn->setToolTip(QApplication::translate("MainWindow", "\346\224\276\345\244\247", nullptr));
#endif // QT_NO_TOOLTIP
        btnZoomIn->setText(QString());
#ifndef QT_NO_TOOLTIP
        btnZoomOut->setToolTip(QApplication::translate("MainWindow", "\347\274\251\345\260\217", nullptr));
#endif // QT_NO_TOOLTIP
        btnZoomOut->setText(QString());
#ifndef QT_NO_TOOLTIP
        btnClose->setToolTip(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#endif // QT_NO_TOOLTIP
        btnClose->setText(QString());
        LabCurMedia->setText(QApplication::translate("MainWindow", "\346\227\240\346\233\262\347\233\256", nullptr));
        LabRatio->setText(QApplication::translate("MainWindow", "\350\277\233\345\272\246", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
