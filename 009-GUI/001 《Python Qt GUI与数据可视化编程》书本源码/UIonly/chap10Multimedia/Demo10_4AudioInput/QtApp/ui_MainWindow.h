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
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actDeviceTest;
    QAction *actStart;
    QAction *actStop;
    QAction *actQuit;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_3;
    QGroupBox *groupBox_saveMode;
    QHBoxLayout *horizontalLayout_3;
    QRadioButton *radioSaveMode_Inner;
    QRadioButton *radioSaveMode_QFile;
    QSplitter *splitter;
    QGroupBox *groupBoxDevice;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QComboBox *comboDevices;
    QFrame *line;
    QGridLayout *gridLayout;
    QLabel *actualCodecLabel;
    QComboBox *comboCodec;
    QLabel *actualFreqLabel;
    QComboBox *comboSampleRate;
    QLabel *actualChannelLabel;
    QComboBox *comboChannels;
    QLabel *actualSampleTypeLabel;
    QComboBox *comboSampleTypes;
    QLabel *actualSampleSizeLabel;
    QComboBox *comboSampleSizes;
    QLabel *actualEndianLabel;
    QComboBox *comboByteOrder;
    QFrame *line_2;
    QLabel *label_2;
    QFrame *line_4;
    QHBoxLayout *horizontalLayout_2;
    QLabel *LabVol;
    QSlider *sliderVolumn;
    QGroupBox *groupBox_disp;
    QVBoxLayout *verticalLayout;
    QGridLayout *gridLayout_2;
    QProgressBar *progBar_Max;
    QProgressBar *progBar_Min;
    QProgressBar *progBar_Diff;
    QLabel *label_3;
    QLabel *label_4;
    QLabel *label_5;
    QFrame *line_3;
    QLabel *LabBufferSize;
    QLabel *LabBytesReady;
    QLabel *LabBlockSize;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(477, 456);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        actDeviceTest = new QAction(MainWindow);
        actDeviceTest->setObjectName(QStringLiteral("actDeviceTest"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/22.ico"), QSize(), QIcon::Normal, QIcon::Off);
        actDeviceTest->setIcon(icon);
        actStart = new QAction(MainWindow);
        actStart->setObjectName(QStringLiteral("actStart"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/626.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actStart->setIcon(icon1);
        actStop = new QAction(MainWindow);
        actStop->setObjectName(QStringLiteral("actStop"));
        actStop->setEnabled(false);
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/624.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actStop->setIcon(icon2);
        actQuit = new QAction(MainWindow);
        actQuit->setObjectName(QStringLiteral("actQuit"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actQuit->setIcon(icon3);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout_3 = new QVBoxLayout(centralWidget);
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(5, 5, 5, 5);
        groupBox_saveMode = new QGroupBox(centralWidget);
        groupBox_saveMode->setObjectName(QStringLiteral("groupBox_saveMode"));
        groupBox_saveMode->setMinimumSize(QSize(0, 50));
        horizontalLayout_3 = new QHBoxLayout(groupBox_saveMode);
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        radioSaveMode_Inner = new QRadioButton(groupBox_saveMode);
        radioSaveMode_Inner->setObjectName(QStringLiteral("radioSaveMode_Inner"));
        radioSaveMode_Inner->setChecked(true);

        horizontalLayout_3->addWidget(radioSaveMode_Inner);

        radioSaveMode_QFile = new QRadioButton(groupBox_saveMode);
        radioSaveMode_QFile->setObjectName(QStringLiteral("radioSaveMode_QFile"));

        horizontalLayout_3->addWidget(radioSaveMode_QFile);


        verticalLayout_3->addWidget(groupBox_saveMode);

        splitter = new QSplitter(centralWidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        groupBoxDevice = new QGroupBox(splitter);
        groupBoxDevice->setObjectName(QStringLiteral("groupBoxDevice"));
        verticalLayout_2 = new QVBoxLayout(groupBoxDevice);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label = new QLabel(groupBoxDevice);
        label->setObjectName(QStringLiteral("label"));
        label->setMaximumSize(QSize(80, 16777215));

        horizontalLayout->addWidget(label);

        comboDevices = new QComboBox(groupBoxDevice);
        comboDevices->setObjectName(QStringLiteral("comboDevices"));

        horizontalLayout->addWidget(comboDevices);


        verticalLayout_2->addLayout(horizontalLayout);

        line = new QFrame(groupBoxDevice);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        verticalLayout_2->addWidget(line);

        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        actualCodecLabel = new QLabel(groupBoxDevice);
        actualCodecLabel->setObjectName(QStringLiteral("actualCodecLabel"));

        gridLayout->addWidget(actualCodecLabel, 0, 0, 1, 1);

        comboCodec = new QComboBox(groupBoxDevice);
        comboCodec->setObjectName(QStringLiteral("comboCodec"));

        gridLayout->addWidget(comboCodec, 0, 1, 1, 1);

        actualFreqLabel = new QLabel(groupBoxDevice);
        actualFreqLabel->setObjectName(QStringLiteral("actualFreqLabel"));

        gridLayout->addWidget(actualFreqLabel, 1, 0, 1, 1);

        comboSampleRate = new QComboBox(groupBoxDevice);
        comboSampleRate->setObjectName(QStringLiteral("comboSampleRate"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(comboSampleRate->sizePolicy().hasHeightForWidth());
        comboSampleRate->setSizePolicy(sizePolicy);

        gridLayout->addWidget(comboSampleRate, 1, 1, 1, 1);

        actualChannelLabel = new QLabel(groupBoxDevice);
        actualChannelLabel->setObjectName(QStringLiteral("actualChannelLabel"));

        gridLayout->addWidget(actualChannelLabel, 2, 0, 1, 1);

        comboChannels = new QComboBox(groupBoxDevice);
        comboChannels->setObjectName(QStringLiteral("comboChannels"));

        gridLayout->addWidget(comboChannels, 2, 1, 1, 1);

        actualSampleTypeLabel = new QLabel(groupBoxDevice);
        actualSampleTypeLabel->setObjectName(QStringLiteral("actualSampleTypeLabel"));

        gridLayout->addWidget(actualSampleTypeLabel, 3, 0, 1, 1);

        comboSampleTypes = new QComboBox(groupBoxDevice);
        comboSampleTypes->setObjectName(QStringLiteral("comboSampleTypes"));

        gridLayout->addWidget(comboSampleTypes, 3, 1, 1, 1);

        actualSampleSizeLabel = new QLabel(groupBoxDevice);
        actualSampleSizeLabel->setObjectName(QStringLiteral("actualSampleSizeLabel"));

        gridLayout->addWidget(actualSampleSizeLabel, 4, 0, 1, 1);

        comboSampleSizes = new QComboBox(groupBoxDevice);
        comboSampleSizes->setObjectName(QStringLiteral("comboSampleSizes"));

        gridLayout->addWidget(comboSampleSizes, 4, 1, 1, 1);

        actualEndianLabel = new QLabel(groupBoxDevice);
        actualEndianLabel->setObjectName(QStringLiteral("actualEndianLabel"));

        gridLayout->addWidget(actualEndianLabel, 5, 0, 1, 1);

        comboByteOrder = new QComboBox(groupBoxDevice);
        comboByteOrder->setObjectName(QStringLiteral("comboByteOrder"));

        gridLayout->addWidget(comboByteOrder, 5, 1, 1, 1);


        verticalLayout_2->addLayout(gridLayout);

        line_2 = new QFrame(groupBoxDevice);
        line_2->setObjectName(QStringLiteral("line_2"));
        line_2->setFrameShape(QFrame::HLine);
        line_2->setFrameShadow(QFrame::Sunken);

        verticalLayout_2->addWidget(line_2);

        label_2 = new QLabel(groupBoxDevice);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setTextFormat(Qt::AutoText);

        verticalLayout_2->addWidget(label_2);

        line_4 = new QFrame(groupBoxDevice);
        line_4->setObjectName(QStringLiteral("line_4"));
        line_4->setFrameShape(QFrame::HLine);
        line_4->setFrameShadow(QFrame::Sunken);

        verticalLayout_2->addWidget(line_4);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        LabVol = new QLabel(groupBoxDevice);
        LabVol->setObjectName(QStringLiteral("LabVol"));

        horizontalLayout_2->addWidget(LabVol);

        sliderVolumn = new QSlider(groupBoxDevice);
        sliderVolumn->setObjectName(QStringLiteral("sliderVolumn"));
        sliderVolumn->setMaximum(100);
        sliderVolumn->setValue(90);
        sliderVolumn->setOrientation(Qt::Horizontal);

        horizontalLayout_2->addWidget(sliderVolumn);


        verticalLayout_2->addLayout(horizontalLayout_2);

        splitter->addWidget(groupBoxDevice);
        groupBox_disp = new QGroupBox(splitter);
        groupBox_disp->setObjectName(QStringLiteral("groupBox_disp"));
        verticalLayout = new QVBoxLayout(groupBox_disp);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        gridLayout_2 = new QGridLayout();
        gridLayout_2->setSpacing(6);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        progBar_Max = new QProgressBar(groupBox_disp);
        progBar_Max->setObjectName(QStringLiteral("progBar_Max"));
        progBar_Max->setMaximum(256);
        progBar_Max->setValue(120);
        progBar_Max->setTextVisible(false);
        progBar_Max->setOrientation(Qt::Vertical);

        gridLayout_2->addWidget(progBar_Max, 0, 0, 1, 1);

        progBar_Min = new QProgressBar(groupBox_disp);
        progBar_Min->setObjectName(QStringLiteral("progBar_Min"));
        progBar_Min->setMaximum(256);
        progBar_Min->setValue(120);
        progBar_Min->setTextVisible(false);
        progBar_Min->setOrientation(Qt::Vertical);

        gridLayout_2->addWidget(progBar_Min, 0, 1, 1, 1);

        progBar_Diff = new QProgressBar(groupBox_disp);
        progBar_Diff->setObjectName(QStringLiteral("progBar_Diff"));
        progBar_Diff->setMaximum(256);
        progBar_Diff->setValue(120);
        progBar_Diff->setTextVisible(false);
        progBar_Diff->setOrientation(Qt::Vertical);

        gridLayout_2->addWidget(progBar_Diff, 0, 2, 1, 1);

        label_3 = new QLabel(groupBox_disp);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout_2->addWidget(label_3, 1, 0, 1, 1);

        label_4 = new QLabel(groupBox_disp);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout_2->addWidget(label_4, 1, 1, 1, 1);

        label_5 = new QLabel(groupBox_disp);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout_2->addWidget(label_5, 1, 2, 1, 1);


        verticalLayout->addLayout(gridLayout_2);

        line_3 = new QFrame(groupBox_disp);
        line_3->setObjectName(QStringLiteral("line_3"));
        line_3->setFrameShape(QFrame::HLine);
        line_3->setFrameShadow(QFrame::Sunken);

        verticalLayout->addWidget(line_3);

        LabBufferSize = new QLabel(groupBox_disp);
        LabBufferSize->setObjectName(QStringLiteral("LabBufferSize"));
        LabBufferSize->setMinimumSize(QSize(200, 0));

        verticalLayout->addWidget(LabBufferSize);

        LabBytesReady = new QLabel(groupBox_disp);
        LabBytesReady->setObjectName(QStringLiteral("LabBytesReady"));

        verticalLayout->addWidget(LabBytesReady);

        LabBlockSize = new QLabel(groupBox_disp);
        LabBlockSize->setObjectName(QStringLiteral("LabBlockSize"));
        LabBlockSize->setMinimumSize(QSize(190, 0));

        verticalLayout->addWidget(LabBlockSize);

        splitter->addWidget(groupBox_disp);

        verticalLayout_3->addWidget(splitter);

        MainWindow->setCentralWidget(centralWidget);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setIconSize(QSize(16, 16));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        mainToolBar->addAction(actDeviceTest);
        mainToolBar->addAction(actStart);
        mainToolBar->addAction(actStop);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actQuit);

        retranslateUi(MainWindow);
        QObject::connect(actQuit, SIGNAL(triggered()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "QAudioInput\350\277\233\350\241\214\351\237\263\351\242\221\350\276\223\345\205\245", nullptr));
        actDeviceTest->setText(QApplication::translate("MainWindow", "\346\265\213\350\257\225\351\237\263\351\242\221\350\256\276\347\275\256", nullptr));
#ifndef QT_NO_TOOLTIP
        actDeviceTest->setToolTip(QApplication::translate("MainWindow", "\346\265\213\350\257\225\351\237\263\351\242\221\350\256\276\347\275\256", nullptr));
#endif // QT_NO_TOOLTIP
        actStart->setText(QApplication::translate("MainWindow", "\345\274\200\345\247\213", nullptr));
#ifndef QT_NO_TOOLTIP
        actStart->setToolTip(QApplication::translate("MainWindow", "\345\274\200\345\247\213", nullptr));
#endif // QT_NO_TOOLTIP
        actStop->setText(QApplication::translate("MainWindow", "\345\201\234\346\255\242", nullptr));
#ifndef QT_NO_TOOLTIP
        actStop->setToolTip(QApplication::translate("MainWindow", "\345\201\234\346\255\242", nullptr));
#endif // QT_NO_TOOLTIP
        actQuit->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actQuit->setToolTip(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#endif // QT_NO_TOOLTIP
        groupBox_saveMode->setTitle(QApplication::translate("MainWindow", "\346\225\260\346\215\256\346\265\201\350\257\273\345\217\226\346\226\271\345\274\217", nullptr));
#ifndef QT_NO_TOOLTIP
        radioSaveMode_Inner->setToolTip(QApplication::translate("MainWindow", "strat()\346\226\271\345\274\217\345\220\257\345\212\250\357\274\214\345\210\251\347\224\250\350\277\224\345\233\236\347\232\204\345\206\205\345\273\272IODevice\350\257\273\345\217\226\346\225\260\346\215\256", nullptr));
#endif // QT_NO_TOOLTIP
        radioSaveMode_Inner->setText(QApplication::translate("MainWindow", "\344\275\277\347\224\250\345\206\205\345\273\272IODevice", nullptr));
#ifndef QT_NO_TOOLTIP
        radioSaveMode_QFile->setToolTip(QApplication::translate("MainWindow", "start(file)\346\226\271\345\274\217\345\220\257\345\212\250\357\274\214file\346\230\257\345\244\226\345\273\272QFile\345\257\271\350\261\241", nullptr));
#endif // QT_NO_TOOLTIP
        radioSaveMode_QFile->setText(QApplication::translate("MainWindow", "\344\275\277\347\224\250QFile\345\257\271\350\261\241(test.raw)", nullptr));
        groupBoxDevice->setTitle(QApplication::translate("MainWindow", "\346\224\257\346\214\201\347\232\204\351\237\263\351\242\221\350\276\223\345\205\245\350\256\276\347\275\256", nullptr));
        label->setText(QApplication::translate("MainWindow", "\351\237\263\351\242\221\350\276\223\345\205\245\350\256\276\345\244\207", nullptr));
        actualCodecLabel->setText(QApplication::translate("MainWindow", "codec", nullptr));
        actualFreqLabel->setText(QApplication::translate("MainWindow", "Frequency (Hz)", nullptr));
        actualChannelLabel->setText(QApplication::translate("MainWindow", "Channels", nullptr));
        actualSampleTypeLabel->setText(QApplication::translate("MainWindow", "SampleType", nullptr));
        actualSampleSizeLabel->setText(QApplication::translate("MainWindow", "Sample size (bits)", nullptr));
        actualEndianLabel->setText(QApplication::translate("MainWindow", "Endianness", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "<html><head/><body><p>\344\270\272\344\272\206\345\244\204\347\220\206\346\226\271\344\276\277\357\274\214\346\234\254\347\250\213\345\272\217\345\233\272\345\256\232\351\207\207\346\240\267\346\226\271\345\274\217\357\274\232</p><p>8000Hz, 1 \351\200\232\351\201\223\357\274\214audio/pcm</p><p>8bits,UnSignedInt,LittleEndian</p></body></html>", nullptr));
        LabVol->setText(QApplication::translate("MainWindow", "\345\275\225\351\237\263\351\237\263\351\207\217", nullptr));
        groupBox_disp->setTitle(QApplication::translate("MainWindow", "\345\256\236\346\227\266\350\276\223\345\205\245\351\237\263\351\242\221\347\274\223\345\206\262\345\214\272\346\225\260\346\215\256\347\273\237\350\256\241\345\217\202\346\225\260", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "\346\234\200\345\244\247\345\200\274", nullptr));
        label_4->setText(QApplication::translate("MainWindow", "\346\234\200\345\260\217\345\200\274", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "\345\267\256\345\200\274", nullptr));
        LabBufferSize->setText(QApplication::translate("MainWindow", "QAudioInput.bufferSize()=", nullptr));
        LabBytesReady->setText(QApplication::translate("MainWindow", "QAudioInput.bytesReady()=", nullptr));
        LabBlockSize->setText(QApplication::translate("MainWindow", "IODevice\346\225\260\346\215\256\345\235\227\345\255\227\350\212\202\346\225\260=", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
