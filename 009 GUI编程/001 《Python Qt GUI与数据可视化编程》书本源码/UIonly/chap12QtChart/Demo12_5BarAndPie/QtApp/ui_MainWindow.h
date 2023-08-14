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
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTableView>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_13;
    QFrame *frame_6;
    QHBoxLayout *horizontalLayout_7;
    QToolButton *toolBtn_GenData;
    QToolButton *toolBtn_Counting;
    QLabel *label_7;
    QComboBox *comboTheme;
    QLabel *label_8;
    QComboBox *comboAnimation;
    QSpacerItem *horizontalSpacer_7;
    QToolButton *toolBtn_Quit;
    QSplitter *splitter_2;
    QFrame *frameData;
    QVBoxLayout *verticalLayout_3;
    QSplitter *splitter;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout;
    QFrame *frame_2;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_5;
    QSpinBox *spinCount;
    QSpacerItem *horizontalSpacer_6;
    QTableView *tableView;
    QGroupBox *groupBox_7;
    QVBoxLayout *verticalLayout_2;
    QTreeWidget *treeWidget;
    QTabWidget *tabWidget;
    QWidget *tab;
    QVBoxLayout *verticalLayout_9;
    QWidget *widget;
    QVBoxLayout *verticalLayout_8;
    QFrame *frame;
    QHBoxLayout *horizontalLayout;
    QPushButton *btnBuildBarChart;
    QPushButton *btnBuildBarChartH;
    QSpacerItem *horizontalSpacer_2;
    QChartView *chartViewBar;
    QWidget *tab_3;
    QVBoxLayout *verticalLayout_10;
    QWidget *widget_2;
    QVBoxLayout *verticalLayout_5;
    QFrame *frame_3;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *btnBuildStackedBar;
    QPushButton *btnBuildStackedBarH;
    QSpacerItem *horizontalSpacer_3;
    QChartView *chartViewStackedBar;
    QWidget *tab_5;
    QVBoxLayout *verticalLayout_12;
    QWidget *widget_4;
    QVBoxLayout *verticalLayout_6;
    QFrame *frame_5;
    QHBoxLayout *horizontalLayout_4;
    QPushButton *btnPercentBar;
    QPushButton *btnPercentBarH;
    QSpacerItem *horizontalSpacer_4;
    QChartView *chartViewPercentBar;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_11;
    QWidget *widget_3;
    QVBoxLayout *verticalLayout_4;
    QFrame *frame_4;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label;
    QComboBox *comboCourse;
    QPushButton *btnDrawPieChart;
    QLabel *label_2;
    QDoubleSpinBox *spinHoleSize;
    QLabel *label_3;
    QDoubleSpinBox *spinPieSize;
    QSpacerItem *horizontalSpacer;
    QChartView *chartViewPie;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(901, 533);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout_13 = new QVBoxLayout(centralWidget);
        verticalLayout_13->setSpacing(6);
        verticalLayout_13->setContentsMargins(11, 11, 11, 11);
        verticalLayout_13->setObjectName(QStringLiteral("verticalLayout_13"));
        verticalLayout_13->setContentsMargins(2, 2, 2, 2);
        frame_6 = new QFrame(centralWidget);
        frame_6->setObjectName(QStringLiteral("frame_6"));
        frame_6->setMaximumSize(QSize(16777215, 40));
        frame_6->setFrameShape(QFrame::Panel);
        frame_6->setFrameShadow(QFrame::Raised);
        horizontalLayout_7 = new QHBoxLayout(frame_6);
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        horizontalLayout_7->setContentsMargins(-1, 5, -1, 5);
        toolBtn_GenData = new QToolButton(frame_6);
        toolBtn_GenData->setObjectName(QStringLiteral("toolBtn_GenData"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/828.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        toolBtn_GenData->setIcon(icon);
        toolBtn_GenData->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_7->addWidget(toolBtn_GenData);

        toolBtn_Counting = new QToolButton(frame_6);
        toolBtn_Counting->setObjectName(QStringLiteral("toolBtn_Counting"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/216.GIF"), QSize(), QIcon::Normal, QIcon::Off);
        toolBtn_Counting->setIcon(icon1);
        toolBtn_Counting->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_7->addWidget(toolBtn_Counting);

        label_7 = new QLabel(frame_6);
        label_7->setObjectName(QStringLiteral("label_7"));

        horizontalLayout_7->addWidget(label_7);

        comboTheme = new QComboBox(frame_6);
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

        horizontalLayout_7->addWidget(comboTheme);

        label_8 = new QLabel(frame_6);
        label_8->setObjectName(QStringLiteral("label_8"));

        horizontalLayout_7->addWidget(label_8);

        comboAnimation = new QComboBox(frame_6);
        comboAnimation->addItem(QString());
        comboAnimation->addItem(QString());
        comboAnimation->addItem(QString());
        comboAnimation->addItem(QString());
        comboAnimation->setObjectName(QStringLiteral("comboAnimation"));

        horizontalLayout_7->addWidget(comboAnimation);

        horizontalSpacer_7 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(horizontalSpacer_7);

        toolBtn_Quit = new QToolButton(frame_6);
        toolBtn_Quit->setObjectName(QStringLiteral("toolBtn_Quit"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        toolBtn_Quit->setIcon(icon2);
        toolBtn_Quit->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_7->addWidget(toolBtn_Quit);


        verticalLayout_13->addWidget(frame_6);

        splitter_2 = new QSplitter(centralWidget);
        splitter_2->setObjectName(QStringLiteral("splitter_2"));
        splitter_2->setOrientation(Qt::Horizontal);
        frameData = new QFrame(splitter_2);
        frameData->setObjectName(QStringLiteral("frameData"));
        frameData->setFrameShape(QFrame::Panel);
        frameData->setFrameShadow(QFrame::Raised);
        verticalLayout_3 = new QVBoxLayout(frameData);
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(5, 5, 5, 5);
        splitter = new QSplitter(frameData);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setOrientation(Qt::Vertical);
        groupBox = new QGroupBox(splitter);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        verticalLayout = new QVBoxLayout(groupBox);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(5, 5, 5, 5);
        frame_2 = new QFrame(groupBox);
        frame_2->setObjectName(QStringLiteral("frame_2"));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        horizontalLayout_6 = new QHBoxLayout(frame_2);
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(-1, 2, -1, 2);
        label_5 = new QLabel(frame_2);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_6->addWidget(label_5);

        spinCount = new QSpinBox(frame_2);
        spinCount->setObjectName(QStringLiteral("spinCount"));
        spinCount->setMinimum(5);
        spinCount->setMaximum(5000);
        spinCount->setValue(20);

        horizontalLayout_6->addWidget(spinCount);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_6);


        verticalLayout->addWidget(frame_2);

        tableView = new QTableView(groupBox);
        tableView->setObjectName(QStringLiteral("tableView"));
        tableView->setFont(font);
        tableView->setAlternatingRowColors(true);
        tableView->horizontalHeader()->setDefaultSectionSize(80);
        tableView->verticalHeader()->setDefaultSectionSize(25);

        verticalLayout->addWidget(tableView);

        splitter->addWidget(groupBox);
        groupBox_7 = new QGroupBox(splitter);
        groupBox_7->setObjectName(QStringLiteral("groupBox_7"));
        verticalLayout_2 = new QVBoxLayout(groupBox_7);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(5, 5, 5, 5);
        treeWidget = new QTreeWidget(groupBox_7);
        QTreeWidgetItem *__qtreewidgetitem = new QTreeWidgetItem();
        __qtreewidgetitem->setTextAlignment(3, Qt::AlignCenter);
        __qtreewidgetitem->setTextAlignment(2, Qt::AlignCenter);
        __qtreewidgetitem->setTextAlignment(1, Qt::AlignCenter);
        __qtreewidgetitem->setTextAlignment(0, Qt::AlignCenter);
        treeWidget->setHeaderItem(__qtreewidgetitem);
        QTreeWidgetItem *__qtreewidgetitem1 = new QTreeWidgetItem(treeWidget);
        __qtreewidgetitem1->setTextAlignment(1, Qt::AlignCenter);
        __qtreewidgetitem1->setTextAlignment(0, Qt::AlignCenter);
        QTreeWidgetItem *__qtreewidgetitem2 = new QTreeWidgetItem(treeWidget);
        __qtreewidgetitem2->setTextAlignment(1, Qt::AlignCenter);
        __qtreewidgetitem2->setTextAlignment(0, Qt::AlignCenter);
        QTreeWidgetItem *__qtreewidgetitem3 = new QTreeWidgetItem(treeWidget);
        __qtreewidgetitem3->setTextAlignment(0, Qt::AlignCenter);
        QTreeWidgetItem *__qtreewidgetitem4 = new QTreeWidgetItem(treeWidget);
        __qtreewidgetitem4->setTextAlignment(1, Qt::AlignCenter);
        __qtreewidgetitem4->setTextAlignment(0, Qt::AlignCenter);
        QTreeWidgetItem *__qtreewidgetitem5 = new QTreeWidgetItem(treeWidget);
        __qtreewidgetitem5->setTextAlignment(0, Qt::AlignCenter);
        treeWidget->setObjectName(QStringLiteral("treeWidget"));
        treeWidget->setMaximumSize(QSize(16777215, 300));
        treeWidget->setFont(font);
        treeWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
        treeWidget->setAlternatingRowColors(true);
        treeWidget->setIndentation(0);
        treeWidget->header()->setDefaultSectionSize(90);

        verticalLayout_2->addWidget(treeWidget);

        splitter->addWidget(groupBox_7);

        verticalLayout_3->addWidget(splitter);

        splitter_2->addWidget(frameData);
        tabWidget = new QTabWidget(splitter_2);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setTabPosition(QTabWidget::North);
        tabWidget->setTabShape(QTabWidget::Rounded);
        tabWidget->setDocumentMode(true);
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        verticalLayout_9 = new QVBoxLayout(tab);
        verticalLayout_9->setSpacing(6);
        verticalLayout_9->setContentsMargins(11, 11, 11, 11);
        verticalLayout_9->setObjectName(QStringLiteral("verticalLayout_9"));
        verticalLayout_9->setContentsMargins(2, 2, 2, 2);
        widget = new QWidget(tab);
        widget->setObjectName(QStringLiteral("widget"));
        verticalLayout_8 = new QVBoxLayout(widget);
        verticalLayout_8->setSpacing(0);
        verticalLayout_8->setContentsMargins(11, 11, 11, 11);
        verticalLayout_8->setObjectName(QStringLiteral("verticalLayout_8"));
        verticalLayout_8->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(widget);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setFrameShape(QFrame::Panel);
        frame->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frame);
        horizontalLayout->setSpacing(12);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 5, -1, 5);
        btnBuildBarChart = new QPushButton(frame);
        btnBuildBarChart->setObjectName(QStringLiteral("btnBuildBarChart"));

        horizontalLayout->addWidget(btnBuildBarChart);

        btnBuildBarChartH = new QPushButton(frame);
        btnBuildBarChartH->setObjectName(QStringLiteral("btnBuildBarChartH"));

        horizontalLayout->addWidget(btnBuildBarChartH);

        horizontalSpacer_2 = new QSpacerItem(662, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);


        verticalLayout_8->addWidget(frame);

        chartViewBar = new QChartView(widget);
        chartViewBar->setObjectName(QStringLiteral("chartViewBar"));
        chartViewBar->setRenderHints(QPainter::Antialiasing|QPainter::TextAntialiasing);

        verticalLayout_8->addWidget(chartViewBar);


        verticalLayout_9->addWidget(widget);

        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/3.ico"), QSize(), QIcon::Normal, QIcon::Off);
        tabWidget->addTab(tab, icon3, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QStringLiteral("tab_3"));
        verticalLayout_10 = new QVBoxLayout(tab_3);
        verticalLayout_10->setSpacing(6);
        verticalLayout_10->setContentsMargins(11, 11, 11, 11);
        verticalLayout_10->setObjectName(QStringLiteral("verticalLayout_10"));
        verticalLayout_10->setContentsMargins(2, 2, 2, 2);
        widget_2 = new QWidget(tab_3);
        widget_2->setObjectName(QStringLiteral("widget_2"));
        verticalLayout_5 = new QVBoxLayout(widget_2);
        verticalLayout_5->setSpacing(0);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(0, 0, 0, 0);
        frame_3 = new QFrame(widget_2);
        frame_3->setObjectName(QStringLiteral("frame_3"));
        frame_3->setFrameShape(QFrame::Panel);
        frame_3->setFrameShadow(QFrame::Raised);
        horizontalLayout_3 = new QHBoxLayout(frame_3);
        horizontalLayout_3->setSpacing(12);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(-1, 5, -1, 5);
        btnBuildStackedBar = new QPushButton(frame_3);
        btnBuildStackedBar->setObjectName(QStringLiteral("btnBuildStackedBar"));

        horizontalLayout_3->addWidget(btnBuildStackedBar);

        btnBuildStackedBarH = new QPushButton(frame_3);
        btnBuildStackedBarH->setObjectName(QStringLiteral("btnBuildStackedBarH"));

        horizontalLayout_3->addWidget(btnBuildStackedBarH);

        horizontalSpacer_3 = new QSpacerItem(536, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_3);


        verticalLayout_5->addWidget(frame_3);

        chartViewStackedBar = new QChartView(widget_2);
        chartViewStackedBar->setObjectName(QStringLiteral("chartViewStackedBar"));

        verticalLayout_5->addWidget(chartViewStackedBar);


        verticalLayout_10->addWidget(widget_2);

        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/281.GIF"), QSize(), QIcon::Normal, QIcon::Off);
        tabWidget->addTab(tab_3, icon4, QString());
        tab_5 = new QWidget();
        tab_5->setObjectName(QStringLiteral("tab_5"));
        verticalLayout_12 = new QVBoxLayout(tab_5);
        verticalLayout_12->setSpacing(6);
        verticalLayout_12->setContentsMargins(11, 11, 11, 11);
        verticalLayout_12->setObjectName(QStringLiteral("verticalLayout_12"));
        verticalLayout_12->setContentsMargins(2, 2, 2, 2);
        widget_4 = new QWidget(tab_5);
        widget_4->setObjectName(QStringLiteral("widget_4"));
        verticalLayout_6 = new QVBoxLayout(widget_4);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setContentsMargins(11, 11, 11, 11);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
        frame_5 = new QFrame(widget_4);
        frame_5->setObjectName(QStringLiteral("frame_5"));
        frame_5->setFrameShape(QFrame::Panel);
        frame_5->setFrameShadow(QFrame::Raised);
        horizontalLayout_4 = new QHBoxLayout(frame_5);
        horizontalLayout_4->setSpacing(12);
        horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(-1, 5, -1, 5);
        btnPercentBar = new QPushButton(frame_5);
        btnPercentBar->setObjectName(QStringLiteral("btnPercentBar"));

        horizontalLayout_4->addWidget(btnPercentBar);

        btnPercentBarH = new QPushButton(frame_5);
        btnPercentBarH->setObjectName(QStringLiteral("btnPercentBarH"));

        horizontalLayout_4->addWidget(btnPercentBarH);

        horizontalSpacer_4 = new QSpacerItem(523, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_4);


        verticalLayout_6->addWidget(frame_5);

        chartViewPercentBar = new QChartView(widget_4);
        chartViewPercentBar->setObjectName(QStringLiteral("chartViewPercentBar"));

        verticalLayout_6->addWidget(chartViewPercentBar);


        verticalLayout_12->addWidget(widget_4);

        QIcon icon5;
        icon5.addFile(QStringLiteral(":/icons/images/f4.ico"), QSize(), QIcon::Normal, QIcon::Off);
        tabWidget->addTab(tab_5, icon5, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        verticalLayout_11 = new QVBoxLayout(tab_2);
        verticalLayout_11->setSpacing(6);
        verticalLayout_11->setContentsMargins(11, 11, 11, 11);
        verticalLayout_11->setObjectName(QStringLiteral("verticalLayout_11"));
        verticalLayout_11->setContentsMargins(2, 2, 2, 2);
        widget_3 = new QWidget(tab_2);
        widget_3->setObjectName(QStringLiteral("widget_3"));
        verticalLayout_4 = new QVBoxLayout(widget_3);
        verticalLayout_4->setSpacing(0);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(0, 0, 0, 0);
        frame_4 = new QFrame(widget_3);
        frame_4->setObjectName(QStringLiteral("frame_4"));
        frame_4->setFrameShape(QFrame::Panel);
        frame_4->setFrameShadow(QFrame::Raised);
        horizontalLayout_2 = new QHBoxLayout(frame_4);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(-1, 5, -1, 5);
        label = new QLabel(frame_4);
        label->setObjectName(QStringLiteral("label"));
        label->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_2->addWidget(label);

        comboCourse = new QComboBox(frame_4);
        comboCourse->addItem(QString());
        comboCourse->addItem(QString());
        comboCourse->addItem(QString());
        comboCourse->setObjectName(QStringLiteral("comboCourse"));
        comboCourse->setMinimumSize(QSize(60, 0));

        horizontalLayout_2->addWidget(comboCourse);

        btnDrawPieChart = new QPushButton(frame_4);
        btnDrawPieChart->setObjectName(QStringLiteral("btnDrawPieChart"));

        horizontalLayout_2->addWidget(btnDrawPieChart);

        label_2 = new QLabel(frame_4);
        label_2->setObjectName(QStringLiteral("label_2"));

        horizontalLayout_2->addWidget(label_2);

        spinHoleSize = new QDoubleSpinBox(frame_4);
        spinHoleSize->setObjectName(QStringLiteral("spinHoleSize"));
        spinHoleSize->setMinimumSize(QSize(70, 0));
        spinHoleSize->setDecimals(2);
        spinHoleSize->setMaximum(1);
        spinHoleSize->setSingleStep(0.1);

        horizontalLayout_2->addWidget(spinHoleSize);

        label_3 = new QLabel(frame_4);
        label_3->setObjectName(QStringLiteral("label_3"));

        horizontalLayout_2->addWidget(label_3);

        spinPieSize = new QDoubleSpinBox(frame_4);
        spinPieSize->setObjectName(QStringLiteral("spinPieSize"));
        spinPieSize->setMinimumSize(QSize(70, 0));
        spinPieSize->setMaximum(1);
        spinPieSize->setSingleStep(0.1);
        spinPieSize->setValue(0.7);

        horizontalLayout_2->addWidget(spinPieSize);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);


        verticalLayout_4->addWidget(frame_4);

        chartViewPie = new QChartView(widget_3);
        chartViewPie->setObjectName(QStringLiteral("chartViewPie"));

        verticalLayout_4->addWidget(chartViewPie);


        verticalLayout_11->addWidget(widget_3);

        QIcon icon6;
        icon6.addFile(QStringLiteral(":/icons/images/43.ico"), QSize(), QIcon::Normal, QIcon::Off);
        tabWidget->addTab(tab_2, icon6, QString());
        splitter_2->addWidget(tabWidget);

        verticalLayout_13->addWidget(splitter_2);

        MainWindow->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);
        QObject::connect(toolBtn_Quit, SIGNAL(clicked()), MainWindow, SLOT(close()));

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "\351\245\274\345\233\276\345\222\214\346\237\261\347\212\266\345\233\276", nullptr));
#ifndef QT_NO_TOOLTIP
        toolBtn_GenData->setToolTip(QApplication::translate("MainWindow", "\351\207\215\346\226\260\347\224\237\346\210\220\346\225\260\346\215\256\345\271\266\347\273\237\350\256\241", nullptr));
#endif // QT_NO_TOOLTIP
        toolBtn_GenData->setText(QApplication::translate("MainWindow", "\351\207\215\346\226\260\347\224\237\346\210\220\346\225\260\346\215\256", nullptr));
#ifndef QT_NO_TOOLTIP
        toolBtn_Counting->setToolTip(QApplication::translate("MainWindow", "\351\207\215\346\226\260\347\273\237\350\256\241", nullptr));
#endif // QT_NO_TOOLTIP
        toolBtn_Counting->setText(QApplication::translate("MainWindow", "\351\207\215\346\226\260\347\273\237\350\256\241", nullptr));
        label_7->setText(QApplication::translate("MainWindow", "\345\233\276\350\241\250\344\270\273\351\242\230", nullptr));
        comboTheme->setItemText(0, QApplication::translate("MainWindow", "Light", nullptr));
        comboTheme->setItemText(1, QApplication::translate("MainWindow", "BlueCerulean", nullptr));
        comboTheme->setItemText(2, QApplication::translate("MainWindow", "Dark", nullptr));
        comboTheme->setItemText(3, QApplication::translate("MainWindow", "BrownSand", nullptr));
        comboTheme->setItemText(4, QApplication::translate("MainWindow", "BlueNcs", nullptr));
        comboTheme->setItemText(5, QApplication::translate("MainWindow", "HighContrast", nullptr));
        comboTheme->setItemText(6, QApplication::translate("MainWindow", "BlueIcy", nullptr));
        comboTheme->setItemText(7, QApplication::translate("MainWindow", "Qt", nullptr));

        label_8->setText(QApplication::translate("MainWindow", "\345\233\276\350\241\250\345\212\250\347\224\273\346\225\210\346\236\234", nullptr));
        comboAnimation->setItemText(0, QApplication::translate("MainWindow", "NoAnimation", nullptr));
        comboAnimation->setItemText(1, QApplication::translate("MainWindow", "GridAxisAnimations", nullptr));
        comboAnimation->setItemText(2, QApplication::translate("MainWindow", "SeriesAnimations", nullptr));
        comboAnimation->setItemText(3, QApplication::translate("MainWindow", "AllAnimations", nullptr));

        toolBtn_Quit->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
        groupBox->setTitle(QApplication::translate("MainWindow", "\345\216\237\345\247\213\345\210\206\346\225\260", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "\345\255\246\347\224\237\344\272\272\346\225\260", nullptr));
        groupBox_7->setTitle(QApplication::translate("MainWindow", "\344\272\272\346\225\260\347\273\237\350\256\241", nullptr));
        QTreeWidgetItem *___qtreewidgetitem = treeWidget->headerItem();
        ___qtreewidgetitem->setText(3, QApplication::translate("MainWindow", "\350\213\261\350\257\255", nullptr));
        ___qtreewidgetitem->setText(2, QApplication::translate("MainWindow", "\350\257\255\346\226\207", nullptr));
        ___qtreewidgetitem->setText(1, QApplication::translate("MainWindow", "\346\225\260\345\255\246", nullptr));
        ___qtreewidgetitem->setText(0, QApplication::translate("MainWindow", "\345\210\206\346\225\260\346\256\265", nullptr));

        const bool __sortingEnabled = treeWidget->isSortingEnabled();
        treeWidget->setSortingEnabled(false);
        QTreeWidgetItem *___qtreewidgetitem1 = treeWidget->topLevelItem(0);
        ___qtreewidgetitem1->setText(0, QApplication::translate("MainWindow", "60\345\210\206\344\273\245\344\270\213", nullptr));
        QTreeWidgetItem *___qtreewidgetitem2 = treeWidget->topLevelItem(1);
        ___qtreewidgetitem2->setText(0, QApplication::translate("MainWindow", "60-69\345\210\206", nullptr));
        QTreeWidgetItem *___qtreewidgetitem3 = treeWidget->topLevelItem(2);
        ___qtreewidgetitem3->setText(0, QApplication::translate("MainWindow", "70-79\345\210\206", nullptr));
        QTreeWidgetItem *___qtreewidgetitem4 = treeWidget->topLevelItem(3);
        ___qtreewidgetitem4->setText(0, QApplication::translate("MainWindow", "80-89\345\210\206", nullptr));
        QTreeWidgetItem *___qtreewidgetitem5 = treeWidget->topLevelItem(4);
        ___qtreewidgetitem5->setText(0, QApplication::translate("MainWindow", "90\345\210\206\344\273\245\344\270\212", nullptr));
        treeWidget->setSortingEnabled(__sortingEnabled);

        btnBuildBarChart->setText(QApplication::translate("MainWindow", "\347\273\230\345\210\266\346\237\261\347\212\266\345\233\276", nullptr));
        btnBuildBarChartH->setText(QApplication::translate("MainWindow", "\347\273\230\345\210\266\346\260\264\345\271\263\346\237\261\347\212\266\345\233\276", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("MainWindow", "BarChart", nullptr));
        btnBuildStackedBar->setText(QApplication::translate("MainWindow", "\347\273\230\345\210\266\345\240\206\345\217\240\346\237\261\347\212\266\345\233\276", nullptr));
        btnBuildStackedBarH->setText(QApplication::translate("MainWindow", "\347\273\230\345\210\266\346\260\264\345\271\263\345\240\206\345\217\240\346\237\261\347\212\266\345\233\276", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QApplication::translate("MainWindow", "StackedBar", nullptr));
        btnPercentBar->setText(QApplication::translate("MainWindow", "\347\273\230\345\210\266\347\231\276\345\210\206\346\257\224\346\237\261\347\212\266\345\233\276", nullptr));
        btnPercentBarH->setText(QApplication::translate("MainWindow", "\347\273\230\345\210\266\346\260\264\345\271\263\347\231\276\345\210\206\346\257\224\346\237\261\347\212\266\345\233\276", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_5), QApplication::translate("MainWindow", "PercentBar", nullptr));
        label->setText(QApplication::translate("MainWindow", "\345\210\206\346\236\220\346\225\260\346\215\256", nullptr));
        comboCourse->setItemText(0, QApplication::translate("MainWindow", "\346\225\260\345\255\246", nullptr));
        comboCourse->setItemText(1, QApplication::translate("MainWindow", "\350\257\255\346\226\207", nullptr));
        comboCourse->setItemText(2, QApplication::translate("MainWindow", "\350\213\261\350\257\255", nullptr));

        btnDrawPieChart->setText(QApplication::translate("MainWindow", "\347\273\230\345\210\266\351\245\274\345\233\276", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "HoleSize", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "PieSize", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("MainWindow", "PieChart", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
