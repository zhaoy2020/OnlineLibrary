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
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLayout>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolBox>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actList_Ini;
    QAction *actList_Clear;
    QAction *actList_Insert;
    QAction *actList_Append;
    QAction *actList_Delete;
    QAction *actSel_ALL;
    QAction *actSel_None;
    QAction *actSel_Invs;
    QAction *actQuit;
    QAction *actSelPopMenu;
    QWidget *centralWidget;
    QSplitter *splitter;
    QToolBox *toolBox;
    QWidget *page;
    QGridLayout *gridLayout;
    QToolButton *btnList_Ini;
    QToolButton *btnList_Clear;
    QToolButton *btnList_Insert;
    QToolButton *btnList_Append;
    QToolButton *btnList_Delete;
    QWidget *page_2;
    QWidget *page_3;
    QTabWidget *tabWidget;
    QWidget *tab_1;
    QVBoxLayout *verticalLayout_5;
    QFrame *frame;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *editCurItemText;
    QCheckBox *chkBoxList_Editable;
    QGroupBox *groupBox_2;
    QHBoxLayout *horizontalLayout_2;
    QToolButton *btnSelectItem;
    QToolButton *btnSel_ALL;
    QToolButton *btnSel_None;
    QToolButton *btnSel_Invs;
    QListWidget *listWidget;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_2;
    QWidget *tab_3;
    QPushButton *pushButton;
    QToolBar *mainToolBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(898, 492);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        actList_Ini = new QAction(MainWindow);
        actList_Ini->setObjectName(QStringLiteral("actList_Ini"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/128.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actList_Ini->setIcon(icon);
        actList_Clear = new QAction(MainWindow);
        actList_Clear->setObjectName(QStringLiteral("actList_Clear"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/delete1.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actList_Clear->setIcon(icon1);
        actList_Insert = new QAction(MainWindow);
        actList_Insert->setObjectName(QStringLiteral("actList_Insert"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/424.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actList_Insert->setIcon(icon2);
        actList_Append = new QAction(MainWindow);
        actList_Append->setObjectName(QStringLiteral("actList_Append"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/316.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actList_Append->setIcon(icon3);
        actList_Delete = new QAction(MainWindow);
        actList_Delete->setObjectName(QStringLiteral("actList_Delete"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/324.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actList_Delete->setIcon(icon4);
        actSel_ALL = new QAction(MainWindow);
        actSel_ALL->setObjectName(QStringLiteral("actSel_ALL"));
        actSel_None = new QAction(MainWindow);
        actSel_None->setObjectName(QStringLiteral("actSel_None"));
        actSel_None->setMenuRole(QAction::AboutRole);
        actSel_Invs = new QAction(MainWindow);
        actSel_Invs->setObjectName(QStringLiteral("actSel_Invs"));
        actQuit = new QAction(MainWindow);
        actQuit->setObjectName(QStringLiteral("actQuit"));
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actQuit->setIcon(icon5);
        actSelPopMenu = new QAction(MainWindow);
        actSelPopMenu->setObjectName(QStringLiteral("actSelPopMenu"));
        QIcon icon6;
        icon6.addFile(QStringLiteral(":/icons/images/124.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actSelPopMenu->setIcon(icon6);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        splitter = new QSplitter(centralWidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setGeometry(QRect(6, 6, 601, 301));
        splitter->setOrientation(Qt::Horizontal);
        toolBox = new QToolBox(splitter);
        toolBox->setObjectName(QStringLiteral("toolBox"));
        toolBox->setMinimumSize(QSize(150, 0));
        toolBox->setMaximumSize(QSize(220, 16777215));
        toolBox->setFrameShape(QFrame::Panel);
        page = new QWidget();
        page->setObjectName(QStringLiteral("page"));
        page->setGeometry(QRect(0, 0, 148, 209));
        gridLayout = new QGridLayout(page);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        btnList_Ini = new QToolButton(page);
        btnList_Ini->setObjectName(QStringLiteral("btnList_Ini"));
        btnList_Ini->setMinimumSize(QSize(120, 25));
        btnList_Ini->setPopupMode(QToolButton::DelayedPopup);
        btnList_Ini->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        btnList_Ini->setAutoRaise(false);
        btnList_Ini->setArrowType(Qt::NoArrow);

        gridLayout->addWidget(btnList_Ini, 0, 0, 1, 1);

        btnList_Clear = new QToolButton(page);
        btnList_Clear->setObjectName(QStringLiteral("btnList_Clear"));
        btnList_Clear->setMinimumSize(QSize(120, 25));
        btnList_Clear->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        gridLayout->addWidget(btnList_Clear, 1, 0, 1, 1);

        btnList_Insert = new QToolButton(page);
        btnList_Insert->setObjectName(QStringLiteral("btnList_Insert"));
        btnList_Insert->setMinimumSize(QSize(120, 25));
        btnList_Insert->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        gridLayout->addWidget(btnList_Insert, 2, 0, 1, 1);

        btnList_Append = new QToolButton(page);
        btnList_Append->setObjectName(QStringLiteral("btnList_Append"));
        btnList_Append->setMinimumSize(QSize(120, 25));
        btnList_Append->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        gridLayout->addWidget(btnList_Append, 3, 0, 1, 1);

        btnList_Delete = new QToolButton(page);
        btnList_Delete->setObjectName(QStringLiteral("btnList_Delete"));
        btnList_Delete->setMinimumSize(QSize(120, 25));
        btnList_Delete->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        gridLayout->addWidget(btnList_Delete, 4, 0, 1, 1);

        QIcon icon7;
        icon7.addFile(QStringLiteral(":/icons/images/408.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        toolBox->addItem(page, icon7, QString::fromUtf8("QListWidget\346\223\215\344\275\234"));
        page_2 = new QWidget();
        page_2->setObjectName(QStringLiteral("page_2"));
        page_2->setGeometry(QRect(0, 0, 148, 209));
        QIcon icon8;
        icon8.addFile(QStringLiteral(":/icons/images/410.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        toolBox->addItem(page_2, icon8, QString::fromUtf8("QTreeWidget[\347\251\272]"));
        page_3 = new QWidget();
        page_3->setObjectName(QStringLiteral("page_3"));
        page_3->setGeometry(QRect(0, 0, 148, 209));
        QIcon icon9;
        icon9.addFile(QStringLiteral(":/icons/images/412.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        toolBox->addItem(page_3, icon9, QString::fromUtf8("QTableWidget[\347\251\272]"));
        splitter->addWidget(toolBox);
        tabWidget = new QTabWidget(splitter);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setTabPosition(QTabWidget::North);
        tabWidget->setTabShape(QTabWidget::Rounded);
        tabWidget->setDocumentMode(true);
        tab_1 = new QWidget();
        tab_1->setObjectName(QStringLiteral("tab_1"));
        verticalLayout_5 = new QVBoxLayout(tab_1);
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(2, 2, 2, 2);
        frame = new QFrame(tab_1);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setFrameShape(QFrame::Panel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frame);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(5, 5, 5, 5);
        groupBox = new QGroupBox(frame);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        horizontalLayout = new QHBoxLayout(groupBox);
        horizontalLayout->setSpacing(10);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 2, -1, 2);
        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));

        horizontalLayout->addWidget(label);

        editCurItemText = new QLineEdit(groupBox);
        editCurItemText->setObjectName(QStringLiteral("editCurItemText"));
        editCurItemText->setStyleSheet(QStringLiteral("color: rgb(0, 0, 255);"));
        editCurItemText->setClearButtonEnabled(false);

        horizontalLayout->addWidget(editCurItemText);

        chkBoxList_Editable = new QCheckBox(groupBox);
        chkBoxList_Editable->setObjectName(QStringLiteral("chkBoxList_Editable"));
        chkBoxList_Editable->setChecked(true);

        horizontalLayout->addWidget(chkBoxList_Editable);


        verticalLayout->addWidget(groupBox);

        groupBox_2 = new QGroupBox(frame);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        horizontalLayout_2 = new QHBoxLayout(groupBox_2);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(-1, 2, -1, 2);
        btnSelectItem = new QToolButton(groupBox_2);
        btnSelectItem->setObjectName(QStringLiteral("btnSelectItem"));
        btnSelectItem->setMinimumSize(QSize(100, 25));
        btnSelectItem->setPopupMode(QToolButton::MenuButtonPopup);
        btnSelectItem->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

        horizontalLayout_2->addWidget(btnSelectItem);

        btnSel_ALL = new QToolButton(groupBox_2);
        btnSel_ALL->setObjectName(QStringLiteral("btnSel_ALL"));
        btnSel_ALL->setMinimumSize(QSize(70, 25));

        horizontalLayout_2->addWidget(btnSel_ALL);

        btnSel_None = new QToolButton(groupBox_2);
        btnSel_None->setObjectName(QStringLiteral("btnSel_None"));
        btnSel_None->setMinimumSize(QSize(70, 25));

        horizontalLayout_2->addWidget(btnSel_None);

        btnSel_Invs = new QToolButton(groupBox_2);
        btnSel_Invs->setObjectName(QStringLiteral("btnSel_Invs"));
        btnSel_Invs->setMinimumSize(QSize(70, 25));

        horizontalLayout_2->addWidget(btnSel_Invs);


        verticalLayout->addWidget(groupBox_2);

        listWidget = new QListWidget(frame);
        QIcon icon10;
        icon10.addFile(QStringLiteral(":/icons/images/724.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        QListWidgetItem *__qlistwidgetitem = new QListWidgetItem(listWidget);
        __qlistwidgetitem->setCheckState(Qt::Checked);
        __qlistwidgetitem->setIcon(icon10);
        __qlistwidgetitem->setFlags(Qt::ItemIsSelectable|Qt::ItemIsEditable|Qt::ItemIsDragEnabled|Qt::ItemIsUserCheckable|Qt::ItemIsEnabled);
        QListWidgetItem *__qlistwidgetitem1 = new QListWidgetItem(listWidget);
        __qlistwidgetitem1->setCheckState(Qt::Unchecked);
        __qlistwidgetitem1->setIcon(icon10);
        __qlistwidgetitem1->setFlags(Qt::ItemIsSelectable|Qt::ItemIsEditable|Qt::ItemIsUserCheckable|Qt::ItemIsEnabled);
        listWidget->setObjectName(QStringLiteral("listWidget"));
        listWidget->setContextMenuPolicy(Qt::CustomContextMenu);
        listWidget->setLayoutMode(QListView::SinglePass);
        listWidget->setViewMode(QListView::ListMode);
        listWidget->setSelectionRectVisible(false);

        verticalLayout->addWidget(listWidget);


        verticalLayout_5->addWidget(frame);

        tabWidget->addTab(tab_1, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        verticalLayout_2 = new QVBoxLayout(tab_2);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        tabWidget->addTab(tab_2, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QStringLiteral("tab_3"));
        tabWidget->addTab(tab_3, QString());
        splitter->addWidget(tabWidget);
        pushButton = new QPushButton(centralWidget);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(695, 270, 75, 23));
        MainWindow->setCentralWidget(centralWidget);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);

        mainToolBar->addAction(actList_Ini);
        mainToolBar->addAction(actList_Clear);
        mainToolBar->addAction(actList_Insert);
        mainToolBar->addAction(actList_Append);
        mainToolBar->addAction(actList_Delete);

        retranslateUi(MainWindow);
        QObject::connect(actQuit, SIGNAL(triggered()), MainWindow, SLOT(close()));
        QObject::connect(actSelPopMenu, SIGNAL(triggered()), actSel_Invs, SLOT(trigger()));

        toolBox->setCurrentIndex(0);
        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Demo3_8 QListWidget\345\222\214QToolButton", nullptr));
        actList_Ini->setText(QApplication::translate("MainWindow", "\345\210\235\345\247\213\345\214\226\345\210\227\350\241\250", nullptr));
#ifndef QT_NO_TOOLTIP
        actList_Ini->setToolTip(QApplication::translate("MainWindow", "\345\210\235\345\247\213\345\214\226\345\210\227\350\241\250", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actList_Ini->setShortcut(QApplication::translate("MainWindow", "Ctrl+I", nullptr));
#endif // QT_NO_SHORTCUT
        actList_Clear->setText(QApplication::translate("MainWindow", "\346\270\205\351\231\244\345\210\227\350\241\250", nullptr));
#ifndef QT_NO_TOOLTIP
        actList_Clear->setToolTip(QApplication::translate("MainWindow", "\346\270\205\351\231\244\345\210\227\350\241\250", nullptr));
#endif // QT_NO_TOOLTIP
        actList_Insert->setText(QApplication::translate("MainWindow", "\346\217\222\345\205\245\351\241\271", nullptr));
#ifndef QT_NO_TOOLTIP
        actList_Insert->setToolTip(QApplication::translate("MainWindow", "\346\217\222\345\205\245\351\241\271", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actList_Insert->setShortcut(QApplication::translate("MainWindow", "Ctrl+S", nullptr));
#endif // QT_NO_SHORTCUT
        actList_Append->setText(QApplication::translate("MainWindow", "\346\267\273\345\212\240\351\241\271", nullptr));
#ifndef QT_NO_TOOLTIP
        actList_Append->setToolTip(QApplication::translate("MainWindow", "\346\267\273\345\212\240\351\241\271", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actList_Append->setShortcut(QApplication::translate("MainWindow", "Ctrl+A", nullptr));
#endif // QT_NO_SHORTCUT
        actList_Delete->setText(QApplication::translate("MainWindow", "\345\210\240\351\231\244\345\275\223\345\211\215\351\241\271", nullptr));
#ifndef QT_NO_TOOLTIP
        actList_Delete->setToolTip(QApplication::translate("MainWindow", "\345\210\240\351\231\244\345\275\223\345\211\215\351\241\271", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actList_Delete->setShortcut(QApplication::translate("MainWindow", "Del", nullptr));
#endif // QT_NO_SHORTCUT
        actSel_ALL->setText(QApplication::translate("MainWindow", "\345\205\250\351\200\211", nullptr));
#ifndef QT_NO_TOOLTIP
        actSel_ALL->setToolTip(QApplication::translate("MainWindow", "\345\205\250\351\200\211", nullptr));
#endif // QT_NO_TOOLTIP
        actSel_None->setText(QApplication::translate("MainWindow", "\345\205\250\344\270\215\351\200\211", nullptr));
#ifndef QT_NO_TOOLTIP
        actSel_None->setToolTip(QApplication::translate("MainWindow", "\345\205\250\344\270\215\351\200\211", nullptr));
#endif // QT_NO_TOOLTIP
        actSel_Invs->setText(QApplication::translate("MainWindow", "\345\217\215\351\200\211", nullptr));
#ifndef QT_NO_TOOLTIP
        actSel_Invs->setToolTip(QApplication::translate("MainWindow", "\345\217\215\351\200\211", nullptr));
#endif // QT_NO_TOOLTIP
        actQuit->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actQuit->setToolTip(QApplication::translate("MainWindow", "\351\200\200\345\207\272\347\250\213\345\272\217", nullptr));
#endif // QT_NO_TOOLTIP
        actSelPopMenu->setText(QApplication::translate("MainWindow", "\351\241\271\351\200\211\346\213\251", nullptr));
#ifndef QT_NO_TOOLTIP
        actSelPopMenu->setToolTip(QApplication::translate("MainWindow", "\351\241\271\351\200\211\346\213\251", nullptr));
#endif // QT_NO_TOOLTIP
        btnList_Ini->setText(QApplication::translate("MainWindow", "btnList_Ini", nullptr));
        btnList_Clear->setText(QApplication::translate("MainWindow", "btnList_Clear", nullptr));
        btnList_Insert->setText(QApplication::translate("MainWindow", "btnList_Insert", nullptr));
        btnList_Append->setText(QApplication::translate("MainWindow", "btnList_Append", nullptr));
        btnList_Delete->setText(QApplication::translate("MainWindow", "btnList_Delete", nullptr));
        toolBox->setItemText(toolBox->indexOf(page), QApplication::translate("MainWindow", "QListWidget\346\223\215\344\275\234", nullptr));
        toolBox->setItemText(toolBox->indexOf(page_2), QApplication::translate("MainWindow", "QTreeWidget[\347\251\272]", nullptr));
        toolBox->setItemText(toolBox->indexOf(page_3), QApplication::translate("MainWindow", "QTableWidget[\347\251\272]", nullptr));
        groupBox->setTitle(QString());
        label->setText(QApplication::translate("MainWindow", "\345\275\223\345\211\215\351\241\271\345\217\230\345\214\226", nullptr));
        chkBoxList_Editable->setText(QApplication::translate("MainWindow", "\345\217\257\347\274\226\350\276\221", nullptr));
        groupBox_2->setTitle(QString());
        btnSelectItem->setText(QApplication::translate("MainWindow", "btnSelectItem", nullptr));
        btnSel_ALL->setText(QApplication::translate("MainWindow", "btnSel_ALL", nullptr));
        btnSel_None->setText(QApplication::translate("MainWindow", "btnSel_None", nullptr));
        btnSel_Invs->setText(QApplication::translate("MainWindow", "btnSel_Invs", nullptr));

        const bool __sortingEnabled = listWidget->isSortingEnabled();
        listWidget->setSortingEnabled(false);
        QListWidgetItem *___qlistwidgetitem = listWidget->item(0);
        ___qlistwidgetitem->setText(QApplication::translate("MainWindow", "New Item", nullptr));
        QListWidgetItem *___qlistwidgetitem1 = listWidget->item(1);
        ___qlistwidgetitem1->setText(QApplication::translate("MainWindow", "New Item2", nullptr));
        listWidget->setSortingEnabled(__sortingEnabled);

        tabWidget->setTabText(tabWidget->indexOf(tab_1), QApplication::translate("MainWindow", "QListWidget", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("MainWindow", "QTreeWidget[\347\251\272]", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QApplication::translate("MainWindow", "QTableWidget[\347\251\272]", nullptr));
        pushButton->setText(QApplication::translate("MainWindow", "PushButton", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
