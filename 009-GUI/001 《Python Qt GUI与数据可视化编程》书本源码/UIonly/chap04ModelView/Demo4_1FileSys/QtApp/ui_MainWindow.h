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
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QTableView>
#include <QtWidgets/QTreeView>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout_5;
    QSplitter *splitterMain;
    QGroupBox *groupBox_3;
    QVBoxLayout *verticalLayout_3;
    QTreeView *treeView;
    QSplitter *splitter;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout;
    QListView *listView;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_2;
    QTableView *tableView;
    QGroupBox *groupBox_4;
    QVBoxLayout *verticalLayout_4;
    QHBoxLayout *horizontalLayout;
    QLabel *LabFileName;
    QLabel *LabFileSize;
    QLabel *LabType;
    QCheckBox *chkBox_IsDir;
    QFrame *line;
    QLabel *LabPath;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(876, 513);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout_5 = new QVBoxLayout(centralWidget);
        verticalLayout_5->setSpacing(3);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(3, 3, 3, 3);
        splitterMain = new QSplitter(centralWidget);
        splitterMain->setObjectName(QStringLiteral("splitterMain"));
        splitterMain->setOrientation(Qt::Horizontal);
        groupBox_3 = new QGroupBox(splitterMain);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        verticalLayout_3 = new QVBoxLayout(groupBox_3);
        verticalLayout_3->setSpacing(4);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(4, 4, 4, 4);
        treeView = new QTreeView(groupBox_3);
        treeView->setObjectName(QStringLiteral("treeView"));
        treeView->setEditTriggers(QAbstractItemView::NoEditTriggers);

        verticalLayout_3->addWidget(treeView);

        splitterMain->addWidget(groupBox_3);
        splitter = new QSplitter(splitterMain);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setOrientation(Qt::Vertical);
        groupBox = new QGroupBox(splitter);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        verticalLayout = new QVBoxLayout(groupBox);
        verticalLayout->setSpacing(4);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(4, 4, 4, 4);
        listView = new QListView(groupBox);
        listView->setObjectName(QStringLiteral("listView"));
        listView->setEditTriggers(QAbstractItemView::NoEditTriggers);

        verticalLayout->addWidget(listView);

        splitter->addWidget(groupBox);
        groupBox_2 = new QGroupBox(splitter);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        verticalLayout_2 = new QVBoxLayout(groupBox_2);
        verticalLayout_2->setSpacing(3);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(4, 4, 4, 4);
        tableView = new QTableView(groupBox_2);
        tableView->setObjectName(QStringLiteral("tableView"));
        tableView->setEditTriggers(QAbstractItemView::NoEditTriggers);
        tableView->setShowGrid(true);
        tableView->setGridStyle(Qt::DotLine);

        verticalLayout_2->addWidget(tableView);

        splitter->addWidget(groupBox_2);
        splitterMain->addWidget(splitter);

        verticalLayout_5->addWidget(splitterMain);

        groupBox_4 = new QGroupBox(centralWidget);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        groupBox_4->setMaximumSize(QSize(16777215, 75));
        verticalLayout_4 = new QVBoxLayout(groupBox_4);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(-1, 5, -1, 2);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        LabFileName = new QLabel(groupBox_4);
        LabFileName->setObjectName(QStringLiteral("LabFileName"));

        horizontalLayout->addWidget(LabFileName);

        LabFileSize = new QLabel(groupBox_4);
        LabFileSize->setObjectName(QStringLiteral("LabFileSize"));

        horizontalLayout->addWidget(LabFileSize);

        LabType = new QLabel(groupBox_4);
        LabType->setObjectName(QStringLiteral("LabType"));

        horizontalLayout->addWidget(LabType);

        chkBox_IsDir = new QCheckBox(groupBox_4);
        chkBox_IsDir->setObjectName(QStringLiteral("chkBox_IsDir"));

        horizontalLayout->addWidget(chkBox_IsDir);


        verticalLayout_4->addLayout(horizontalLayout);

        line = new QFrame(groupBox_4);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        verticalLayout_4->addWidget(line);

        LabPath = new QLabel(groupBox_4);
        LabPath->setObjectName(QStringLiteral("LabPath"));

        verticalLayout_4->addWidget(LabPath);


        verticalLayout_5->addWidget(groupBox_4);

        MainWindow->setCentralWidget(centralWidget);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Demo4_1, \346\226\207\344\273\266\347\263\273\347\273\237Model/View", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "TreeView", nullptr));
        groupBox->setTitle(QApplication::translate("MainWindow", "ListView", nullptr));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "TableView", nullptr));
        groupBox_4->setTitle(QString());
        LabFileName->setText(QApplication::translate("MainWindow", "\346\226\207\344\273\266\345\220\215\357\274\232", nullptr));
        LabFileSize->setText(QApplication::translate("MainWindow", "\346\226\207\344\273\266\345\244\247\345\260\217\357\274\232", nullptr));
        LabType->setText(QApplication::translate("MainWindow", "\350\212\202\347\202\271\347\261\273\345\236\213\357\274\232", nullptr));
        chkBox_IsDir->setText(QApplication::translate("MainWindow", "\350\212\202\347\202\271\346\230\257\347\233\256\345\275\225", nullptr));
        LabPath->setText(QApplication::translate("MainWindow", "\350\267\257\345\276\204\345\220\215\357\274\232", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
