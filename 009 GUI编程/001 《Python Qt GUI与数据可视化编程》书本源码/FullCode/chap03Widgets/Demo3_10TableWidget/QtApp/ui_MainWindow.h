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
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QSplitter *splitterMain;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QPushButton *btnInsertRow;
    QPushButton *btnAppendRow;
    QCheckBox *chkBoxHeaderH;
    QCheckBox *chkBoxEditable;
    QPushButton *btnReadToText;
    QCheckBox *chkBoxRowColor;
    QRadioButton *radioSelectItem;
    QCheckBox *chkBoxHeaderV;
    QRadioButton *radioSelectRow;
    QPushButton *btnIniData;
    QSpinBox *spinRowCount;
    QPushButton *btnSetHeader;
    QPushButton *btnSetRows;
    QPushButton *btnAutoHeight;
    QPushButton *btnAutoWidth;
    QPushButton *btnClearContents;
    QPushButton *btnDelCurRow;
    QSplitter *splitter;
    QTableWidget *tableInfo;
    QPlainTextEdit *textEdit;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(837, 471);
        QFont font;
        font.setPointSize(10);
        MainWindow->setFont(font);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(5, 5, 5, 5);
        splitterMain = new QSplitter(centralWidget);
        splitterMain->setObjectName(QStringLiteral("splitterMain"));
        splitterMain->setOrientation(Qt::Horizontal);
        groupBox = new QGroupBox(splitterMain);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setMaximumSize(QSize(300, 16777215));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        btnInsertRow = new QPushButton(groupBox);
        btnInsertRow->setObjectName(QStringLiteral("btnInsertRow"));

        gridLayout->addWidget(btnInsertRow, 3, 0, 1, 1);

        btnAppendRow = new QPushButton(groupBox);
        btnAppendRow->setObjectName(QStringLiteral("btnAppendRow"));

        gridLayout->addWidget(btnAppendRow, 3, 1, 1, 1);

        chkBoxHeaderH = new QCheckBox(groupBox);
        chkBoxHeaderH->setObjectName(QStringLiteral("chkBoxHeaderH"));
        chkBoxHeaderH->setChecked(true);

        gridLayout->addWidget(chkBoxHeaderH, 9, 0, 1, 1);

        chkBoxEditable = new QCheckBox(groupBox);
        chkBoxEditable->setObjectName(QStringLiteral("chkBoxEditable"));
        chkBoxEditable->setChecked(true);

        gridLayout->addWidget(chkBoxEditable, 8, 0, 1, 1);

        btnReadToText = new QPushButton(groupBox);
        btnReadToText->setObjectName(QStringLiteral("btnReadToText"));
        btnReadToText->setMinimumSize(QSize(0, 25));

        gridLayout->addWidget(btnReadToText, 7, 0, 1, 2);

        chkBoxRowColor = new QCheckBox(groupBox);
        chkBoxRowColor->setObjectName(QStringLiteral("chkBoxRowColor"));
        chkBoxRowColor->setChecked(true);

        gridLayout->addWidget(chkBoxRowColor, 8, 1, 1, 1);

        radioSelectItem = new QRadioButton(groupBox);
        radioSelectItem->setObjectName(QStringLiteral("radioSelectItem"));
        radioSelectItem->setChecked(true);

        gridLayout->addWidget(radioSelectItem, 10, 1, 1, 1);

        chkBoxHeaderV = new QCheckBox(groupBox);
        chkBoxHeaderV->setObjectName(QStringLiteral("chkBoxHeaderV"));
        chkBoxHeaderV->setChecked(true);

        gridLayout->addWidget(chkBoxHeaderV, 9, 1, 1, 1);

        radioSelectRow = new QRadioButton(groupBox);
        radioSelectRow->setObjectName(QStringLiteral("radioSelectRow"));
        radioSelectRow->setChecked(false);

        gridLayout->addWidget(radioSelectRow, 10, 0, 1, 1);

        btnIniData = new QPushButton(groupBox);
        btnIniData->setObjectName(QStringLiteral("btnIniData"));
        btnIniData->setMinimumSize(QSize(0, 25));

        gridLayout->addWidget(btnIniData, 2, 0, 1, 2);

        spinRowCount = new QSpinBox(groupBox);
        spinRowCount->setObjectName(QStringLiteral("spinRowCount"));
        spinRowCount->setMinimum(2);
        spinRowCount->setValue(8);

        gridLayout->addWidget(spinRowCount, 1, 1, 1, 1);

        btnSetHeader = new QPushButton(groupBox);
        btnSetHeader->setObjectName(QStringLiteral("btnSetHeader"));
        btnSetHeader->setMinimumSize(QSize(0, 25));

        gridLayout->addWidget(btnSetHeader, 0, 0, 1, 2);

        btnSetRows = new QPushButton(groupBox);
        btnSetRows->setObjectName(QStringLiteral("btnSetRows"));
        btnSetRows->setMinimumSize(QSize(0, 25));

        gridLayout->addWidget(btnSetRows, 1, 0, 1, 1);

        btnAutoHeight = new QPushButton(groupBox);
        btnAutoHeight->setObjectName(QStringLiteral("btnAutoHeight"));

        gridLayout->addWidget(btnAutoHeight, 6, 0, 1, 1);

        btnAutoWidth = new QPushButton(groupBox);
        btnAutoWidth->setObjectName(QStringLiteral("btnAutoWidth"));

        gridLayout->addWidget(btnAutoWidth, 6, 1, 1, 1);

        btnClearContents = new QPushButton(groupBox);
        btnClearContents->setObjectName(QStringLiteral("btnClearContents"));

        gridLayout->addWidget(btnClearContents, 4, 1, 1, 1);

        btnDelCurRow = new QPushButton(groupBox);
        btnDelCurRow->setObjectName(QStringLiteral("btnDelCurRow"));

        gridLayout->addWidget(btnDelCurRow, 4, 0, 1, 1);

        splitterMain->addWidget(groupBox);
        splitter = new QSplitter(splitterMain);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setFrameShape(QFrame::NoFrame);
        splitter->setFrameShadow(QFrame::Plain);
        splitter->setLineWidth(2);
        splitter->setOrientation(Qt::Vertical);
        tableInfo = new QTableWidget(splitter);
        if (tableInfo->columnCount() < 4)
            tableInfo->setColumnCount(4);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        tableInfo->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        tableInfo->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        tableInfo->setHorizontalHeaderItem(2, __qtablewidgetitem2);
        if (tableInfo->rowCount() < 5)
            tableInfo->setRowCount(5);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        tableInfo->setItem(0, 0, __qtablewidgetitem3);
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/boy.ico"), QSize(), QIcon::Normal, QIcon::Off);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        __qtablewidgetitem4->setIcon(icon);
        tableInfo->setItem(0, 1, __qtablewidgetitem4);
        QTableWidgetItem *__qtablewidgetitem5 = new QTableWidgetItem();
        __qtablewidgetitem5->setCheckState(Qt::Checked);
        tableInfo->setItem(0, 2, __qtablewidgetitem5);
        QTableWidgetItem *__qtablewidgetitem6 = new QTableWidgetItem();
        tableInfo->setItem(0, 3, __qtablewidgetitem6);
        QTableWidgetItem *__qtablewidgetitem7 = new QTableWidgetItem();
        tableInfo->setItem(1, 0, __qtablewidgetitem7);
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/girl.ico"), QSize(), QIcon::Normal, QIcon::Off);
        QTableWidgetItem *__qtablewidgetitem8 = new QTableWidgetItem();
        __qtablewidgetitem8->setIcon(icon1);
        tableInfo->setItem(1, 1, __qtablewidgetitem8);
        QTableWidgetItem *__qtablewidgetitem9 = new QTableWidgetItem();
        __qtablewidgetitem9->setCheckState(Qt::Checked);
        tableInfo->setItem(1, 2, __qtablewidgetitem9);
        QTableWidgetItem *__qtablewidgetitem10 = new QTableWidgetItem();
        tableInfo->setItem(2, 0, __qtablewidgetitem10);
        QTableWidgetItem *__qtablewidgetitem11 = new QTableWidgetItem();
        tableInfo->setItem(2, 1, __qtablewidgetitem11);
        QTableWidgetItem *__qtablewidgetitem12 = new QTableWidgetItem();
        tableInfo->setItem(2, 2, __qtablewidgetitem12);
        QTableWidgetItem *__qtablewidgetitem13 = new QTableWidgetItem();
        tableInfo->setItem(3, 0, __qtablewidgetitem13);
        QTableWidgetItem *__qtablewidgetitem14 = new QTableWidgetItem();
        tableInfo->setItem(3, 1, __qtablewidgetitem14);
        QTableWidgetItem *__qtablewidgetitem15 = new QTableWidgetItem();
        tableInfo->setItem(4, 0, __qtablewidgetitem15);
        QTableWidgetItem *__qtablewidgetitem16 = new QTableWidgetItem();
        tableInfo->setItem(4, 1, __qtablewidgetitem16);
        QTableWidgetItem *__qtablewidgetitem17 = new QTableWidgetItem();
        tableInfo->setItem(4, 2, __qtablewidgetitem17);
        QTableWidgetItem *__qtablewidgetitem18 = new QTableWidgetItem();
        tableInfo->setItem(4, 3, __qtablewidgetitem18);
        tableInfo->setObjectName(QStringLiteral("tableInfo"));
        tableInfo->setAlternatingRowColors(true);
        tableInfo->setRowCount(5);
        tableInfo->setColumnCount(4);
        splitter->addWidget(tableInfo);
        tableInfo->horizontalHeader()->setDefaultSectionSize(100);
        tableInfo->verticalHeader()->setDefaultSectionSize(30);
        textEdit = new QPlainTextEdit(splitter);
        textEdit->setObjectName(QStringLiteral("textEdit"));
        splitter->addWidget(textEdit);
        splitterMain->addWidget(splitter);

        verticalLayout->addWidget(splitterMain);

        MainWindow->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Demo3_9, QTableWidget\347\232\204\344\275\277\347\224\250", nullptr));
        groupBox->setTitle(QString());
        btnInsertRow->setText(QApplication::translate("MainWindow", "\346\217\222\345\205\245\350\241\214", nullptr));
        btnAppendRow->setText(QApplication::translate("MainWindow", "\346\267\273\345\212\240\350\241\214", nullptr));
        chkBoxHeaderH->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\350\241\214\350\241\250\345\244\264", nullptr));
        chkBoxEditable->setText(QApplication::translate("MainWindow", "\350\241\250\346\240\274\345\217\257\347\274\226\350\276\221", nullptr));
        btnReadToText->setText(QApplication::translate("MainWindow", "\350\257\273\345\217\226\350\241\250\346\240\274\345\206\205\345\256\271\345\210\260\346\226\207\346\234\254", nullptr));
        chkBoxRowColor->setText(QApplication::translate("MainWindow", "\351\227\264\351\232\224\350\241\214\345\272\225\350\211\262", nullptr));
        radioSelectItem->setText(QApplication::translate("MainWindow", "\345\215\225\345\205\203\346\240\274\351\200\211\346\213\251", nullptr));
        chkBoxHeaderV->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\345\210\227\350\241\250\345\244\264", nullptr));
        radioSelectRow->setText(QApplication::translate("MainWindow", "\350\241\214\351\200\211\346\213\251", nullptr));
        btnIniData->setText(QApplication::translate("MainWindow", "\345\210\235\345\247\213\345\214\226\350\241\250\346\240\274\346\225\260\346\215\256", nullptr));
        btnSetHeader->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256\350\241\250\345\244\264", nullptr));
        btnSetRows->setText(QApplication::translate("MainWindow", "\350\256\276\347\275\256\350\241\214\346\225\260", nullptr));
        btnAutoHeight->setText(QApplication::translate("MainWindow", "\350\207\252\345\212\250\350\260\203\350\212\202\350\241\214\351\253\230", nullptr));
        btnAutoWidth->setText(QApplication::translate("MainWindow", "\350\207\252\345\212\250\350\260\203\350\212\202\345\210\227\345\256\275", nullptr));
        btnClearContents->setText(QApplication::translate("MainWindow", "\346\270\205\347\251\272\350\241\250\346\240\274\345\206\205\345\256\271", nullptr));
        btnDelCurRow->setText(QApplication::translate("MainWindow", "\345\210\240\351\231\244\345\275\223\345\211\215\350\241\214", nullptr));
        QTableWidgetItem *___qtablewidgetitem = tableInfo->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("MainWindow", "\345\210\2271", nullptr));
        QTableWidgetItem *___qtablewidgetitem1 = tableInfo->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("MainWindow", "\345\210\2272", nullptr));
        QTableWidgetItem *___qtablewidgetitem2 = tableInfo->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QApplication::translate("MainWindow", "\345\210\2273", nullptr));

        const bool __sortingEnabled = tableInfo->isSortingEnabled();
        tableInfo->setSortingEnabled(false);
        QTableWidgetItem *___qtablewidgetitem3 = tableInfo->item(0, 0);
        ___qtablewidgetitem3->setText(QApplication::translate("MainWindow", "0\350\241\214\357\274\2140\345\210\227", nullptr));
        QTableWidgetItem *___qtablewidgetitem4 = tableInfo->item(0, 1);
        ___qtablewidgetitem4->setText(QApplication::translate("MainWindow", "0\350\241\214\357\274\2141\345\210\227", nullptr));
        QTableWidgetItem *___qtablewidgetitem5 = tableInfo->item(0, 2);
        ___qtablewidgetitem5->setText(QApplication::translate("MainWindow", "1\350\241\214\357\274\2142\345\210\227", nullptr));
        QTableWidgetItem *___qtablewidgetitem6 = tableInfo->item(0, 3);
        ___qtablewidgetitem6->setText(QApplication::translate("MainWindow", "0\350\241\214\357\274\2143\345\210\227", nullptr));
        QTableWidgetItem *___qtablewidgetitem7 = tableInfo->item(1, 0);
        ___qtablewidgetitem7->setText(QApplication::translate("MainWindow", "1\350\241\214\357\274\2140\345\210\227", nullptr));
        QTableWidgetItem *___qtablewidgetitem8 = tableInfo->item(1, 1);
        ___qtablewidgetitem8->setText(QApplication::translate("MainWindow", "1\350\241\214\357\274\2141\345\210\227", nullptr));
        QTableWidgetItem *___qtablewidgetitem9 = tableInfo->item(2, 0);
        ___qtablewidgetitem9->setText(QApplication::translate("MainWindow", "2\350\241\214\357\274\2140\345\210\227", nullptr));
        QTableWidgetItem *___qtablewidgetitem10 = tableInfo->item(3, 0);
        ___qtablewidgetitem10->setText(QApplication::translate("MainWindow", "3\350\241\214\357\274\2140\345\210\227", nullptr));
        QTableWidgetItem *___qtablewidgetitem11 = tableInfo->item(4, 0);
        ___qtablewidgetitem11->setText(QApplication::translate("MainWindow", "4\350\241\214\357\274\2140\345\210\227", nullptr));
        QTableWidgetItem *___qtablewidgetitem12 = tableInfo->item(4, 3);
        ___qtablewidgetitem12->setText(QApplication::translate("MainWindow", "4\350\241\214\357\274\2143\345\210\227", nullptr));
        tableInfo->setSortingEnabled(__sortingEnabled);

    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
