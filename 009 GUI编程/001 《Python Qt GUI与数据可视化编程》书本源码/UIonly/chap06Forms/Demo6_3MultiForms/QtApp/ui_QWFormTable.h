/********************************************************************************
** Form generated from reading UI file 'QWFormTable.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QWFORMTABLE_H
#define UI_QWFORMTABLE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QTableView>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QWFormTable
{
public:
    QAction *actSetSize;
    QAction *actSetHeader;
    QAction *actClose;
    QWidget *centralwidget;
    QTableView *tableView;
    QToolBar *toolBar;

    void setupUi(QMainWindow *QWFormTable)
    {
        if (QWFormTable->objectName().isEmpty())
            QWFormTable->setObjectName(QStringLiteral("QWFormTable"));
        QWFormTable->resize(493, 321);
        actSetSize = new QAction(QWFormTable);
        actSetSize->setObjectName(QStringLiteral("actSetSize"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/230.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actSetSize->setIcon(icon);
        actSetHeader = new QAction(QWFormTable);
        actSetHeader->setObjectName(QStringLiteral("actSetHeader"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/516.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actSetHeader->setIcon(icon1);
        actClose = new QAction(QWFormTable);
        actClose->setObjectName(QStringLiteral("actClose"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actClose->setIcon(icon2);
        centralwidget = new QWidget(QWFormTable);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        tableView = new QTableView(centralwidget);
        tableView->setObjectName(QStringLiteral("tableView"));
        tableView->setGeometry(QRect(65, 10, 256, 192));
        QWFormTable->setCentralWidget(centralwidget);
        toolBar = new QToolBar(QWFormTable);
        toolBar->setObjectName(QStringLiteral("toolBar"));
        toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        QWFormTable->addToolBar(Qt::TopToolBarArea, toolBar);

        toolBar->addAction(actSetSize);
        toolBar->addAction(actSetHeader);
        toolBar->addSeparator();
        toolBar->addAction(actClose);

        retranslateUi(QWFormTable);
        QObject::connect(actClose, SIGNAL(triggered()), QWFormTable, SLOT(close()));

        QMetaObject::connectSlotsByName(QWFormTable);
    } // setupUi

    void retranslateUi(QMainWindow *QWFormTable)
    {
        QWFormTable->setWindowTitle(QApplication::translate("QWFormTable", "Table\346\225\260\346\215\256", nullptr));
        actSetSize->setText(QApplication::translate("QWFormTable", "\350\256\276\347\275\256\350\241\250\346\240\274\345\244\247\345\260\217", nullptr));
#ifndef QT_NO_TOOLTIP
        actSetSize->setToolTip(QApplication::translate("QWFormTable", "\350\256\276\347\275\256\350\241\250\346\240\274\345\244\247\345\260\217", nullptr));
#endif // QT_NO_TOOLTIP
        actSetHeader->setText(QApplication::translate("QWFormTable", "\350\256\276\347\275\256\350\241\250\345\244\264", nullptr));
#ifndef QT_NO_TOOLTIP
        actSetHeader->setToolTip(QApplication::translate("QWFormTable", "\350\256\276\347\275\256\350\241\250\345\244\264\346\226\207\345\255\227", nullptr));
#endif // QT_NO_TOOLTIP
        actClose->setText(QApplication::translate("QWFormTable", "\345\205\263\351\227\255", nullptr));
#ifndef QT_NO_TOOLTIP
        actClose->setToolTip(QApplication::translate("QWFormTable", "\345\205\263\351\227\255\346\234\254\347\252\227\345\217\243", nullptr));
#endif // QT_NO_TOOLTIP
        toolBar->setWindowTitle(QApplication::translate("QWFormTable", "toolBar", nullptr));
    } // retranslateUi

};

namespace Ui {
    class QWFormTable: public Ui_QWFormTable {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QWFORMTABLE_H
