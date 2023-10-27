/********************************************************************************
** Form generated from reading UI file 'QWFormDoc.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QWFORMDOC_H
#define UI_QWFORMDOC_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QWFormDoc
{
public:
    QAction *actOpen;
    QAction *actCut;
    QAction *actCopy;
    QAction *actPaste;
    QAction *actFont;
    QAction *actClose;
    QAction *actUndo;
    QAction *actRedo;
    QPlainTextEdit *plainTextEdit;

    void setupUi(QWidget *QWFormDoc)
    {
        if (QWFormDoc->objectName().isEmpty())
            QWFormDoc->setObjectName(QStringLiteral("QWFormDoc"));
        QWFormDoc->resize(628, 389);
        QFont font;
        font.setPointSize(10);
        QWFormDoc->setFont(font);
        QWFormDoc->setAutoFillBackground(true);
        actOpen = new QAction(QWFormDoc);
        actOpen->setObjectName(QStringLiteral("actOpen"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/122.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actOpen->setIcon(icon);
        actCut = new QAction(QWFormDoc);
        actCut->setObjectName(QStringLiteral("actCut"));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/200.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actCut->setIcon(icon1);
        actCopy = new QAction(QWFormDoc);
        actCopy->setObjectName(QStringLiteral("actCopy"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/202.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actCopy->setIcon(icon2);
        actPaste = new QAction(QWFormDoc);
        actPaste->setObjectName(QStringLiteral("actPaste"));
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/204.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actPaste->setIcon(icon3);
        actFont = new QAction(QWFormDoc);
        actFont->setObjectName(QStringLiteral("actFont"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/506.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFont->setIcon(icon4);
        actClose = new QAction(QWFormDoc);
        actClose->setObjectName(QStringLiteral("actClose"));
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actClose->setIcon(icon5);
        actUndo = new QAction(QWFormDoc);
        actUndo->setObjectName(QStringLiteral("actUndo"));
        QIcon icon6;
        icon6.addFile(QStringLiteral(":/icons/images/206.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actUndo->setIcon(icon6);
        actRedo = new QAction(QWFormDoc);
        actRedo->setObjectName(QStringLiteral("actRedo"));
        QIcon icon7;
        icon7.addFile(QStringLiteral(":/icons/images/208.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actRedo->setIcon(icon7);
        plainTextEdit = new QPlainTextEdit(QWFormDoc);
        plainTextEdit->setObjectName(QStringLiteral("plainTextEdit"));
        plainTextEdit->setGeometry(QRect(75, 45, 256, 192));

        retranslateUi(QWFormDoc);
        QObject::connect(actClose, SIGNAL(triggered()), QWFormDoc, SLOT(close()));
        QObject::connect(actCut, SIGNAL(triggered()), plainTextEdit, SLOT(cut()));
        QObject::connect(actCopy, SIGNAL(triggered()), plainTextEdit, SLOT(copy()));
        QObject::connect(actPaste, SIGNAL(triggered()), plainTextEdit, SLOT(paste()));
        QObject::connect(actUndo, SIGNAL(triggered()), plainTextEdit, SLOT(undo()));
        QObject::connect(actRedo, SIGNAL(triggered()), plainTextEdit, SLOT(redo()));

        QMetaObject::connectSlotsByName(QWFormDoc);
    } // setupUi

    void retranslateUi(QWidget *QWFormDoc)
    {
        QWFormDoc->setWindowTitle(QApplication::translate("QWFormDoc", "new document", nullptr));
        actOpen->setText(QApplication::translate("QWFormDoc", "\346\211\223\345\274\200", nullptr));
#ifndef QT_NO_TOOLTIP
        actOpen->setToolTip(QApplication::translate("QWFormDoc", "\346\211\223\345\274\200\346\226\207\344\273\266", nullptr));
#endif // QT_NO_TOOLTIP
        actCut->setText(QApplication::translate("QWFormDoc", "\345\211\252\345\210\207", nullptr));
#ifndef QT_NO_TOOLTIP
        actCut->setToolTip(QApplication::translate("QWFormDoc", "\345\211\252\345\210\207", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actCut->setShortcut(QApplication::translate("QWFormDoc", "Ctrl+X", nullptr));
#endif // QT_NO_SHORTCUT
        actCopy->setText(QApplication::translate("QWFormDoc", "\345\244\215\345\210\266", nullptr));
#ifndef QT_NO_TOOLTIP
        actCopy->setToolTip(QApplication::translate("QWFormDoc", "\345\244\215\345\210\266", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actCopy->setShortcut(QApplication::translate("QWFormDoc", "Ctrl+C", nullptr));
#endif // QT_NO_SHORTCUT
        actPaste->setText(QApplication::translate("QWFormDoc", "\347\262\230\350\264\264", nullptr));
#ifndef QT_NO_TOOLTIP
        actPaste->setToolTip(QApplication::translate("QWFormDoc", "\347\262\230\350\264\264", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actPaste->setShortcut(QApplication::translate("QWFormDoc", "Ctrl+V", nullptr));
#endif // QT_NO_SHORTCUT
        actFont->setText(QApplication::translate("QWFormDoc", "\345\255\227\344\275\223", nullptr));
#ifndef QT_NO_TOOLTIP
        actFont->setToolTip(QApplication::translate("QWFormDoc", "\350\256\276\347\275\256\345\255\227\344\275\223", nullptr));
#endif // QT_NO_TOOLTIP
        actClose->setText(QApplication::translate("QWFormDoc", "\345\205\263\351\227\255", nullptr));
#ifndef QT_NO_TOOLTIP
        actClose->setToolTip(QApplication::translate("QWFormDoc", "\345\205\263\351\227\255\346\234\254\347\252\227\345\217\243", nullptr));
#endif // QT_NO_TOOLTIP
        actUndo->setText(QApplication::translate("QWFormDoc", "\346\222\244\351\224\200", nullptr));
#ifndef QT_NO_TOOLTIP
        actUndo->setToolTip(QApplication::translate("QWFormDoc", "\346\222\244\351\224\200\347\274\226\350\276\221\346\223\215\344\275\234", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actUndo->setShortcut(QApplication::translate("QWFormDoc", "Ctrl+Z", nullptr));
#endif // QT_NO_SHORTCUT
        actRedo->setText(QApplication::translate("QWFormDoc", "\351\207\215\345\244\215", nullptr));
#ifndef QT_NO_TOOLTIP
        actRedo->setToolTip(QApplication::translate("QWFormDoc", "\351\207\215\345\244\215\347\274\226\350\276\221\346\223\215\344\275\234", nullptr));
#endif // QT_NO_TOOLTIP
    } // retranslateUi

};

namespace Ui {
    class QWFormDoc: public Ui_QWFormDoc {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QWFORMDOC_H
