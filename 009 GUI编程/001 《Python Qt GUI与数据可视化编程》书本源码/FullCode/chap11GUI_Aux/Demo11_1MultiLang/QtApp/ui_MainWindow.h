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
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actEdit_Cut;
    QAction *actEdit_Copy;
    QAction *actEdit_Paste;
    QAction *actFont_Bold;
    QAction *actFont_Italic;
    QAction *actFont_UnderLine;
    QAction *actClose;
    QAction *actSys_ToggleText;
    QAction *actEdit_Clear;
    QAction *actEdit_Undo;
    QAction *actEdit_Redo;
    QAction *actEdit_SelectAll;
    QAction *actFile_New;
    QAction *actFile_Open;
    QAction *actFile_Save;
    QAction *actLang_CN;
    QAction *actLang_EN;
    QWidget *centralWidget;
    QPlainTextEdit *textEdit;
    QMenuBar *menuBar;
    QMenu *menu_E;
    QMenu *menu_F;
    QMenu *menu;
    QMenu *menu_F_2;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(869, 466);
        MainWindow->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        actEdit_Cut = new QAction(MainWindow);
        actEdit_Cut->setObjectName(QStringLiteral("actEdit_Cut"));
        actEdit_Cut->setEnabled(false);
        QIcon icon;
        icon.addFile(QStringLiteral(":/icons/images/200.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Cut->setIcon(icon);
        actEdit_Copy = new QAction(MainWindow);
        actEdit_Copy->setObjectName(QStringLiteral("actEdit_Copy"));
        actEdit_Copy->setEnabled(false);
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/icons/images/202.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Copy->setIcon(icon1);
        actEdit_Paste = new QAction(MainWindow);
        actEdit_Paste->setObjectName(QStringLiteral("actEdit_Paste"));
        actEdit_Paste->setEnabled(false);
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/icons/images/204.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Paste->setIcon(icon2);
        actFont_Bold = new QAction(MainWindow);
        actFont_Bold->setObjectName(QStringLiteral("actFont_Bold"));
        actFont_Bold->setCheckable(true);
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/icons/images/500.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFont_Bold->setIcon(icon3);
        actFont_Italic = new QAction(MainWindow);
        actFont_Italic->setObjectName(QStringLiteral("actFont_Italic"));
        actFont_Italic->setCheckable(true);
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/icons/images/502.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFont_Italic->setIcon(icon4);
        actFont_UnderLine = new QAction(MainWindow);
        actFont_UnderLine->setObjectName(QStringLiteral("actFont_UnderLine"));
        actFont_UnderLine->setCheckable(true);
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/icons/images/504.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFont_UnderLine->setIcon(icon5);
        actClose = new QAction(MainWindow);
        actClose->setObjectName(QStringLiteral("actClose"));
        QIcon icon6;
        icon6.addFile(QStringLiteral(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actClose->setIcon(icon6);
        actSys_ToggleText = new QAction(MainWindow);
        actSys_ToggleText->setObjectName(QStringLiteral("actSys_ToggleText"));
        actSys_ToggleText->setCheckable(true);
        actSys_ToggleText->setChecked(true);
        actEdit_Clear = new QAction(MainWindow);
        actEdit_Clear->setObjectName(QStringLiteral("actEdit_Clear"));
        QIcon icon7;
        icon7.addFile(QStringLiteral(":/icons/images/212.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Clear->setIcon(icon7);
        actEdit_Undo = new QAction(MainWindow);
        actEdit_Undo->setObjectName(QStringLiteral("actEdit_Undo"));
        actEdit_Undo->setEnabled(false);
        QIcon icon8;
        icon8.addFile(QStringLiteral(":/icons/images/206.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Undo->setIcon(icon8);
        actEdit_Redo = new QAction(MainWindow);
        actEdit_Redo->setObjectName(QStringLiteral("actEdit_Redo"));
        actEdit_Redo->setEnabled(false);
        QIcon icon9;
        icon9.addFile(QStringLiteral(":/icons/images/208.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Redo->setIcon(icon9);
        actEdit_SelectAll = new QAction(MainWindow);
        actEdit_SelectAll->setObjectName(QStringLiteral("actEdit_SelectAll"));
        actFile_New = new QAction(MainWindow);
        actFile_New->setObjectName(QStringLiteral("actFile_New"));
        QIcon icon10;
        icon10.addFile(QStringLiteral(":/icons/images/100.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFile_New->setIcon(icon10);
        actFile_Open = new QAction(MainWindow);
        actFile_Open->setObjectName(QStringLiteral("actFile_Open"));
        QIcon icon11;
        icon11.addFile(QStringLiteral(":/icons/images/122.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFile_Open->setIcon(icon11);
        actFile_Save = new QAction(MainWindow);
        actFile_Save->setObjectName(QStringLiteral("actFile_Save"));
        QIcon icon12;
        icon12.addFile(QStringLiteral(":/icons/images/104.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFile_Save->setIcon(icon12);
        actLang_CN = new QAction(MainWindow);
        actLang_CN->setObjectName(QStringLiteral("actLang_CN"));
        actLang_CN->setCheckable(true);
        actLang_CN->setChecked(true);
        QIcon icon13;
        icon13.addFile(QStringLiteral(":/icons/images/CN.jpg"), QSize(), QIcon::Normal, QIcon::Off);
        actLang_CN->setIcon(icon13);
        actLang_EN = new QAction(MainWindow);
        actLang_EN->setObjectName(QStringLiteral("actLang_EN"));
        actLang_EN->setCheckable(true);
        actLang_EN->setChecked(false);
        QIcon icon14;
        icon14.addFile(QStringLiteral(":/icons/images/timg2.jpg"), QSize(), QIcon::Normal, QIcon::Off);
        actLang_EN->setIcon(icon14);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        textEdit = new QPlainTextEdit(centralWidget);
        textEdit->setObjectName(QStringLiteral("textEdit"));
        textEdit->setGeometry(QRect(45, 25, 476, 221));
        QFont font;
        font.setPointSize(12);
        textEdit->setFont(font);
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 869, 23));
        menu_E = new QMenu(menuBar);
        menu_E->setObjectName(QStringLiteral("menu_E"));
        menu_F = new QMenu(menuBar);
        menu_F->setObjectName(QStringLiteral("menu_F"));
        menu = new QMenu(menu_F);
        menu->setObjectName(QStringLiteral("menu"));
        menu_F_2 = new QMenu(menuBar);
        menu_F_2->setObjectName(QStringLiteral("menu_F_2"));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menu_F_2->menuAction());
        menuBar->addAction(menu_E->menuAction());
        menuBar->addAction(menu_F->menuAction());
        menu_E->addAction(actEdit_Cut);
        menu_E->addAction(actEdit_Copy);
        menu_E->addAction(actEdit_Paste);
        menu_E->addSeparator();
        menu_E->addAction(actEdit_Undo);
        menu_E->addAction(actEdit_Redo);
        menu_E->addSeparator();
        menu_E->addAction(actEdit_SelectAll);
        menu_E->addAction(actEdit_Clear);
        menu_F->addAction(actFont_Bold);
        menu_F->addAction(actFont_Italic);
        menu_F->addAction(actFont_UnderLine);
        menu_F->addSeparator();
        menu_F->addAction(actSys_ToggleText);
        menu_F->addAction(menu->menuAction());
        menu->addAction(actLang_CN);
        menu->addAction(actLang_EN);
        menu_F_2->addAction(actFile_New);
        menu_F_2->addAction(actFile_Open);
        menu_F_2->addAction(actFile_Save);
        menu_F_2->addSeparator();
        menu_F_2->addAction(actClose);
        mainToolBar->addAction(actFile_New);
        mainToolBar->addAction(actFile_Open);
        mainToolBar->addAction(actFile_Save);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actEdit_Cut);
        mainToolBar->addAction(actEdit_Copy);
        mainToolBar->addAction(actEdit_Paste);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actEdit_Undo);
        mainToolBar->addAction(actEdit_Redo);
        mainToolBar->addAction(actEdit_Clear);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actFont_Bold);
        mainToolBar->addAction(actFont_Italic);
        mainToolBar->addAction(actFont_UnderLine);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actLang_CN);
        mainToolBar->addAction(actLang_EN);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actClose);

        retranslateUi(MainWindow);
        QObject::connect(actEdit_Cut, SIGNAL(triggered()), textEdit, SLOT(cut()));
        QObject::connect(actEdit_Copy, SIGNAL(triggered()), textEdit, SLOT(copy()));
        QObject::connect(actEdit_Paste, SIGNAL(triggered()), textEdit, SLOT(paste()));
        QObject::connect(actEdit_Clear, SIGNAL(triggered()), textEdit, SLOT(clear()));
        QObject::connect(actEdit_Undo, SIGNAL(triggered()), textEdit, SLOT(undo()));
        QObject::connect(actEdit_Redo, SIGNAL(triggered()), textEdit, SLOT(redo()));
        QObject::connect(textEdit, SIGNAL(undoAvailable(bool)), actEdit_Undo, SLOT(setEnabled(bool)));
        QObject::connect(textEdit, SIGNAL(redoAvailable(bool)), actEdit_Redo, SLOT(setEnabled(bool)));
        QObject::connect(actEdit_SelectAll, SIGNAL(triggered()), textEdit, SLOT(selectAll()));
        QObject::connect(actClose, SIGNAL(triggered()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Demo3_7 QAction\357\274\214QMainWindow\357\274\214QPlainTextEdit", nullptr));
        actEdit_Cut->setText(QApplication::translate("MainWindow", "\345\211\252\345\210\207", nullptr));
#ifndef QT_NO_TOOLTIP
        actEdit_Cut->setToolTip(QApplication::translate("MainWindow", "\345\211\252\345\210\207\345\210\260\347\262\230\350\264\264\346\235\277", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actEdit_Cut->setShortcut(QApplication::translate("MainWindow", "Ctrl+X", nullptr));
#endif // QT_NO_SHORTCUT
        actEdit_Copy->setText(QApplication::translate("MainWindow", "\345\244\215\345\210\266", nullptr));
#ifndef QT_NO_TOOLTIP
        actEdit_Copy->setToolTip(QApplication::translate("MainWindow", "\345\244\215\345\210\266\345\210\260\347\262\230\350\264\264\346\235\277", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actEdit_Copy->setShortcut(QApplication::translate("MainWindow", "Ctrl+C", nullptr));
#endif // QT_NO_SHORTCUT
        actEdit_Paste->setText(QApplication::translate("MainWindow", "\347\262\230\350\264\264", nullptr));
#ifndef QT_NO_TOOLTIP
        actEdit_Paste->setToolTip(QApplication::translate("MainWindow", "\344\273\216\347\262\230\350\264\264\346\235\277\347\262\230\350\264\264", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actEdit_Paste->setShortcut(QApplication::translate("MainWindow", "Ctrl+V", nullptr));
#endif // QT_NO_SHORTCUT
        actFont_Bold->setText(QApplication::translate("MainWindow", "\347\262\227\344\275\223", nullptr));
#ifndef QT_NO_TOOLTIP
        actFont_Bold->setToolTip(QApplication::translate("MainWindow", "\347\262\227\344\275\223", nullptr));
#endif // QT_NO_TOOLTIP
        actFont_Italic->setText(QApplication::translate("MainWindow", "\346\226\234\344\275\223", nullptr));
#ifndef QT_NO_TOOLTIP
        actFont_Italic->setToolTip(QApplication::translate("MainWindow", "\346\226\234\344\275\223", nullptr));
#endif // QT_NO_TOOLTIP
        actFont_UnderLine->setText(QApplication::translate("MainWindow", "\344\270\213\345\210\222\347\272\277", nullptr));
#ifndef QT_NO_TOOLTIP
        actFont_UnderLine->setToolTip(QApplication::translate("MainWindow", "\344\270\213\345\210\222\347\272\277", nullptr));
#endif // QT_NO_TOOLTIP
        actClose->setText(QApplication::translate("MainWindow", "\345\205\263\351\227\255", nullptr));
#ifndef QT_NO_TOOLTIP
        actClose->setToolTip(QApplication::translate("MainWindow", "\345\205\263\351\227\255\346\234\254\347\252\227\345\217\243", nullptr));
#endif // QT_NO_TOOLTIP
        actSys_ToggleText->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\345\267\245\345\205\267\346\240\217\346\226\207\345\255\227", nullptr));
#ifndef QT_NO_TOOLTIP
        actSys_ToggleText->setToolTip(QApplication::translate("MainWindow", "\346\230\276\347\244\272\345\267\245\345\205\267\346\240\217\346\226\207\345\255\227", nullptr));
#endif // QT_NO_TOOLTIP
        actEdit_Clear->setText(QApplication::translate("MainWindow", "\346\270\205\347\251\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actEdit_Clear->setToolTip(QApplication::translate("MainWindow", "\346\270\205\347\251\272", nullptr));
#endif // QT_NO_TOOLTIP
        actEdit_Undo->setText(QApplication::translate("MainWindow", "\346\222\244\351\224\200", nullptr));
#ifndef QT_NO_TOOLTIP
        actEdit_Undo->setToolTip(QApplication::translate("MainWindow", "\346\222\244\351\224\200\344\270\212\346\254\241\347\274\226\350\276\221\346\223\215\344\275\234", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actEdit_Undo->setShortcut(QApplication::translate("MainWindow", "Ctrl+Z", nullptr));
#endif // QT_NO_SHORTCUT
        actEdit_Redo->setText(QApplication::translate("MainWindow", "\351\207\215\345\201\232", nullptr));
#ifndef QT_NO_TOOLTIP
        actEdit_Redo->setToolTip(QApplication::translate("MainWindow", "\351\207\215\345\201\232\344\270\212\346\254\241\346\223\215\344\275\234", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actEdit_Redo->setShortcut(QApplication::translate("MainWindow", "Ctrl+Y", nullptr));
#endif // QT_NO_SHORTCUT
        actEdit_SelectAll->setText(QApplication::translate("MainWindow", "\345\205\250\351\200\211", nullptr));
#ifndef QT_NO_TOOLTIP
        actEdit_SelectAll->setToolTip(QApplication::translate("MainWindow", "\345\205\250\351\200\211", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actEdit_SelectAll->setShortcut(QApplication::translate("MainWindow", "Ctrl+A", nullptr));
#endif // QT_NO_SHORTCUT
        actFile_New->setText(QApplication::translate("MainWindow", "\346\226\260\345\273\272", nullptr));
#ifndef QT_NO_TOOLTIP
        actFile_New->setToolTip(QApplication::translate("MainWindow", "\346\226\260\345\273\272\346\226\207\344\273\266", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actFile_New->setShortcut(QApplication::translate("MainWindow", "Ctrl+N", nullptr));
#endif // QT_NO_SHORTCUT
        actFile_Open->setText(QApplication::translate("MainWindow", "\346\211\223\345\274\200...", nullptr));
#ifndef QT_NO_TOOLTIP
        actFile_Open->setToolTip(QApplication::translate("MainWindow", "\346\211\223\345\274\200\346\226\207\344\273\266", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actFile_Open->setShortcut(QApplication::translate("MainWindow", "Ctrl+O", nullptr));
#endif // QT_NO_SHORTCUT
        actFile_Save->setText(QApplication::translate("MainWindow", "\344\277\235\345\255\230", nullptr));
#ifndef QT_NO_TOOLTIP
        actFile_Save->setToolTip(QApplication::translate("MainWindow", "\344\277\235\345\255\230\344\277\256\346\224\271", nullptr));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_SHORTCUT
        actFile_Save->setShortcut(QApplication::translate("MainWindow", "Ctrl+S", nullptr));
#endif // QT_NO_SHORTCUT
        actLang_CN->setText(QApplication::translate("MainWindow", "\346\261\211\350\257\255", nullptr));
#ifndef QT_NO_TOOLTIP
        actLang_CN->setToolTip(QApplication::translate("MainWindow", "\346\261\211\350\257\255\347\225\214\351\235\242", nullptr));
#endif // QT_NO_TOOLTIP
        actLang_EN->setText(QApplication::translate("MainWindow", "English", nullptr));
#ifndef QT_NO_TOOLTIP
        actLang_EN->setToolTip(QApplication::translate("MainWindow", "English interface", nullptr));
#endif // QT_NO_TOOLTIP
        menu_E->setTitle(QApplication::translate("MainWindow", "\347\274\226\350\276\221(&E)", nullptr));
        menu_F->setTitle(QApplication::translate("MainWindow", "\346\240\274\345\274\217(&M)", nullptr));
        menu->setTitle(QApplication::translate("MainWindow", "\347\225\214\351\235\242\350\257\255\350\250\200", nullptr));
        menu_F_2->setTitle(QApplication::translate("MainWindow", "\346\226\207\344\273\266(&F)", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
