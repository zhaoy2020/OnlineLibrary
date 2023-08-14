/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
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
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(649, 405);
        MainWindow->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        actEdit_Cut = new QAction(MainWindow);
        actEdit_Cut->setObjectName(QString::fromUtf8("actEdit_Cut"));
        actEdit_Cut->setEnabled(false);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/images/200.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Cut->setIcon(icon);
        actEdit_Copy = new QAction(MainWindow);
        actEdit_Copy->setObjectName(QString::fromUtf8("actEdit_Copy"));
        actEdit_Copy->setEnabled(false);
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/images/202.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Copy->setIcon(icon1);
        actEdit_Paste = new QAction(MainWindow);
        actEdit_Paste->setObjectName(QString::fromUtf8("actEdit_Paste"));
        actEdit_Paste->setEnabled(false);
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/images/204.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Paste->setIcon(icon2);
        actFont_Bold = new QAction(MainWindow);
        actFont_Bold->setObjectName(QString::fromUtf8("actFont_Bold"));
        actFont_Bold->setCheckable(true);
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/icons/images/500.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFont_Bold->setIcon(icon3);
        actFont_Italic = new QAction(MainWindow);
        actFont_Italic->setObjectName(QString::fromUtf8("actFont_Italic"));
        actFont_Italic->setCheckable(true);
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/icons/images/502.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFont_Italic->setIcon(icon4);
        actFont_UnderLine = new QAction(MainWindow);
        actFont_UnderLine->setObjectName(QString::fromUtf8("actFont_UnderLine"));
        actFont_UnderLine->setCheckable(true);
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/icons/images/504.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFont_UnderLine->setIcon(icon5);
        actClose = new QAction(MainWindow);
        actClose->setObjectName(QString::fromUtf8("actClose"));
        QIcon icon6;
        icon6.addFile(QString::fromUtf8(":/icons/images/132.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actClose->setIcon(icon6);
        actSys_ToggleText = new QAction(MainWindow);
        actSys_ToggleText->setObjectName(QString::fromUtf8("actSys_ToggleText"));
        actSys_ToggleText->setCheckable(true);
        actSys_ToggleText->setChecked(true);
        actEdit_Clear = new QAction(MainWindow);
        actEdit_Clear->setObjectName(QString::fromUtf8("actEdit_Clear"));
        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/icons/images/212.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Clear->setIcon(icon7);
        actEdit_Undo = new QAction(MainWindow);
        actEdit_Undo->setObjectName(QString::fromUtf8("actEdit_Undo"));
        actEdit_Undo->setEnabled(false);
        QIcon icon8;
        icon8.addFile(QString::fromUtf8(":/icons/images/206.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Undo->setIcon(icon8);
        actEdit_Redo = new QAction(MainWindow);
        actEdit_Redo->setObjectName(QString::fromUtf8("actEdit_Redo"));
        actEdit_Redo->setEnabled(false);
        QIcon icon9;
        icon9.addFile(QString::fromUtf8(":/icons/images/208.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actEdit_Redo->setIcon(icon9);
        actEdit_SelectAll = new QAction(MainWindow);
        actEdit_SelectAll->setObjectName(QString::fromUtf8("actEdit_SelectAll"));
        actFile_New = new QAction(MainWindow);
        actFile_New->setObjectName(QString::fromUtf8("actFile_New"));
        QIcon icon10;
        icon10.addFile(QString::fromUtf8(":/icons/images/100.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFile_New->setIcon(icon10);
        actFile_Open = new QAction(MainWindow);
        actFile_Open->setObjectName(QString::fromUtf8("actFile_Open"));
        QIcon icon11;
        icon11.addFile(QString::fromUtf8(":/icons/images/122.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFile_Open->setIcon(icon11);
        actFile_Save = new QAction(MainWindow);
        actFile_Save->setObjectName(QString::fromUtf8("actFile_Save"));
        QIcon icon12;
        icon12.addFile(QString::fromUtf8(":/icons/images/104.bmp"), QSize(), QIcon::Normal, QIcon::Off);
        actFile_Save->setIcon(icon12);
        actLang_CN = new QAction(MainWindow);
        actLang_CN->setObjectName(QString::fromUtf8("actLang_CN"));
        actLang_CN->setCheckable(true);
        actLang_CN->setChecked(true);
        QIcon icon13;
        icon13.addFile(QString::fromUtf8(":/icons/images/CN.jpg"), QSize(), QIcon::Normal, QIcon::Off);
        actLang_CN->setIcon(icon13);
        actLang_EN = new QAction(MainWindow);
        actLang_EN->setObjectName(QString::fromUtf8("actLang_EN"));
        actLang_EN->setCheckable(true);
        QIcon icon14;
        icon14.addFile(QString::fromUtf8(":/icons/images/timg2.jpg"), QSize(), QIcon::Normal, QIcon::Off);
        actLang_EN->setIcon(icon14);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        textEdit = new QPlainTextEdit(centralWidget);
        textEdit->setObjectName(QString::fromUtf8("textEdit"));
        textEdit->setGeometry(QRect(55, 25, 496, 276));
        QFont font;
        font.setPointSize(12);
        textEdit->setFont(font);
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 649, 23));
        menu_E = new QMenu(menuBar);
        menu_E->setObjectName(QString::fromUtf8("menu_E"));
        menu_F = new QMenu(menuBar);
        menu_F->setObjectName(QString::fromUtf8("menu_F"));
        menu = new QMenu(menu_F);
        menu->setObjectName(QString::fromUtf8("menu"));
        menu_F_2 = new QMenu(menuBar);
        menu_F_2->setObjectName(QString::fromUtf8("menu_F_2"));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonIconOnly);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
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
        mainToolBar->addSeparator();
        mainToolBar->addAction(actFont_Bold);
        mainToolBar->addAction(actFont_Italic);
        mainToolBar->addAction(actFont_UnderLine);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actLang_CN);
        mainToolBar->addAction(actLang_EN);
        mainToolBar->addSeparator();

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
        actSys_ToggleText->setText(QApplication::translate("MainWindow", "\346\230\276\347\244\272\346\214\211\351\222\256\346\226\207\345\255\227", nullptr));
#ifndef QT_NO_TOOLTIP
        actSys_ToggleText->setToolTip(QApplication::translate("MainWindow", "\346\230\276\347\244\272\345\267\245\345\205\267\346\240\217\346\214\211\351\222\256\346\226\207\345\255\227", nullptr));
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
        actLang_EN->setText(QApplication::translate("MainWindow", "\350\213\261\350\257\255", nullptr));
#ifndef QT_NO_TOOLTIP
        actLang_EN->setToolTip(QApplication::translate("MainWindow", "\350\213\261\350\257\255", nullptr));
#endif // QT_NO_TOOLTIP
        textEdit->setPlainText(QApplication::translate("MainWindow", "import sys\n"
"\n"
"from PyQt5.QtWidgets import  QApplication, QMainWindow\n"
"\n"
"from PyQt5.QtWidgets import   QLabel, QProgressBar, QSpinBox, QFontComboBox\n"
"\n"
"from PyQt5.QtCore import  Qt, pyqtSlot\n"
"\n"
"from PyQt5.QtGui import  QTextCharFormat, QFont\n"
"\n"
"from ui_MainWindow import Ui_MainWindow\n"
"\n"
"\n"
"class QmyMainWindow(QMainWindow): \n"
"\n"
"    def __init__(self, parent=None):\n"
"        super().__init__(parent) # \350\260\203\347\224\250\347\210\266\347\261\273\346\236\204\351\200\240\345\207\275\346\225\260\357\274\214\345\210\233\345\273\272\347\252\227\344\275\223\n"
"        self.ui=Ui_MainWindow()     #\345\210\233\345\273\272UI\345\257\271\350\261\241\n"
"        self.ui.setupUi(self)       #\346\236\204\351\200\240UI\347\225\214\351\235\242\n"
"\n"
"        self.__buildUI()\n"
"        self.__spinFontSize.valueChanged[int].connect(self.do_fontSize_Changed)\n"
"        self.__comboFontName.currentIndexChanged[str].connect(self.do_fontName_Changed)\n"
"\n"
"        self.setC"
                        "entralWidget(self.ui.textEdit)\n"
"\n"
"##  ============\350\207\252\345\256\232\344\271\211\345\212\237\350\203\275\345\207\275\346\225\260================================        \n"
"\n"
"    def __buildUI(self):  #\347\252\227\344\275\223\344\270\212\345\212\250\346\200\201\346\267\273\345\212\240\347\273\204\344\273\266\n"
"        # \345\210\233\345\273\272\347\212\266\346\200\201\346\240\217\344\270\212\347\232\204\347\273\204\344\273\266\n"
"        self.__LabFile=QLabel(self)     # QLabel\347\273\204\344\273\266\346\230\276\347\244\272\344\277\241\346\201\257\n"
"        self.__LabFile.setMinimumWidth(150)\n"
"        self.__LabFile.setText(\"\346\226\207\344\273\266\345\220\215\357\274\232 \")\n"
"        self.ui.statusBar.addWidget(self.__LabFile)\n"
"\n"
"        self.__progressBar1=QProgressBar(self)      # progressBar1\n"
"        self.__progressBar1.setMaximumWidth(200)\n"
"        self.__progressBar1.setMinimum(5)\n"
"        self.__progressBar1.setMaximum(50)\n"
"        self.__progressBar1.set"
                        "Value(self.ui.textEdit.font().pointSize())\n"
"        self.ui.statusBar.addWidget(self.__progressBar1)\n"
"        \n"
"        self.__LabInfo=QLabel(self)     # QLabel\347\273\204\344\273\266\346\230\276\347\244\272\345\255\227\344\275\223\345\220\215\347\247\260\n"
"        self.__LabInfo.setText(\"\351\200\211\346\213\251\345\255\227\344\275\223\345\220\215\347\247\260\357\274\232 \")\n"
"        self.ui.statusBar.addPermanentWidget(self.__LabInfo)\n"
"        \n"
"        #\345\210\233\345\273\272\345\267\245\345\205\267\346\240\217\344\270\212\347\232\204\347\273\204\344\273\266\n"
"        self.__spinFontSize=QSpinBox(self)  #\345\255\227\344\275\223\345\244\247\345\260\217spinbox\n"
"        self.__spinFontSize.setMinimum(5)\n"
"        self.__spinFontSize.setMaximum(50)\n"
"        self.__spinFontSize.setValue(self.ui.textEdit.font().pointSize())\n"
"        self.__spinFontSize.setMinimumWidth(50)\n"
"        self.ui.mainToolBar.addWidget(self.__spinFontSize) #SpinBox\346\267\273\345\212\240\345\210\260"
                        "\345\267\245\345\205\267\346\240\217\n"
"        \n"
"        self.__comboFontName=QFontComboBox(self)  #\345\255\227\344\275\223 combobox\n"
"        self.__comboFontName.setMinimumWidth(150)\n"
"        self.ui.mainToolBar.addWidget(self.__comboFontName) \n"
"\n"
"        self.ui.mainToolBar.addSeparator()  #\346\267\273\345\212\240\344\270\200\344\270\252\345\210\206\351\232\224\346\235\241\n"
"        \n"
"        self.ui.mainToolBar.addAction(self.ui.actClose)  #\346\267\273\345\212\240 actClose\344\275\234\344\270\272\342\200\234\345\205\263\351\227\255\342\200\235\346\214\211\351\222\256\n"
"        \n"
"        \n"
"##  ===========\347\224\261connectSlotsByName() \350\207\252\345\212\250\350\277\236\346\216\245\347\232\204\346\247\275\345\207\275\346\225\260=====================      \n"
"\n"
"    @pyqtSlot(bool)\n"
"    def on_actFont_Bold_triggered(self, checked):     #\350\256\276\347\275\256\347\262\227\344\275\223 \n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        if (checked == Tru"
                        "e):\n"
"            fmt.setFontWeight(QFont.Bold)\n"
"        else:\n"
"            fmt.setFontWeight(QFont.Normal)\n"
"        self.ui.textEdit.mergeCurrentCharFormat(fmt)\n"
"\n"
"    @pyqtSlot(bool)\n"
"    def on_actFont_Italic_triggered(self,checked):      #\350\256\276\347\275\256\346\226\234\344\275\223 \n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        fmt.setFontItalic(checked)\n"
"        self.ui.textEdit.mergeCurrentCharFormat(fmt)\n"
"        \n"
"    @pyqtSlot(bool)\n"
"    def on_actFont_UnderLine_triggered(self,checked):      #\350\256\276\347\275\256\344\270\213\345\210\222\347\272\277 \n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        fmt.setFontUnderline(checked)\n"
"        self.ui.textEdit.mergeCurrentCharFormat(fmt)\n"
"\n"
"    @pyqtSlot(bool)\n"
"    def on_actSys_ToggleText_triggered(self,checked):   #\350\256\276\347\275\256\345\267\245\345\205\267\346\240\217\346\214\211\351\222\256\346\240\267\345\274\217 \n"
"        if(checked):\n"
"            self.ui"
                        ".mainToolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)\n"
"        else:\n"
"            self.ui.mainToolBar.setToolButtonStyle(Qt.ToolButtonIconOnly)\n"
"\n"
"    def on_textEdit_copyAvailable(self, avi):       #\346\226\207\346\234\254\346\241\206\345\206\205\345\256\271\345\217\257copy\n"
"        self.ui.actEdit_Cut.setEnabled(avi)\n"
"        self.ui.actEdit_Copy.setEnabled(avi)\n"
"        self.ui.actEdit_Paste.setEnabled(self.ui.textEdit.canPaste())\n"
"                                         \n"
"    def on_textEdit_selectionChanged(self):     #\346\226\207\346\234\254\351\200\211\346\213\251\345\206\205\345\256\271\345\217\221\347\224\237\345\217\230\345\214\226\n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        self.ui.actFont_Bold.setChecked(fmt.font().bold())\n"
"        self.ui.actFont_Italic.setChecked(fmt.fontItalic())\n"
"        self.ui.actFont_UnderLine.setChecked(fmt.fontUnderline())\n"
"\n"
"    def on_actFile_New_triggered(self):     #\346\226\260\345\273\272\346\226\207"
                        "\344\273\266\357\274\214\344\270\215\345\256\236\347\216\260\345\205\267\344\275\223\345\212\237\350\203\275\n"
"        self.__LabFile.setText(\" \346\226\260\345\273\272\346\226\207\344\273\266 \")\n"
"\n"
"    def on_actFile_Open_triggered(self):    #\346\211\223\345\274\200\346\226\207\344\273\266\357\274\214\344\270\215\345\256\236\347\216\260\345\205\267\344\275\223\345\212\237\350\203\275\n"
"        self.__LabFile.setText(\" \346\211\223\345\274\200\347\232\204\346\226\207\344\273\266 \")\n"
"        \n"
"    def on_actFile_Save_triggered(self):    #\344\277\235\345\255\230\346\226\207\344\273\266\357\274\214\344\270\215\345\256\236\347\216\260\345\205\267\344\275\223\345\212\237\350\203\275\n"
"        self.__LabFile.setText(\" \346\226\207\344\273\266\345\267\262\344\277\235\345\255\230 \")\n"
"        \n"
"        \n"
"##  =============\350\207\252\345\256\232\344\271\211\346\247\275\345\207\275\346\225\260===============================        \n"
"\n"
"    @pyqtSlot(int)\n"
"    def do_fontSize_Ch"
                        "anged(self, fontSize):      #\350\256\276\347\275\256\345\255\227\344\275\223\345\244\247\345\260\217 __spinFontSize\n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        fmt.setFontPointSize(fontSize)\n"
"        self.ui.textEdit.mergeCurrentCharFormat(fmt)\n"
"        self.__progressBar1.setValue(fontSize)    #\346\263\250\346\204\217\345\256\203\345\261\236\344\272\216self, \350\200\214\344\270\215\346\230\257self.ui\n"
"\n"
"\n"
"    @pyqtSlot(str)\n"
"    def do_fontName_Changed(self, fontName):    # \351\200\211\346\213\251\345\255\227\344\275\223 __comboFontName\n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        fmt.setFontFamily(fontName)\n"
"        self.ui.textEdit.mergeCurrentCharFormat(fmt)\n"
"        self.__LabInfo.setText(\"\345\255\227\344\275\223\345\220\215\347\247\260\357\274\232%s   \"%fontName)\n"
"        \n"
"   \n"
"##  ===========\347\252\227\344\275\223\346\265\213\350\257\225\347\250\213\345\272\217=================================        \n"
"if  __name__ =="
                        " \"__main__\":         #\347\224\250\344\272\216\345\275\223\345\211\215\347\252\227\344\275\223\346\265\213\350\257\225\n"
"    app = QApplication(sys.argv)    #\345\210\233\345\273\272GUI\345\272\224\347\224\250\347\250\213\345\272\217\n"
"    form=QmyMainWindow()            #\345\210\233\345\273\272\347\252\227\344\275\223\n"
"    form.show()\n"
"    sys.exit(app.exec_())\n"
"", nullptr));
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
