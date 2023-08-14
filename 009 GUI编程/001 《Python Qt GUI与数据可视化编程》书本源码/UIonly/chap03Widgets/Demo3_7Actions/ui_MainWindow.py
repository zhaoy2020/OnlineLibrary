# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(743, 379)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.textEdit = QtWidgets.QPlainTextEdit(self.centralWidget)
        self.textEdit.setGeometry(QtCore.QRect(45, 25, 476, 221))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textEdit.setFont(font)
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 743, 23))
        self.menuBar.setObjectName("menuBar")
        self.menu_E = QtWidgets.QMenu(self.menuBar)
        self.menu_E.setObjectName("menu_E")
        self.menu_F = QtWidgets.QMenu(self.menuBar)
        self.menu_F.setObjectName("menu_F")
        self.menu = QtWidgets.QMenu(self.menu_F)
        self.menu.setObjectName("menu")
        self.menu_F_2 = QtWidgets.QMenu(self.menuBar)
        self.menu_F_2.setObjectName("menu_F_2")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actEdit_Cut = QtWidgets.QAction(MainWindow)
        self.actEdit_Cut.setEnabled(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/200.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actEdit_Cut.setIcon(icon)
        self.actEdit_Cut.setObjectName("actEdit_Cut")
        self.actEdit_Copy = QtWidgets.QAction(MainWindow)
        self.actEdit_Copy.setEnabled(False)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/202.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actEdit_Copy.setIcon(icon1)
        self.actEdit_Copy.setObjectName("actEdit_Copy")
        self.actEdit_Paste = QtWidgets.QAction(MainWindow)
        self.actEdit_Paste.setEnabled(False)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/images/204.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actEdit_Paste.setIcon(icon2)
        self.actEdit_Paste.setObjectName("actEdit_Paste")
        self.actFont_Bold = QtWidgets.QAction(MainWindow)
        self.actFont_Bold.setCheckable(True)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/images/500.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actFont_Bold.setIcon(icon3)
        self.actFont_Bold.setObjectName("actFont_Bold")
        self.actFont_Italic = QtWidgets.QAction(MainWindow)
        self.actFont_Italic.setCheckable(True)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/images/502.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actFont_Italic.setIcon(icon4)
        self.actFont_Italic.setObjectName("actFont_Italic")
        self.actFont_UnderLine = QtWidgets.QAction(MainWindow)
        self.actFont_UnderLine.setCheckable(True)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icons/images/504.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actFont_UnderLine.setIcon(icon5)
        self.actFont_UnderLine.setObjectName("actFont_UnderLine")
        self.actClose = QtWidgets.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icons/images/132.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actClose.setIcon(icon6)
        self.actClose.setObjectName("actClose")
        self.actSys_ToggleText = QtWidgets.QAction(MainWindow)
        self.actSys_ToggleText.setCheckable(True)
        self.actSys_ToggleText.setChecked(True)
        self.actSys_ToggleText.setObjectName("actSys_ToggleText")
        self.actEdit_Clear = QtWidgets.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/icons/images/212.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actEdit_Clear.setIcon(icon7)
        self.actEdit_Clear.setObjectName("actEdit_Clear")
        self.actEdit_Undo = QtWidgets.QAction(MainWindow)
        self.actEdit_Undo.setEnabled(False)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/icons/images/206.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actEdit_Undo.setIcon(icon8)
        self.actEdit_Undo.setObjectName("actEdit_Undo")
        self.actEdit_Redo = QtWidgets.QAction(MainWindow)
        self.actEdit_Redo.setEnabled(False)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/icons/images/208.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actEdit_Redo.setIcon(icon9)
        self.actEdit_Redo.setObjectName("actEdit_Redo")
        self.actEdit_SelectAll = QtWidgets.QAction(MainWindow)
        self.actEdit_SelectAll.setObjectName("actEdit_SelectAll")
        self.actFile_New = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/icons/images/100.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actFile_New.setIcon(icon10)
        self.actFile_New.setObjectName("actFile_New")
        self.actFile_Open = QtWidgets.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(":/icons/images/122.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actFile_Open.setIcon(icon11)
        self.actFile_Open.setObjectName("actFile_Open")
        self.actFile_Save = QtWidgets.QAction(MainWindow)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(":/icons/images/104.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actFile_Save.setIcon(icon12)
        self.actFile_Save.setObjectName("actFile_Save")
        self.actLang_CN = QtWidgets.QAction(MainWindow)
        self.actLang_CN.setCheckable(True)
        self.actLang_CN.setChecked(True)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(":/icons/images/CN.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actLang_CN.setIcon(icon13)
        self.actLang_CN.setObjectName("actLang_CN")
        self.actLang_EN = QtWidgets.QAction(MainWindow)
        self.actLang_EN.setCheckable(True)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap(":/icons/images/timg2.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actLang_EN.setIcon(icon14)
        self.actLang_EN.setObjectName("actLang_EN")
        self.menu_E.addAction(self.actEdit_Cut)
        self.menu_E.addAction(self.actEdit_Copy)
        self.menu_E.addAction(self.actEdit_Paste)
        self.menu_E.addSeparator()
        self.menu_E.addAction(self.actEdit_Undo)
        self.menu_E.addAction(self.actEdit_Redo)
        self.menu_E.addSeparator()
        self.menu_E.addAction(self.actEdit_SelectAll)
        self.menu_E.addAction(self.actEdit_Clear)
        self.menu.addAction(self.actLang_CN)
        self.menu.addAction(self.actLang_EN)
        self.menu_F.addAction(self.actFont_Bold)
        self.menu_F.addAction(self.actFont_Italic)
        self.menu_F.addAction(self.actFont_UnderLine)
        self.menu_F.addSeparator()
        self.menu_F.addAction(self.actSys_ToggleText)
        self.menu_F.addAction(self.menu.menuAction())
        self.menu_F_2.addAction(self.actFile_New)
        self.menu_F_2.addAction(self.actFile_Open)
        self.menu_F_2.addAction(self.actFile_Save)
        self.menu_F_2.addSeparator()
        self.menu_F_2.addAction(self.actClose)
        self.menuBar.addAction(self.menu_F_2.menuAction())
        self.menuBar.addAction(self.menu_E.menuAction())
        self.menuBar.addAction(self.menu_F.menuAction())
        self.mainToolBar.addAction(self.actFile_New)
        self.mainToolBar.addAction(self.actFile_Open)
        self.mainToolBar.addAction(self.actFile_Save)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actEdit_Cut)
        self.mainToolBar.addAction(self.actEdit_Copy)
        self.mainToolBar.addAction(self.actEdit_Paste)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actEdit_Undo)
        self.mainToolBar.addAction(self.actEdit_Redo)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actFont_Bold)
        self.mainToolBar.addAction(self.actFont_Italic)
        self.mainToolBar.addAction(self.actFont_UnderLine)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actLang_CN)
        self.mainToolBar.addAction(self.actLang_EN)
        self.mainToolBar.addSeparator()

        self.retranslateUi(MainWindow)
        self.actEdit_Cut.triggered.connect(self.textEdit.cut)
        self.actEdit_Copy.triggered.connect(self.textEdit.copy)
        self.actEdit_Paste.triggered.connect(self.textEdit.paste)
        self.actEdit_Clear.triggered.connect(self.textEdit.clear)
        self.actEdit_Undo.triggered.connect(self.textEdit.undo)
        self.actEdit_Redo.triggered.connect(self.textEdit.redo)
        self.textEdit.undoAvailable['bool'].connect(self.actEdit_Undo.setEnabled)
        self.textEdit.redoAvailable['bool'].connect(self.actEdit_Redo.setEnabled)
        self.actEdit_SelectAll.triggered.connect(self.textEdit.selectAll)
        self.actClose.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Demo3_7 QAction，QMainWindow，QPlainTextEdit"))
        self.textEdit.setPlainText(_translate("MainWindow", "import sys\n"
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
"        super().__init__(parent) # 调用父类构造函数，创建窗体\n"
"        self.ui=Ui_MainWindow()     #创建UI对象\n"
"        self.ui.setupUi(self)       #构造UI界面\n"
"\n"
"        self.__buildUI()\n"
"        self.__spinFontSize.valueChanged[int].connect(self.do_fontSize_Changed)\n"
"        self.__comboFontName.currentIndexChanged[str].connect(self.do_fontName_Changed)\n"
"\n"
"        self.setCentralWidget(self.ui.textEdit)\n"
"\n"
"##  ============自定义功能函数================================        \n"
"\n"
"    def __buildUI(self):  #窗体上动态添加组件\n"
"        # 创建状态栏上的组件\n"
"        self.__LabFile=QLabel(self)     # QLabel组件显示信息\n"
"        self.__LabFile.setMinimumWidth(150)\n"
"        self.__LabFile.setText(\"文件名： \")\n"
"        self.ui.statusBar.addWidget(self.__LabFile)\n"
"\n"
"        self.__progressBar1=QProgressBar(self)      # progressBar1\n"
"        self.__progressBar1.setMaximumWidth(200)\n"
"        self.__progressBar1.setMinimum(5)\n"
"        self.__progressBar1.setMaximum(50)\n"
"        self.__progressBar1.setValue(self.ui.textEdit.font().pointSize())\n"
"        self.ui.statusBar.addWidget(self.__progressBar1)\n"
"        \n"
"        self.__LabInfo=QLabel(self)     # QLabel组件显示字体名称\n"
"        self.__LabInfo.setText(\"选择字体名称： \")\n"
"        self.ui.statusBar.addPermanentWidget(self.__LabInfo)\n"
"        \n"
"        #创建工具栏上的组件\n"
"        self.__spinFontSize=QSpinBox(self)  #字体大小spinbox\n"
"        self.__spinFontSize.setMinimum(5)\n"
"        self.__spinFontSize.setMaximum(50)\n"
"        self.__spinFontSize.setValue(self.ui.textEdit.font().pointSize())\n"
"        self.__spinFontSize.setMinimumWidth(50)\n"
"        self.ui.mainToolBar.addWidget(self.__spinFontSize) #SpinBox添加到工具栏\n"
"        \n"
"        self.__comboFontName=QFontComboBox(self)  #字体 combobox\n"
"        self.__comboFontName.setMinimumWidth(150)\n"
"        self.ui.mainToolBar.addWidget(self.__comboFontName) \n"
"\n"
"        self.ui.mainToolBar.addSeparator()  #添加一个分隔条\n"
"        \n"
"        self.ui.mainToolBar.addAction(self.ui.actClose)  #添加 actClose作为“关闭”按钮\n"
"        \n"
"        \n"
"##  ===========由connectSlotsByName() 自动连接的槽函数=====================      \n"
"\n"
"    @pyqtSlot(bool)\n"
"    def on_actFont_Bold_triggered(self, checked):     #设置粗体 \n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        if (checked == True):\n"
"            fmt.setFontWeight(QFont.Bold)\n"
"        else:\n"
"            fmt.setFontWeight(QFont.Normal)\n"
"        self.ui.textEdit.mergeCurrentCharFormat(fmt)\n"
"\n"
"    @pyqtSlot(bool)\n"
"    def on_actFont_Italic_triggered(self,checked):      #设置斜体 \n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        fmt.setFontItalic(checked)\n"
"        self.ui.textEdit.mergeCurrentCharFormat(fmt)\n"
"        \n"
"    @pyqtSlot(bool)\n"
"    def on_actFont_UnderLine_triggered(self,checked):      #设置下划线 \n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        fmt.setFontUnderline(checked)\n"
"        self.ui.textEdit.mergeCurrentCharFormat(fmt)\n"
"\n"
"    @pyqtSlot(bool)\n"
"    def on_actSys_ToggleText_triggered(self,checked):   #设置工具栏按钮样式 \n"
"        if(checked):\n"
"            self.ui.mainToolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)\n"
"        else:\n"
"            self.ui.mainToolBar.setToolButtonStyle(Qt.ToolButtonIconOnly)\n"
"\n"
"    def on_textEdit_copyAvailable(self, avi):       #文本框内容可copy\n"
"        self.ui.actEdit_Cut.setEnabled(avi)\n"
"        self.ui.actEdit_Copy.setEnabled(avi)\n"
"        self.ui.actEdit_Paste.setEnabled(self.ui.textEdit.canPaste())\n"
"                                         \n"
"    def on_textEdit_selectionChanged(self):     #文本选择内容发生变化\n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        self.ui.actFont_Bold.setChecked(fmt.font().bold())\n"
"        self.ui.actFont_Italic.setChecked(fmt.fontItalic())\n"
"        self.ui.actFont_UnderLine.setChecked(fmt.fontUnderline())\n"
"\n"
"    def on_actFile_New_triggered(self):     #新建文件，不实现具体功能\n"
"        self.__LabFile.setText(\" 新建文件 \")\n"
"\n"
"    def on_actFile_Open_triggered(self):    #打开文件，不实现具体功能\n"
"        self.__LabFile.setText(\" 打开的文件 \")\n"
"        \n"
"    def on_actFile_Save_triggered(self):    #保存文件，不实现具体功能\n"
"        self.__LabFile.setText(\" 文件已保存 \")\n"
"        \n"
"        \n"
"##  =============自定义槽函数===============================        \n"
"\n"
"    @pyqtSlot(int)\n"
"    def do_fontSize_Changed(self, fontSize):      #设置字体大小 __spinFontSize\n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        fmt.setFontPointSize(fontSize)\n"
"        self.ui.textEdit.mergeCurrentCharFormat(fmt)\n"
"        self.__progressBar1.setValue(fontSize)    #注意它属于self, 而不是self.ui\n"
"\n"
"\n"
"    @pyqtSlot(str)\n"
"    def do_fontName_Changed(self, fontName):    # 选择字体 __comboFontName\n"
"        fmt=self.ui.textEdit.currentCharFormat()\n"
"        fmt.setFontFamily(fontName)\n"
"        self.ui.textEdit.mergeCurrentCharFormat(fmt)\n"
"        self.__LabInfo.setText(\"字体名称：%s   \"%fontName)\n"
"        \n"
"   \n"
"##  ===========窗体测试程序=================================        \n"
"if  __name__ == \"__main__\":         #用于当前窗体测试\n"
"    app = QApplication(sys.argv)    #创建GUI应用程序\n"
"    form=QmyMainWindow()            #创建窗体\n"
"    form.show()\n"
"    sys.exit(app.exec_())\n"
""))
        self.menu_E.setTitle(_translate("MainWindow", "编辑(&E)"))
        self.menu_F.setTitle(_translate("MainWindow", "格式(&M)"))
        self.menu.setTitle(_translate("MainWindow", "界面语言"))
        self.menu_F_2.setTitle(_translate("MainWindow", "文件(&F)"))
        self.actEdit_Cut.setText(_translate("MainWindow", "剪切"))
        self.actEdit_Cut.setToolTip(_translate("MainWindow", "剪切到粘贴板"))
        self.actEdit_Cut.setShortcut(_translate("MainWindow", "Ctrl+X"))
        self.actEdit_Copy.setText(_translate("MainWindow", "复制"))
        self.actEdit_Copy.setToolTip(_translate("MainWindow", "复制到粘贴板"))
        self.actEdit_Copy.setShortcut(_translate("MainWindow", "Ctrl+C"))
        self.actEdit_Paste.setText(_translate("MainWindow", "粘贴"))
        self.actEdit_Paste.setToolTip(_translate("MainWindow", "从粘贴板粘贴"))
        self.actEdit_Paste.setShortcut(_translate("MainWindow", "Ctrl+V"))
        self.actFont_Bold.setText(_translate("MainWindow", "粗体"))
        self.actFont_Bold.setToolTip(_translate("MainWindow", "粗体"))
        self.actFont_Italic.setText(_translate("MainWindow", "斜体"))
        self.actFont_Italic.setToolTip(_translate("MainWindow", "斜体"))
        self.actFont_UnderLine.setText(_translate("MainWindow", "下划线"))
        self.actFont_UnderLine.setToolTip(_translate("MainWindow", "下划线"))
        self.actClose.setText(_translate("MainWindow", "关闭"))
        self.actClose.setToolTip(_translate("MainWindow", "关闭本窗口"))
        self.actSys_ToggleText.setText(_translate("MainWindow", "显示工具栏文字"))
        self.actSys_ToggleText.setToolTip(_translate("MainWindow", "显示工具栏文字"))
        self.actEdit_Clear.setText(_translate("MainWindow", "清空"))
        self.actEdit_Clear.setToolTip(_translate("MainWindow", "清空"))
        self.actEdit_Undo.setText(_translate("MainWindow", "撤销"))
        self.actEdit_Undo.setToolTip(_translate("MainWindow", "撤销上次编辑操作"))
        self.actEdit_Undo.setShortcut(_translate("MainWindow", "Ctrl+Z"))
        self.actEdit_Redo.setText(_translate("MainWindow", "重做"))
        self.actEdit_Redo.setToolTip(_translate("MainWindow", "重做上次操作"))
        self.actEdit_Redo.setShortcut(_translate("MainWindow", "Ctrl+Y"))
        self.actEdit_SelectAll.setText(_translate("MainWindow", "全选"))
        self.actEdit_SelectAll.setToolTip(_translate("MainWindow", "全选"))
        self.actEdit_SelectAll.setShortcut(_translate("MainWindow", "Ctrl+A"))
        self.actFile_New.setText(_translate("MainWindow", "新建"))
        self.actFile_New.setToolTip(_translate("MainWindow", "新建文件"))
        self.actFile_New.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actFile_Open.setText(_translate("MainWindow", "打开..."))
        self.actFile_Open.setToolTip(_translate("MainWindow", "打开文件"))
        self.actFile_Open.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actFile_Save.setText(_translate("MainWindow", "保存"))
        self.actFile_Save.setToolTip(_translate("MainWindow", "保存修改"))
        self.actFile_Save.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actLang_CN.setText(_translate("MainWindow", "汉语"))
        self.actLang_CN.setToolTip(_translate("MainWindow", "汉语界面"))
        self.actLang_EN.setText(_translate("MainWindow", "英语"))
        self.actLang_EN.setToolTip(_translate("MainWindow", "英语"))

import res_rc
