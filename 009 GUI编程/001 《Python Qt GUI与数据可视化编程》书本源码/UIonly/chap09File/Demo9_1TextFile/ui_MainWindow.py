# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(685, 445)
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.textEdit = QtWidgets.QPlainTextEdit(self.centralWidget)
        self.textEdit.setGeometry(QtCore.QRect(125, 55, 391, 241))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textEdit.setFont(font)
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 685, 23))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actQFile_Open = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/122.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actQFile_Open.setIcon(icon)
        self.actQFile_Open.setObjectName("actQFile_Open")
        self.actStream_Open = QtWidgets.QAction(MainWindow)
        self.actStream_Open.setIcon(icon)
        self.actStream_Open.setObjectName("actStream_Open")
        self.actQuit = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/132.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actQuit.setIcon(icon1)
        self.actQuit.setObjectName("actQuit")
        self.actQFile_Save = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/images/104.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actQFile_Save.setIcon(icon2)
        self.actQFile_Save.setObjectName("actQFile_Save")
        self.actStream_Save = QtWidgets.QAction(MainWindow)
        self.actStream_Save.setIcon(icon2)
        self.actStream_Save.setObjectName("actStream_Save")
        self.actPY_Open = QtWidgets.QAction(MainWindow)
        self.actPY_Open.setIcon(icon)
        self.actPY_Open.setObjectName("actPY_Open")
        self.actPY_Save = QtWidgets.QAction(MainWindow)
        self.actPY_Save.setIcon(icon2)
        self.actPY_Save.setObjectName("actPY_Save")
        self.mainToolBar.addAction(self.actQFile_Open)
        self.mainToolBar.addAction(self.actQFile_Save)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actStream_Open)
        self.mainToolBar.addAction(self.actStream_Save)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actPY_Open)
        self.mainToolBar.addAction(self.actPY_Save)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actQuit)

        self.retranslateUi(MainWindow)
        self.actQuit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Demo9_1，文本文件读写"))
        self.actQFile_Open.setText(_translate("MainWindow", "QFile打开"))
        self.actQFile_Open.setToolTip(_translate("MainWindow", "用QFile打开文本文件"))
        self.actStream_Open.setText(_translate("MainWindow", "Stream打开"))
        self.actStream_Open.setToolTip(_translate("MainWindow", "用QtextStream打开文本文件"))
        self.actQuit.setText(_translate("MainWindow", "退出"))
        self.actQuit.setToolTip(_translate("MainWindow", "退出"))
        self.actQFile_Save.setText(_translate("MainWindow", "QFile另存"))
        self.actQFile_Save.setToolTip(_translate("MainWindow", "用QFile直接另存文件"))
        self.actStream_Save.setText(_translate("MainWindow", "Stream另存"))
        self.actStream_Save.setToolTip(_translate("MainWindow", "用QTextStream另存文件"))
        self.actPY_Open.setText(_translate("MainWindow", "Python打开"))
        self.actPY_Open.setToolTip(_translate("MainWindow", "使用Python内置功能打开"))
        self.actPY_Save.setText(_translate("MainWindow", "Python另存"))
        self.actPY_Save.setToolTip(_translate("MainWindow", "采用Python内置功能另存"))


import res_rc
