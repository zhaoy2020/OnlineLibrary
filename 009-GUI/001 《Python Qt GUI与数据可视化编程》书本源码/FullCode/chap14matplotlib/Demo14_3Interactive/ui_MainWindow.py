# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(824, 502)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        MainWindow.setCentralWidget(self.centralWidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actQuit = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/exit_24.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actQuit.setIcon(icon)
        self.actQuit.setObjectName("actQuit")
        self.actTightLayout = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/39.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actTightLayout.setIcon(icon1)
        self.actTightLayout.setObjectName("actTightLayout")
        self.actScatterAgain = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/images/017.GIF"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actScatterAgain.setIcon(icon2)
        self.actScatterAgain.setObjectName("actScatterAgain")
        self.actSetCursor = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/images/range.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actSetCursor.setIcon(icon3)
        self.actSetCursor.setObjectName("actSetCursor")

        self.retranslateUi(MainWindow)
        self.actQuit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "QChart绘图详细功能"))
        self.actQuit.setText(_translate("MainWindow", "退出"))
        self.actQuit.setToolTip(_translate("MainWindow", "退出"))
        self.actTightLayout.setText(_translate("MainWindow", "紧凑布局"))
        self.actTightLayout.setToolTip(_translate("MainWindow", "重新紧凑布局"))
        self.actScatterAgain.setText(_translate("MainWindow", "重绘散点"))
        self.actScatterAgain.setToolTip(_translate("MainWindow", "重新生成散点数据"))
        self.actSetCursor.setText(_translate("MainWindow", "十字光标"))
        self.actSetCursor.setToolTip(_translate("MainWindow", "设置为十字光标"))

import res_rc
