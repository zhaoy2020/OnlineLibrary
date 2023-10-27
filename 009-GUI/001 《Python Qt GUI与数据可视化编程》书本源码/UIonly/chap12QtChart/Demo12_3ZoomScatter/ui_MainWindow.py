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
        MainWindow.resize(770, 501)
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 770, 23))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actZoomReset = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/414.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actZoomReset.setIcon(icon)
        self.actZoomReset.setObjectName("actZoomReset")
        self.actQuit = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/132.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actQuit.setIcon(icon1)
        self.actQuit.setObjectName("actQuit")
        self.actZoomIn = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/images/418.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actZoomIn.setIcon(icon2)
        self.actZoomIn.setObjectName("actZoomIn")
        self.actZoomOut = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/images/416.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actZoomOut.setIcon(icon3)
        self.actZoomOut.setObjectName("actZoomOut")
        self.mainToolBar.addAction(self.actZoomIn)
        self.mainToolBar.addAction(self.actZoomOut)
        self.mainToolBar.addAction(self.actZoomReset)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actQuit)

        self.retranslateUi(MainWindow)
        self.actQuit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "QScatterSeries序列与鼠标选择缩放"))
        self.actZoomReset.setText(_translate("MainWindow", "原始大小"))
        self.actZoomReset.setToolTip(_translate("MainWindow", "原始大小"))
        self.actQuit.setText(_translate("MainWindow", "退出"))
        self.actQuit.setToolTip(_translate("MainWindow", "退出"))
        self.actZoomIn.setText(_translate("MainWindow", "放大"))
        self.actZoomIn.setToolTip(_translate("MainWindow", "放大"))
        self.actZoomOut.setText(_translate("MainWindow", "缩小"))
        self.actZoomOut.setToolTip(_translate("MainWindow", "缩小"))

import res_rc
