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
        MainWindow.resize(827, 520)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralWidget)
        self.tabWidget.setGeometry(QtCore.QRect(70, 30, 336, 196))
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab, "")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 827, 23))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setAutoFillBackground(False)
        self.mainToolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actWindowInsite = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/808.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actWindowInsite.setIcon(icon)
        self.actWindowInsite.setObjectName("actWindowInsite")
        self.actWidgetInsite = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/430.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actWidgetInsite.setIcon(icon1)
        self.actWidgetInsite.setObjectName("actWidgetInsite")
        self.actQuit = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/images/132.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actQuit.setIcon(icon2)
        self.actQuit.setObjectName("actQuit")
        self.actWindow = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/images/804.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actWindow.setIcon(icon3)
        self.actWindow.setObjectName("actWindow")
        self.actWidget = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/images/806.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actWidget.setIcon(icon4)
        self.actWidget.setObjectName("actWidget")
        self.mainToolBar.addAction(self.actWidgetInsite)
        self.mainToolBar.addAction(self.actWidget)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actWindowInsite)
        self.mainToolBar.addAction(self.actWindow)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actQuit)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.actQuit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Demo6_3  多窗口应用程序"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Page"))
        self.actWindowInsite.setText(_translate("MainWindow", "嵌入式MainWindow"))
        self.actWindowInsite.setToolTip(_translate("MainWindow", "嵌入式MainWindow"))
        self.actWidgetInsite.setText(_translate("MainWindow", "嵌入式Widget"))
        self.actWidgetInsite.setToolTip(_translate("MainWindow", "Widget嵌入式窗体"))
        self.actQuit.setText(_translate("MainWindow", "退出"))
        self.actQuit.setToolTip(_translate("MainWindow", "退出本系统"))
        self.actWindow.setText(_translate("MainWindow", "独立MainWindow窗口"))
        self.actWindow.setToolTip(_translate("MainWindow", "独立MainWindow窗口"))
        self.actWidget.setText(_translate("MainWindow", "独立Widget窗口"))
        self.actWidget.setToolTip(_translate("MainWindow", "新建Widget独立窗口"))

import res_rc
