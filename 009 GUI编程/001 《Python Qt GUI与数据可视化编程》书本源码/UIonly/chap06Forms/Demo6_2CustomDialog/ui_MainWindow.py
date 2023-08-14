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
        MainWindow.resize(567, 367)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.tableView = QtWidgets.QTableView(self.centralWidget)
        self.tableView.setGeometry(QtCore.QRect(20, 15, 506, 236))
        self.tableView.setAlternatingRowColors(True)
        self.tableView.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tableView.setObjectName("tableView")
        self.tableView.verticalHeader().setDefaultSectionSize(25)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 567, 23))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actTab_SetSize = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/230.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actTab_SetSize.setIcon(icon)
        self.actTab_SetSize.setObjectName("actTab_SetSize")
        self.actTab_SetHeader = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/516.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actTab_SetHeader.setIcon(icon1)
        self.actTab_SetHeader.setObjectName("actTab_SetHeader")
        self.actFile_Quit = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/images/132.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actFile_Quit.setIcon(icon2)
        self.actFile_Quit.setObjectName("actFile_Quit")
        self.actTab_Locate = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/images/304.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actTab_Locate.setIcon(icon3)
        self.actTab_Locate.setObjectName("actTab_Locate")
        self.mainToolBar.addAction(self.actTab_SetSize)
        self.mainToolBar.addAction(self.actTab_SetHeader)
        self.mainToolBar.addAction(self.actTab_Locate)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.actFile_Quit)

        self.retranslateUi(MainWindow)
        self.actFile_Quit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Demo6_2 自定义对话框及其调用"))
        self.actTab_SetSize.setText(_translate("MainWindow", "设置行数列数"))
        self.actTab_SetSize.setToolTip(_translate("MainWindow", "设置表格行数列数"))
        self.actTab_SetHeader.setText(_translate("MainWindow", "设置表头标题"))
        self.actTab_SetHeader.setToolTip(_translate("MainWindow", "设置表头标题"))
        self.actFile_Quit.setText(_translate("MainWindow", "退出"))
        self.actFile_Quit.setToolTip(_translate("MainWindow", "退出本软件"))
        self.actTab_Locate.setText(_translate("MainWindow", "定位单元格"))
        self.actTab_Locate.setToolTip(_translate("MainWindow", "定位到某个单元格"))

import res_rc
