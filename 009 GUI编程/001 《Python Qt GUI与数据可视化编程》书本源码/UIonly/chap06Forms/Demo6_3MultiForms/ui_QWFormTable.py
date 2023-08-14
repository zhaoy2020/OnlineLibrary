# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QWFormTable.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QWFormTable(object):
    def setupUi(self, QWFormTable):
        QWFormTable.setObjectName("QWFormTable")
        QWFormTable.resize(555, 341)
        self.centralwidget = QtWidgets.QWidget(QWFormTable)
        self.centralwidget.setObjectName("centralwidget")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(35, 20, 256, 192))
        self.tableView.setAlternatingRowColors(True)
        self.tableView.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tableView.setObjectName("tableView")
        self.tableView.verticalHeader().setDefaultSectionSize(25)
        QWFormTable.setCentralWidget(self.centralwidget)
        self.toolBar = QtWidgets.QToolBar(QWFormTable)
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBar.setObjectName("toolBar")
        QWFormTable.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actSetSize = QtWidgets.QAction(QWFormTable)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/230.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actSetSize.setIcon(icon)
        self.actSetSize.setObjectName("actSetSize")
        self.actSetHeader = QtWidgets.QAction(QWFormTable)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/516.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actSetHeader.setIcon(icon1)
        self.actSetHeader.setObjectName("actSetHeader")
        self.actClose = QtWidgets.QAction(QWFormTable)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/images/132.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actClose.setIcon(icon2)
        self.actClose.setObjectName("actClose")
        self.toolBar.addAction(self.actSetSize)
        self.toolBar.addAction(self.actSetHeader)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actClose)

        self.retranslateUi(QWFormTable)
        self.actClose.triggered.connect(QWFormTable.close)
        QtCore.QMetaObject.connectSlotsByName(QWFormTable)

    def retranslateUi(self, QWFormTable):
        _translate = QtCore.QCoreApplication.translate
        QWFormTable.setWindowTitle(_translate("QWFormTable", "Table数据"))
        self.toolBar.setWindowTitle(_translate("QWFormTable", "toolBar"))
        self.actSetSize.setText(_translate("QWFormTable", "设置表格大小"))
        self.actSetSize.setToolTip(_translate("QWFormTable", "设置表格大小"))
        self.actSetHeader.setText(_translate("QWFormTable", "设置表头"))
        self.actSetHeader.setToolTip(_translate("QWFormTable", "设置表头文字"))
        self.actClose.setText(_translate("QWFormTable", "关闭"))
        self.actClose.setToolTip(_translate("QWFormTable", "关闭本窗口"))

import res_rc
