# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FormHello.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FormHello(object):
    def setupUi(self, FormHello):
        FormHello.setObjectName("FormHello")
        FormHello.resize(283, 151)
        self.LabHello = QtWidgets.QLabel(FormHello)
        self.LabHello.setGeometry(QtCore.QRect(45, 40, 189, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.LabHello.setFont(font)
        self.LabHello.setScaledContents(False)
        self.LabHello.setObjectName("LabHello")
        self.btnClose = QtWidgets.QPushButton(FormHello)
        self.btnClose.setGeometry(QtCore.QRect(105, 90, 75, 23))
        self.btnClose.setObjectName("btnClose")

        self.retranslateUi(FormHello)
        QtCore.QMetaObject.connectSlotsByName(FormHello)

    def retranslateUi(self, FormHello):
        _translate = QtCore.QCoreApplication.translate
        FormHello.setWindowTitle(_translate("FormHello", "Demo2_2"))
        self.LabHello.setText(_translate("FormHello", "Hello, by UI Designer"))
        self.btnClose.setText(_translate("FormHello", "关闭"))


