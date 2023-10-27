# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Widget.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Widget(object):
    def setupUi(self, Widget):
        Widget.setObjectName("Widget")
        Widget.resize(392, 266)
        font = QtGui.QFont()
        font.setPointSize(10)
        Widget.setFont(font)
        Widget.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.btnTest = QtWidgets.QPushButton(Widget)
        self.btnTest.setGeometry(QtCore.QRect(200, 70, 146, 51))
        self.btnTest.setObjectName("btnTest")
        self.btnMove = QtWidgets.QPushButton(Widget)
        self.btnMove.setGeometry(QtCore.QRect(30, 170, 141, 51))
        self.btnMove.setObjectName("btnMove")
        self.LabMove = QtWidgets.QLabel(Widget)
        self.LabMove.setGeometry(QtCore.QRect(55, 35, 196, 41))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.LabMove.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.LabMove.setFont(font)
        self.LabMove.setObjectName("LabMove")

        self.retranslateUi(Widget)
        QtCore.QMetaObject.connectSlotsByName(Widget)

    def retranslateUi(self, Widget):
        _translate = QtCore.QCoreApplication.translate
        Widget.setWindowTitle(_translate("Widget", "Demo5_1, 缺省的事件处理函数"))
        self.btnTest.setText(_translate("Widget", "Button at Center\n"
"resizeEvent事件"))
        self.btnMove.setText(_translate("Widget", "Movable Button\n"
"W,S,A,D键移动"))
        self.LabMove.setText(_translate("Widget", "点击鼠标左键"))

