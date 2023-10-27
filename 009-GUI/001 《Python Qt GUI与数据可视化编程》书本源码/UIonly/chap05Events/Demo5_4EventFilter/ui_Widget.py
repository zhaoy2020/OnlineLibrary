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
        Widget.resize(265, 154)
        self.verticalLayout = QtWidgets.QVBoxLayout(Widget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(20)
        self.verticalLayout.setObjectName("verticalLayout")
        self.LabHover = QtWidgets.QLabel(Widget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.LabHover.setFont(font)
        self.LabHover.setAutoFillBackground(False)
        self.LabHover.setStyleSheet("")
        self.LabHover.setAlignment(QtCore.Qt.AlignCenter)
        self.LabHover.setObjectName("LabHover")
        self.verticalLayout.addWidget(self.LabHover)
        self.LabDBClick = QtWidgets.QLabel(Widget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.LabDBClick.setFont(font)
        self.LabDBClick.setAutoFillBackground(False)
        self.LabDBClick.setStyleSheet("")
        self.LabDBClick.setAlignment(QtCore.Qt.AlignCenter)
        self.LabDBClick.setObjectName("LabDBClick")
        self.verticalLayout.addWidget(self.LabDBClick)

        self.retranslateUi(Widget)
        QtCore.QMetaObject.connectSlotsByName(Widget)

    def retranslateUi(self, Widget):
        _translate = QtCore.QCoreApplication.translate
        Widget.setWindowTitle(_translate("Widget", "Demo5_4，事件过滤器"))
        self.LabHover.setText(_translate("Widget", "靠近我，点击我"))
        self.LabDBClick.setText(_translate("Widget", "可双击的标签"))

