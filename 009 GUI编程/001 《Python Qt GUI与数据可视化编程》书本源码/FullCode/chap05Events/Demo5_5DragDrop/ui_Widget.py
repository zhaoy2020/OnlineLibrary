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
        Widget.resize(468, 462)
        Widget.setMaximumSize(QtCore.QSize(680, 480))
        font = QtGui.QFont()
        font.setPointSize(10)
        Widget.setFont(font)
        Widget.setAcceptDrops(True)
        self.LabPic = QtWidgets.QLabel(Widget)
        self.LabPic.setGeometry(QtCore.QRect(9, 155, 450, 298))
        self.LabPic.setAcceptDrops(True)
        self.LabPic.setText("")
        self.LabPic.setPixmap(QtGui.QPixmap("../../demo5_1SpecificEvents/QtApp/images/sea1.jpg"))
        self.LabPic.setScaledContents(True)
        self.LabPic.setAlignment(QtCore.Qt.AlignCenter)
        self.LabPic.setObjectName("LabPic")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(Widget)
        self.plainTextEdit.setGeometry(QtCore.QRect(9, 9, 450, 140))
        self.plainTextEdit.setMinimumSize(QtCore.QSize(0, 140))
        self.plainTextEdit.setObjectName("plainTextEdit")

        self.retranslateUi(Widget)
        QtCore.QMetaObject.connectSlotsByName(Widget)

    def retranslateUi(self, Widget):
        _translate = QtCore.QCoreApplication.translate
        Widget.setWindowTitle(_translate("Widget", "Demo5_5，外部文件拖放"))

