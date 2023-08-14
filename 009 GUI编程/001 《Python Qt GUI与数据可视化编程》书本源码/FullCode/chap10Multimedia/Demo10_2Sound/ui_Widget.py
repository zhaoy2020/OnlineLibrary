# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Widget.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Widget(object):
    def setupUi(self, Widget):
        Widget.setObjectName("Widget")
        Widget.resize(273, 228)
        Widget.setStyleSheet("QPushButton{min-height:30px;\n"
"}")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(Widget)
        self.verticalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox_3 = QtWidgets.QGroupBox(Widget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.btnEffect_File = QtWidgets.QPushButton(self.groupBox_3)
        self.btnEffect_File.setObjectName("btnEffect_File")
        self.verticalLayout.addWidget(self.btnEffect_File)
        self.btnEffect_Resource = QtWidgets.QPushButton(self.groupBox_3)
        self.btnEffect_Resource.setObjectName("btnEffect_Resource")
        self.verticalLayout.addWidget(self.btnEffect_Resource)
        self.verticalLayout_3.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(Widget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_2.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btnSound_File = QtWidgets.QPushButton(self.groupBox_4)
        self.btnSound_File.setObjectName("btnSound_File")
        self.verticalLayout_2.addWidget(self.btnSound_File)
        self.btnSound_Resource = QtWidgets.QPushButton(self.groupBox_4)
        self.btnSound_Resource.setObjectName("btnSound_Resource")
        self.verticalLayout_2.addWidget(self.btnSound_Resource)
        self.verticalLayout_3.addWidget(self.groupBox_4)

        self.retranslateUi(Widget)
        QtCore.QMetaObject.connectSlotsByName(Widget)

    def retranslateUi(self, Widget):
        _translate = QtCore.QCoreApplication.translate
        Widget.setWindowTitle(_translate("Widget", "QSoundEffect播放声音"))
        self.groupBox_3.setTitle(_translate("Widget", "QSoundEffect播放音频"))
        self.btnEffect_File.setText(_translate("Widget", "播放文件"))
        self.btnEffect_Resource.setText(_translate("Widget", "播放资源文件"))
        self.groupBox_4.setTitle(_translate("Widget", "QSound播放音频"))
        self.btnSound_File.setText(_translate("Widget", "播放文件"))
        self.btnSound_Resource.setText(_translate("Widget", "播放资源文件"))

