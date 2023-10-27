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
        MainWindow.resize(239, 166)
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.battery = QmyBattery(self.centralWidget)
        self.battery.setObjectName("battery")
        self.verticalLayout.addWidget(self.battery)
        self.slider = QtWidgets.QSlider(self.centralWidget)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setObjectName("slider")
        self.verticalLayout.addWidget(self.slider)
        self.LabInfo = QtWidgets.QLabel(self.centralWidget)
        self.LabInfo.setMaximumSize(QtCore.QSize(16777215, 20))
        self.LabInfo.setObjectName("LabInfo")
        self.verticalLayout.addWidget(self.LabInfo)
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow Application"))
        self.LabInfo.setText(_translate("MainWindow", "电池电量："))

from myBattery import QmyBattery
