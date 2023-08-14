# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QWDialogPen.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QWDialogPen(object):
    def setupUi(self, QWDialogPen):
        QWDialogPen.setObjectName("QWDialogPen")
        QWDialogPen.resize(319, 139)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        QWDialogPen.setFont(font)
        self.horizontalLayout = QtWidgets.QHBoxLayout(QWDialogPen)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(QWDialogPen)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(98, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 4, 2, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(98, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 2, 2, 1, 1)
        self.spinWidth = QtWidgets.QSpinBox(self.groupBox)
        self.spinWidth.setMinimumSize(QtCore.QSize(100, 0))
        self.spinWidth.setMinimum(1)
        self.spinWidth.setMaximum(100)
        self.spinWidth.setObjectName("spinWidth")
        self.gridLayout.addWidget(self.spinWidth, 2, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.comboPenStyle = QtWidgets.QComboBox(self.groupBox)
        self.comboPenStyle.setObjectName("comboPenStyle")
        self.gridLayout.addWidget(self.comboPenStyle, 0, 1, 1, 2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 1, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem3, 3, 1, 1, 1)
        self.btnColor = QtWidgets.QPushButton(self.groupBox)
        self.btnColor.setAutoFillBackground(False)
        self.btnColor.setStyleSheet("background-color: rgb(255, 0, 255);")
        self.btnColor.setText("")
        self.btnColor.setFlat(False)
        self.btnColor.setObjectName("btnColor")
        self.gridLayout.addWidget(self.btnColor, 4, 1, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem4)
        self.btnOK = QtWidgets.QPushButton(QWDialogPen)
        self.btnOK.setObjectName("btnOK")
        self.verticalLayout.addWidget(self.btnOK)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem5)
        self.btnCancel = QtWidgets.QPushButton(QWDialogPen)
        self.btnCancel.setObjectName("btnCancel")
        self.verticalLayout.addWidget(self.btnCancel)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem6)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(QWDialogPen)
        self.btnOK.clicked.connect(QWDialogPen.accept)
        self.btnCancel.clicked.connect(QWDialogPen.reject)
        QtCore.QMetaObject.connectSlotsByName(QWDialogPen)

    def retranslateUi(self, QWDialogPen):
        _translate = QtCore.QCoreApplication.translate
        QWDialogPen.setWindowTitle(_translate("QWDialogPen", "QPen属性设置对话框"))
        self.groupBox.setTitle(_translate("QWDialogPen", "Pen属性设置"))
        self.label.setText(_translate("QWDialogPen", "线 型"))
        self.label_3.setText(_translate("QWDialogPen", "颜 色"))
        self.label_2.setText(_translate("QWDialogPen", "线 宽"))
        self.btnOK.setText(_translate("QWDialogPen", "确 定"))
        self.btnCancel.setText(_translate("QWDialogPen", "取 消"))

