# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QWDialogSize.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QWDialogSize(object):
    def setupUi(self, QWDialogSize):
        QWDialogSize.setObjectName("QWDialogSize")
        QWDialogSize.setWindowModality(QtCore.Qt.NonModal)
        QWDialogSize.resize(266, 132)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(QWDialogSize.sizePolicy().hasHeightForWidth())
        QWDialogSize.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        QWDialogSize.setFont(font)
        QWDialogSize.setSizeGripEnabled(False)
        QWDialogSize.setModal(False)
        self.horizontalLayout = QtWidgets.QHBoxLayout(QWDialogSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(QWDialogSize)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setVerticalSpacing(15)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.spin_ColCount = QtWidgets.QSpinBox(self.groupBox)
        self.spin_ColCount.setMinimum(1)
        self.spin_ColCount.setMaximum(500)
        self.spin_ColCount.setProperty("value", 6)
        self.spin_ColCount.setObjectName("spin_ColCount")
        self.gridLayout.addWidget(self.spin_ColCount, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.spin_RwoCount = QtWidgets.QSpinBox(self.groupBox)
        self.spin_RwoCount.setMinimum(1)
        self.spin_RwoCount.setMaximum(500)
        self.spin_RwoCount.setProperty("value", 10)
        self.spin_RwoCount.setObjectName("spin_RwoCount")
        self.gridLayout.addWidget(self.spin_RwoCount, 1, 1, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox)
        self.frame = QtWidgets.QFrame(QWDialogSize)
        self.frame.setMaximumSize(QtCore.QSize(90, 16777215))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.btnOK = QtWidgets.QPushButton(self.frame)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/704.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnOK.setIcon(icon)
        self.btnOK.setObjectName("btnOK")
        self.verticalLayout.addWidget(self.btnOK)
        self.btnCancel = QtWidgets.QPushButton(self.frame)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/706.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnCancel.setIcon(icon1)
        self.btnCancel.setObjectName("btnCancel")
        self.verticalLayout.addWidget(self.btnCancel)
        self.horizontalLayout.addWidget(self.frame)

        self.retranslateUi(QWDialogSize)
        self.btnOK.clicked.connect(QWDialogSize.accept)
        self.btnCancel.clicked.connect(QWDialogSize.reject)
        QtCore.QMetaObject.connectSlotsByName(QWDialogSize)

    def retranslateUi(self, QWDialogSize):
        _translate = QtCore.QCoreApplication.translate
        QWDialogSize.setWindowTitle(_translate("QWDialogSize", "设置表格行数和列数"))
        self.groupBox.setTitle(_translate("QWDialogSize", "设置表格行数和列数"))
        self.label_2.setText(_translate("QWDialogSize", "列  数"))
        self.label.setText(_translate("QWDialogSize", "行  数"))
        self.btnOK.setText(_translate("QWDialogSize", "确定"))
        self.btnCancel.setText(_translate("QWDialogSize", "取消"))

import res_rc
