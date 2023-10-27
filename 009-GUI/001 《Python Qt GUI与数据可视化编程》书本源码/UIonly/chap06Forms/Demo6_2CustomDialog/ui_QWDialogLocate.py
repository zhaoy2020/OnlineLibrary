# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QWDialogLocate.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QWDialogLocate(object):
    def setupUi(self, QWDialogLocate):
        QWDialogLocate.setObjectName("QWDialogLocate")
        QWDialogLocate.resize(313, 148)
        font = QtGui.QFont()
        font.setPointSize(10)
        QWDialogLocate.setFont(font)
        self.horizontalLayout = QtWidgets.QHBoxLayout(QWDialogLocate)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(QWDialogLocate)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.spinColumn = QtWidgets.QSpinBox(self.groupBox)
        self.spinColumn.setMinimum(0)
        self.spinColumn.setObjectName("spinColumn")
        self.gridLayout.addWidget(self.spinColumn, 0, 1, 1, 1)
        self.chkBoxColumn = QtWidgets.QCheckBox(self.groupBox)
        self.chkBoxColumn.setObjectName("chkBoxColumn")
        self.gridLayout.addWidget(self.chkBoxColumn, 0, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.spinRow = QtWidgets.QSpinBox(self.groupBox)
        self.spinRow.setObjectName("spinRow")
        self.gridLayout.addWidget(self.spinRow, 1, 1, 1, 1)
        self.chkBoxRow = QtWidgets.QCheckBox(self.groupBox)
        self.chkBoxRow.setObjectName("chkBoxRow")
        self.gridLayout.addWidget(self.chkBoxRow, 1, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.editCellText = QtWidgets.QLineEdit(self.groupBox)
        self.editCellText.setObjectName("editCellText")
        self.gridLayout.addWidget(self.editCellText, 2, 1, 1, 2)
        self.horizontalLayout.addWidget(self.groupBox)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.btnSetText = QtWidgets.QPushButton(QWDialogLocate)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/322.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnSetText.setIcon(icon)
        self.btnSetText.setObjectName("btnSetText")
        self.verticalLayout.addWidget(self.btnSetText)
        self.btnClose = QtWidgets.QPushButton(QWDialogLocate)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/132.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnClose.setIcon(icon1)
        self.btnClose.setObjectName("btnClose")
        self.verticalLayout.addWidget(self.btnClose)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(QWDialogLocate)
        self.btnClose.clicked.connect(QWDialogLocate.close)
        QtCore.QMetaObject.connectSlotsByName(QWDialogLocate)

    def retranslateUi(self, QWDialogLocate):
        _translate = QtCore.QCoreApplication.translate
        QWDialogLocate.setWindowTitle(_translate("QWDialogLocate", "单元格定位与文字设置"))
        self.label.setText(_translate("QWDialogLocate", "列  号"))
        self.chkBoxColumn.setText(_translate("QWDialogLocate", "列增"))
        self.label_2.setText(_translate("QWDialogLocate", "行  号"))
        self.chkBoxRow.setText(_translate("QWDialogLocate", "行增"))
        self.label_3.setText(_translate("QWDialogLocate", "设定文字"))
        self.btnSetText.setText(_translate("QWDialogLocate", "设定文字"))
        self.btnClose.setText(_translate("QWDialogLocate", "关  闭"))

import res_rc
