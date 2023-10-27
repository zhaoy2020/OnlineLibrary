# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QWDialogHeaders.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QWDialogHeaders(object):
    def setupUi(self, QWDialogHeaders):
        QWDialogHeaders.setObjectName("QWDialogHeaders")
        QWDialogHeaders.resize(289, 318)
        font = QtGui.QFont()
        font.setPointSize(10)
        QWDialogHeaders.setFont(font)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(QWDialogHeaders)
        self.verticalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(QWDialogHeaders)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout.setSpacing(4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.listView = QtWidgets.QListView(self.groupBox)
        self.listView.setAlternatingRowColors(True)
        self.listView.setObjectName("listView")
        self.verticalLayout.addWidget(self.listView)
        self.horizontalLayout.addWidget(self.groupBox)
        self.frame = QtWidgets.QFrame(QWDialogHeaders)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.btnOK = QtWidgets.QPushButton(self.frame)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/704.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnOK.setIcon(icon)
        self.btnOK.setObjectName("btnOK")
        self.verticalLayout_2.addWidget(self.btnOK)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.btnCancel = QtWidgets.QPushButton(self.frame)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/706.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnCancel.setIcon(icon1)
        self.btnCancel.setObjectName("btnCancel")
        self.verticalLayout_2.addWidget(self.btnCancel)
        spacerItem2 = QtWidgets.QSpacerItem(20, 28, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem2)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem3)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem4)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem5)
        self.horizontalLayout.addWidget(self.frame)
        self.verticalLayout_3.addLayout(self.horizontalLayout)

        self.retranslateUi(QWDialogHeaders)
        self.btnOK.clicked.connect(QWDialogHeaders.accept)
        self.btnCancel.clicked.connect(QWDialogHeaders.reject)
        QtCore.QMetaObject.connectSlotsByName(QWDialogHeaders)

    def retranslateUi(self, QWDialogHeaders):
        _translate = QtCore.QCoreApplication.translate
        QWDialogHeaders.setWindowTitle(_translate("QWDialogHeaders", "设置表头标题"))
        self.groupBox.setTitle(_translate("QWDialogHeaders", "表头标题"))
        self.btnOK.setText(_translate("QWDialogHeaders", "确定"))
        self.btnCancel.setText(_translate("QWDialogHeaders", "取消"))

import res_rc
