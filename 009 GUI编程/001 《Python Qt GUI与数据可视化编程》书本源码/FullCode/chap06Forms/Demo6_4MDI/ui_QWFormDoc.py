# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QWFormDoc.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QWFormDoc(object):
    def setupUi(self, QWFormDoc):
        QWFormDoc.setObjectName("QWFormDoc")
        QWFormDoc.resize(666, 401)
        font = QtGui.QFont()
        font.setPointSize(10)
        QWFormDoc.setFont(font)
        self.verticalLayout = QtWidgets.QVBoxLayout(QWFormDoc)
        self.verticalLayout.setContentsMargins(3, 3, 3, 3)
        self.verticalLayout.setSpacing(3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(QWFormDoc)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.verticalLayout.addWidget(self.plainTextEdit)

        self.retranslateUi(QWFormDoc)
        QtCore.QMetaObject.connectSlotsByName(QWFormDoc)

    def retranslateUi(self, QWFormDoc):
        _translate = QtCore.QCoreApplication.translate
        QWFormDoc.setWindowTitle(_translate("QWFormDoc", "new document"))

