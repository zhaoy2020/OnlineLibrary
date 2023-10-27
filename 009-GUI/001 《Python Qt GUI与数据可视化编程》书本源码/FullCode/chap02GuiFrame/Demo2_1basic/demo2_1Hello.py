## demo2_1Hello.py
## 使用PyQt5，纯代码创建一个简单的GUI程序

import sys

from PyQt5 import QtCore, QtGui, QtWidgets  #导入PyQt5包中的几个模块

app = QtWidgets.QApplication(sys.argv)      #创建App，用QApplication类

widgetHello = QtWidgets.QWidget()      #创建一个窗体widgetHello，用QWidget类
widgetHello.resize(280,150)            #设置对话框的宽度和高度
widgetHello.setWindowTitle("Demo2_1")  #设置对话框的标题文字

LabHello = QtWidgets.QLabel(widgetHello)  #创建一个标签LabHello，父容器为widgetHello
LabHello.setText("Hello World, PyQt5")    #设置标签文字

font = QtGui.QFont()    #创建字体对象font，用QFont类
font.setPointSize(12)   #设置字体大小
font.setBold(True)      #设置为粗体
LabHello.setFont(font)  #设置为标签LabHello的字体

size=LabHello.sizeHint()    #获取LabHello的合适大小，返回值aLabSize是QSize类对象

LabHello.setGeometry(70, 60, size.width(), size.height())
##设置LabHello的位置和大小，位置x=70,y=60， 宽度和高度由aLabSize的值确定

widgetHello.show()      #显示对话框

sys.exit(app.exec_())   #应用程序运行
