## appMain1.py
## 使用 ui_FormHello.py文件中的类Ui_FormHello创建App

import sys

from PyQt5 import QtWidgets

import ui_FormHello

app = QtWidgets.QApplication(sys.argv) 

baseWidget=QtWidgets.QWidget()      #创建窗口的基类QWidget的实例

ui =ui_FormHello.Ui_FormHello()     #创建UI窗口的实例
ui.setupUi(baseWidget)              #以baseWidget作为传递参数

baseWidget.show()
##ui.LabHello.setText("Hello,被程序修改")    #可以修改窗体上标签的文字

sys.exit(app.exec_()) 
