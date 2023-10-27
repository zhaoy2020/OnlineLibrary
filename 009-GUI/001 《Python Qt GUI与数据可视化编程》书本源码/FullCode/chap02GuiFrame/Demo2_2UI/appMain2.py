## 多继承，界面组件对象容易与新类定义的变量混淆

import sys

from PyQt5.QtWidgets import  QWidget, QApplication

from ui_FormHello import Ui_FormHello

class QmyWidget(QWidget,Ui_FormHello): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建QWidget窗口

      self.Lab="多继承的QmyWidget"     #新定义的一个变量
      self.setupUi(self)               #self就是QWidget窗体，可以作为参数传给setupUi()
      self.LabHello.setText(self.Lab)  #窗体上的LabHello
    
if  __name__ == "__main__":
   app = QApplication(sys.argv)        #创建App
   myWidget=QmyWidget()
   myWidget.show()
   myWidget.btnClose.setText("不关闭了")
   sys.exit(app.exec_()) 
