## 单继承方法，能更好的进行界面与逻辑的分离

import sys

from PyQt5.QtWidgets import  QWidget, QApplication

from ui_FormHello import Ui_FormHello

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建QWidget窗口

      self.__ui=Ui_FormHello()   #创建UI对象
      self.__ui.setupUi(self)    #构造UI界面
      self.Lab="单继承的QmyWidget"
      self.__ui.LabHello.setText(self.Lab)

   def setBtnText(self, aText):
      self.__ui.btnClose.setText(aText)
        
    
if  __name__ == "__main__":
   app = QApplication(sys.argv)   #创建App，用QApplication类

   myWidget=QmyWidget()
   myWidget.show()
   myWidget.setBtnText("间接设置")
   sys.exit(app.exec_()) 
