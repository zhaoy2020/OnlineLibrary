##与UI窗体类对应的业务逻辑类

import sys

from PyQt5.QtWidgets import  QApplication, QWidget

from PyQt5.QtCore import  pyqtSlot

from PyQt5.QtGui import  QIcon

from ui_Widget import Ui_Widget

from human import Human

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)      #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()           #创建UI对象
      self.ui.setupUi(self)         #构造UI界面
      pass
        
## 由connectSlotsByName() 自动与组件的信号连接的槽函数
   def on_sliderSetAge_valueChanged(self,value):
      pass

   def on_btnSetName_clicked(self):
      pass
        

## 自定义槽函数
   def do_nameChanged(self,name):
      pass

   @pyqtSlot(int)
   def do_ageChanged_int(self,age):
      pass

   @pyqtSlot(str)
   def do_ageChanged_str(self,info):
      pass
   
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   icon = QIcon(":/icons/images/app.ico")
   app.setWindowIcon(icon)
   form=QmyWidget()                 #创建窗体
   form.show()
   sys.exit(app.exec_())
