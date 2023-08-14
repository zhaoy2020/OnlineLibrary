##与UI窗体类对应的业务逻辑类

import sys

from PyQt5.QtWidgets import  QApplication, QDialog

from PyQt5.QtGui import  QPalette

from PyQt5.QtCore import Qt, pyqtSlot

from ui_Dialog import Ui_Dialog

class QmyDialog(QDialog): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Dialog()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      pass

## 由connectSlotsByName() 自动与组件的信号连接的槽函数
##    @pyqtSlot()
   def on_btnClear_clicked(self):
      pass

##    @pyqtSlot()
   def on_chkBoxBold_toggled(self,checked):     #"Bold"
      pass

##    @pyqtSlot()
   def on_chkBoxUnder_clicked(self):  #"Underline"
      pass

   @pyqtSlot(bool)     #修饰符指定参数类型，用于overload型的信号
   def on_chkBoxItalic_clicked(self,checked):
      pass

## 自定义槽函数
   def do_setTextColor(self):
      pass
   
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   form=QmyDialog()                 #创建窗体
   form.show()
   sys.exit(app.exec_()) 
