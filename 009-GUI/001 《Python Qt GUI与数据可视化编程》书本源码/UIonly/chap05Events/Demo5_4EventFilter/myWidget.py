import sys

from PyQt5.QtWidgets import  QApplication,  QWidget, qApp

from PyQt5.QtCore import  Qt,QEvent

##from PyQt5.QtGui import  QKeyEvent

from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      pass


##  =================自定义功能函数=================================

        
##  ==========由connectSlotsByName() 自动连接的槽函数===============        


##  ==================event处理函数=================================
   def eventFilter(self,watched,event):
      pass
      return super().eventFilter(watched,event)
        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":     
   app = QApplication(sys.argv)
   form=QmyWidget()            
   form.show()
   sys.exit(app.exec_())
