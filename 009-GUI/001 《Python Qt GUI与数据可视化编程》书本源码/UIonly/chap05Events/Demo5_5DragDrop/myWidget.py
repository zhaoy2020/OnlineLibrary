import sys,os

from PyQt5.QtWidgets import  QApplication, QWidget

##from PyQt5.QtCore import  Qt

from PyQt5.QtGui import  QPixmap

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
   def dragEnterEvent(self,event):
      pass
   
        
   def dropEvent(self,event):
      pass
        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":     
    app = QApplication(sys.argv)
    form=QmyWidget()            
    form.show()
    sys.exit(app.exec_())
