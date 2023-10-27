import sys

from PyQt5.QtWidgets import  QApplication, QWidget

##from PyQt5.QtCore import  pyqtSlot

from PyQt5.QtCore import   QTime, QTimer

from ui_Widget import Ui_Widget


class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent) #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()      #创建UI对象
      self.ui.setupUi(self)    #构造UI界面
      pass

        
##  =========由connectSlotsByName() 自动连接的槽函数============= 
   def on_btnStart_clicked(self):   ##“开始”按钮
      pass

   def on_btnSetIntv_clicked(self):    ##设置定时器的周期  
      pass

   def on_btnStop_clicked(self):    ##“停止”按钮
      pass
        
        
##  ==========自定义槽函数==================================        
   def do_timer_timeout(self):     ##定时中断响应
      pass

   
##  =========窗体测试程序===================================        
if  __name__ == "__main__":       
   app = QApplication(sys.argv)  
   form=QmyWidget()         
   form.show()
   sys.exit(app.exec_())
