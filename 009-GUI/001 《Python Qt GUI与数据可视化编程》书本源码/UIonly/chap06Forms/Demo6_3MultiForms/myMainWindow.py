import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow

from PyQt5.QtCore import  pyqtSlot, Qt

##from PyQt5.QtWidgets import  

from PyQt5.QtGui import QPainter, QPixmap

from ui_MainWindow import Ui_MainWindow

from myFormDoc import QmyFormDoc

from myFormTable import QmyFormTable

class QmyMainWindow(QMainWindow):
   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      self.setCentralWidget(self.ui.tabWidget)
      pass


##  ==============自定义功能函数============

##  =============event事件处理函数============
   def paintEvent(self,event):
      pass
      super().paintEvent(event)
      
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ## "嵌入式Widget"
   def on_actWidgetInsite_triggered(self):
      pass


   @pyqtSlot()    ##独立Widget窗口
   def on_actWidget_triggered(self):
      pass


   @pyqtSlot()    ##"嵌入式MainWindow"
   def on_actWindowInsite_triggered(self):
      pass


   @pyqtSlot()    ##"独立MainWindow窗口"
   def on_actWindow_triggered(self):
      pass


   def on_tabWidget_currentChanged(self,index):    ##tabWidget当前页面变化
      pass
   

   def on_tabWidget_tabCloseRequested(self,index):  ##分页关闭时关闭窗体
      pass
        
    
##  =============自定义槽函数===============================        
   @pyqtSlot(str)
   def do_docFileChanged(self,shotFilename):   ##显示文件名
      pass
       
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyMainWindow()
   form.show()
   sys.exit(app.exec_())
