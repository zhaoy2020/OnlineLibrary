import sys

from PyQt5.QtWidgets import  QApplication, QWidget,QMessageBox

from PyQt5.QtCore import  pyqtSlot, Qt, QEvent

from PyQt5.QtGui import   QPainter, QPixmap,QCursor

from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

##  =================自定义功能函数=================================
        
        
##  ==========由connectSlotsByName() 自动连接的槽函数===============        


##  ==================event处理函数=================================
   def event(self,event):
      return super().event(event)


   def paintEvent(self,event):   ##绘制窗口背景图片
      pass


   def resizeEvent(self,event):  ##窗体改变大小
      pass

        
   def closeEvent(self,event):   ##窗体关闭时询问
      pass


   def keyPressEvent(self,event):   ##键盘按键按下
      pass
   

   def hideEvent(self,event):   ##窗体最小化时触发
      pass
   
        
   def showEvent(self,event):   ##窗体显示时触发
      pass
   

   def mousePressEvent(self,event):  ##鼠标按键按下
      pass
      super().mousePressEvent(event)
       
        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   form=QmyWidget()                 #创建窗体
   form.show()
   sys.exit(app.exec_())
