import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow, QGraphicsScene,
         QStatusBar,QWidget, QVBoxLayout, QGroupBox, QLabel,QGraphicsView, 
         QGraphicsItem,QGraphicsRectItem, QGraphicsEllipseItem)

from PyQt5.QtCore import  pyqtSlot,Qt,QRectF

from PyQt5.QtGui import  QPen,QBrush

from myGraphicsView import QmyGraphicsView   #自定义视图类

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      pass


##  ==============自定义功能函数============
   def __buildUI(self):   ##构造界面
      pass


   def __iniGraphicsSystem(self):   ##初始化 graphics View系统
      pass


##==============event 处理函数=======================
   def resizeEvent(self,event):
      pass

        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
    
        
        
##  =============自定义槽函数===============================        
   ##鼠标移动事件，point是 GraphicsView的坐标,物理坐标
   def  do_mouseMovePoint(self,point):
      pass
   

   def  do_mouseClicked(self,point):
   ##鼠标移动事件，point是 GraphicsView的坐标,物理坐标

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
