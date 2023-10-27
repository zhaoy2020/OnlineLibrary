
from PyQt5.QtWidgets import  QGraphicsView

from PyQt5.QtCore import  pyqtSignal,QPoint,Qt

from PyQt5.QtGui import    QMouseEvent

class QmyGraphicsView(QGraphicsView): 

##    def __init__(self, parent=None):
##        super().__init__(parent) 

   mouseMove = pyqtSignal(QPoint)
   
   mouseClicked = pyqtSignal(QPoint)

##========== event 处理函数============
   def mouseMoveEvent(self,event): ##鼠标移动事件
      point=event.pos()          
      self.mouseMove.emit(point)    #发射信号
      super().mouseMoveEvent(event)

   def mousePressEvent(self,event): ##鼠标单击事件
      if event.button()==Qt.LeftButton :
         point=event.pos() 
         self.mouseClicked.emit(point)    #发射信号
      super().mousePressEvent(event)
      
