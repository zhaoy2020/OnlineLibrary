
from PyQt5.QtWidgets import  QGraphicsView

from PyQt5.QtCore import  pyqtSignal,QPoint,Qt

from PyQt5.QtGui import    QMouseEvent,QKeyEvent

class QmyGraphicsView(QGraphicsView): 

##    def __init__(self, parent=None):
##        super().__init__(parent) 

   mouseMove = pyqtSignal(QPoint)      #鼠标移动
   
   mouseClicked = pyqtSignal(QPoint)   #鼠标单击
   
   mouseDoubleClick = pyqtSignal(QPoint)   #鼠标双击

   keyPress = pyqtSignal(QKeyEvent)    #按键按下

##========== event 处理函数============
   def mouseMoveEvent(self,event): ##鼠标移动
      point=event.pos()   
      self.mouseMove.emit(point)  #发射信号
      super().mouseMoveEvent(event)

   def mousePressEvent(self,event): ##鼠标单击
      if event.button()==Qt.LeftButton :
         point=event.pos()    
         self.mouseClicked.emit(point) #发射信号
      super().mousePressEvent(event)

   def mouseDoubleClickEvent(self,event): ##鼠标双击
      if event.button()==Qt.LeftButton :
         point=event.pos()    
         self.mouseDoubleClick.emit(point)    #发射信号
      super().mouseDoubleClickEvent(event)

   def keyPressEvent(self,event):   ##按键按下
      self.keyPress.emit(event)    #发射信号
      super().keyPressEvent(event)
      
