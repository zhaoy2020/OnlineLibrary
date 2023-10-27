from PyQt5.QtWidgets import  QGraphicsView

from PyQt5.QtCore import  pyqtSignal,QPoint,Qt, QRectF

from PyQt5.QtGui import    QMouseEvent,QKeyEvent

from PyQt5.QtChart import QChartView

class QmyChartView(QChartView): 

   mouseMove = pyqtSignal(QPoint)   ##鼠标移动

   def __init__(self, parent=None):
      super().__init__(parent) 
      self.setDragMode(QGraphicsView.RubberBandDrag)

      self.__beginPoint=QPoint()
      self.__endPoint=QPoint()


##========== event 处理函数============
   def mousePressEvent(self,event): ##鼠标单击
      if event.button()==Qt.LeftButton :
         self.__beginPoint=event.pos()    
      super().mousePressEvent(event)
      

   def mouseMoveEvent(self,event): ##鼠标移动
      point=event.pos()   
      self.mouseMove.emit(point)  #发射信号
      super().mouseMoveEvent(event)

   def mouseReleaseEvent(self,event): ##鼠标框选放大，右键恢复
      if event.button()==Qt.LeftButton:
         self.__endPoint=event.pos()    
         rectF=QRectF()
         rectF.setTopLeft(self.__beginPoint)
         rectF.setBottomRight(self.__endPoint)
         self.chart().zoomIn(rectF)
      elif event.button()==Qt.RightButton:
         self.chart().zoomReset() #鼠标右键释放，resetZoom

      super().mouseReleaseEvent(event)

   def keyPressEvent(self,event):   ##按键按下
      key=event.key()
      if key==Qt.Key_Plus:  # +键
         self.chart().zoom(1.2)
      elif key==Qt.Key_Minus:  
         self.chart().zoom(0.8)
      elif key==Qt.Key_Left:  
         self.chart().scroll(10,0)
      elif key==Qt.Key_Right:  
         self.chart().scroll(-10,0)
      elif key==Qt.Key_Up:  
         self.chart().scroll(0,-10)
      elif key==Qt.Key_Down:  
         self.chart().scroll(0,10)
      elif key==Qt.Key_PageUp:  
         self.chart().scroll(0,-50)
      elif key==Qt.Key_PageDown:  
         self.chart().scroll(0,50)
      elif key==Qt.Key_Home:  
         self.chart().zoomReset()
      
      super().keyPressEvent(event)
      
