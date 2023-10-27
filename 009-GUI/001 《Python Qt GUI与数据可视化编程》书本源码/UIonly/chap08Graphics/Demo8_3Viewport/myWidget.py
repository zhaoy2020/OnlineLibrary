import sys, math

from PyQt5.QtWidgets import  QApplication, QWidget

from PyQt5.QtCore import  Qt,QRect,QPoint

##from PyQt5.QtWidgets import  

from PyQt5.QtGui import (QPainter, QPen, QBrush,QPalette,
                     QLinearGradient,QGradient)

##from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.setPalette(QPalette(Qt.white))    #设置窗口为白色背景
      self.setAutoFillBackground(True)
      self.resize(300, 300)
      self.setWindowTitle("Demo8_3, 视口和窗口")


##  ==================event处理函数=================================
   def paintEvent(self,event):  ##在窗口上绘图
      painter=QPainter(self)
      painter.setRenderHint(QPainter.Antialiasing)
   ##      painter.setRenderHint(QPainter.TextAntialiasing)

      W=self.width()    #宽度
      H=self.height()   #绘图区高度

      side=min(W,H)     #取长和宽的小值
      rect=QRect((W-side)/2, (H-side)/2,side,side)    #viewport矩形区
      painter.drawRect(rect)  #Viewport大小
      
      painter.setViewport(rect)  #设置Viewport
      painter.setWindow(-100,-100,200,200)   # 设置窗口大小，逻辑坐标


   ##设置画笔
      pen=QPen()
      pen.setWidth(1)   #线宽
      pen.setColor(Qt.blue)      #划线颜色
      pen.setStyle(Qt.SolidLine) #线的类型，实线、虚线等
   ##      pen.setCapStyle(Qt.FlatCap)    #线端点样式
   ##      pen.setJoinStyle(Qt.BevelJoin) #线的连接点样式
      painter.setPen(pen)
      
   ####线性渐变
      linearGrad=QLinearGradient(0,0,100,0)  #从左到右
      linearGrad.setColorAt(0,Qt.yellow)     #起点颜色
      linearGrad.setColorAt(0.9,Qt.green)    #终点颜色
      linearGrad.setSpread(QGradient.ReflectSpread)   #展布模式
      painter.setBrush(linearGrad)

   ##径向渐变
   ##      radialGrad=QRadialGradient(50,0,50,50,0)
   ##      radialGrad.setColorAt(0,Qt.yellow)
   ##      radialGrad.setColorAt(0.8,Qt.green)
   ##      radialGrad.setSpread(QGradient.ReflectSpread)
   #### QGradient::PadSpread ,QGradient::RepeatSpread, QGradient::ReflectSpread
   ##      painter.setBrush(radialGrad)
      

   ##设置复合模式
##      painter.setCompositionMode(QPainter.RasterOp_NotSourceXorDestination)
      painter.setCompositionMode(QPainter.CompositionMode_Difference)
##      painter.setCompositionMode(QPainter.CompositionMode_Exclusion)
      
##      painter.setCompositionMode(QPainter.CompositionMode_ColorBurn)
##      painter.setCompositionMode(QPainter.RasterOp_SourceXorDestination)

   ##绘图
      for i in range(36):
         painter.drawEllipse(QPoint(50,0),50,50)
         painter.rotate(10)
        
##  ==========由connectSlotsByName() 自动连接的槽函数===============        
    
        
        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   form=QmyWidget()                 #创建窗体
   form.show()
   sys.exit(app.exec_())
