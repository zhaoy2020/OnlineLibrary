import sys

from PyQt5.QtWidgets import  QApplication, QWidget

from PyQt5.QtCore import  Qt,QRect,QLine,QPoint,QRectF

##from PyQt5.QtWidgets import  

from PyQt5.QtGui import (QPainter, QPen, QBrush,QPalette,QFont,
               QImage,QPainterPath, QPolygon,QPixmap,
               QRadialGradient,QGradient,QLinearGradient,QConicalGradient)

##from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)      #调用父类构造函数，创建窗体
      
      self.setPalette(QPalette(Qt.white))    #设置窗口为白色背景
      self.setAutoFillBackground(True)
      self.resize(300, 200)
      self.setWindowTitle("Demo8_1, QPainter基本绘图")


##  ==================event处理函数=================================
   def paintEvent(self,event):  #在窗口上绘图
      painter=QPainter(self)
      painter.setRenderHint(QPainter.Antialiasing)
      painter.setRenderHint(QPainter.TextAntialiasing)

   #设置画笔
      pen=QPen()
      pen.setWidth(2)   #线宽
      pen.setColor(Qt.black)   #划线颜色
   ## Qt::NoPen,Qt::SolidLine, Qt::DashLine, Qt::DotLine,Qt::DashDotLine,Qt::DashDotDotLine,Qt::CustomDashLine
      pen.setStyle(Qt.SolidLine) #线的类型，实线、虚线等
   ## Qt::FlatCap, Qt::SquareCap,Qt::RoundCap
      pen.setCapStyle(Qt.RoundCap)   #线端点样式
   ## Qt::MiterJoin,Qt::BevelJoin,Qt::RoundJoin,Qt::SvgMiterJoin
      pen.setJoinStyle(Qt.RoundJoin) #线的连接点样式
      painter.setPen(pen)
      
   ##设置画刷
   ##      QPixmap texturePixmap(":images/images/texture.jpg");
##      texturePixmap=QPixmap("texture2.jpg")
##      brush=QBrush()
##      brush.setColor(Qt.black)     #画刷颜色
## Qt::SolidPattern, Dense6Pattern, HorPattern, VerPattern,CrossPattern,BDiagPattern
##      brush.setStyle(Qt.BDiagPattern)  #填充样式
##      brush.setStyle(Qt.TexturePattern)   #画刷填充样式
##      brush.setTexture(texturePixmap)  #设置材质图片
##      painter.setBrush(brush)


## 设置字体

      W=self.width()    #绘图区宽度
      H=self.height()   #绘图区高度
      
##      font=painter.font()
##      font.setPointSize(25)
##      font.setBold(True)
##      painter.setFont(font)

##      rect=QRect(W/4,H/4,W/2,H/2)
##      painter.fillRect (rect,Qt.green)
##      painter.drawText(rect,Qt.AlignCenter, "Hello,Qt")

##      rect=QRectF(W/4,H/4,W/2,H/2)
##      path=QPainterPath()
##      path.addEllipse(rect)
##      path.addRect(rect)
##      painter.fillPath(path,Qt.red)
      
##      painter.drawRoundedRect(rect,20,20)
##      painter.drawRect(rect)
##      painter.eraseRect(rect)
      
##   ##      painter.drawEllipse(rect)
##
##      pixmap=QPixmap("qt.jpg")
##      painter.drawPixmap(rect, pixmap)
      
##      painter.drawPoint(QPoint(W/2,H/2))

##      points=[QPoint(5*W/12,H/4),
##              QPoint(3*W/4,5*H/12),
##              QPoint(2*W/4,5*H/12) ]
##      painter.drawPoints(QPolygon(points))

      points=[ QPoint(5*W/12,H/4),
               QPoint(3*W/4,5*H/12),
               QPoint(5*W/12,3*H/4),
               QPoint(2*W/4,5*H/12) ]
      painter.drawPolyline(QPolygon(points))
      painter.drawPolygon(QPolygon(points))
      
   
##      image=QImage("qt.jpg")
##      painter.drawImage(rect, image)
##      Line=QLine(W/4,H/4,W/2,H/2)
##      painter.drawLine(Line)


      
##      Lines=[ QLine(rect.topLeft(),rect.bottomRight()),
##              QLine(rect.topRight(),rect.bottomLeft()),
##              QLine(rect.topLeft(),rect.bottomLeft()),
##              QLine(rect.topRight(),rect.bottomRight()) ]
##      for line in Lines:
##         painter.drawLine(line)
      
##      rect=QRectF(W/4,H/4,W/2,H/2)  #不能用QRect
##      path=QPainterPath()
##      path.addEllipse(rect)
##      path.addRect(rect)
##      painter.drawPath(path)


##      rect=QRect(W/4,H/4,W/2,H/2)
##      startAngle = 40 * 16   #起始 40°
##      spanAngle = 120 * 16   #旋转 120°
##      painter.drawPie(rect, startAngle, spanAngle)
##      painter.drawChord(rect, startAngle, spanAngle)
##      painter.drawArc(rect, startAngle, spanAngle)
      

##      points=[ QPoint(5*W/12,H/4),
##                  QPoint(3*W/4,5*H/12),
##                  QPoint(5*W/12,3*H/4),
##                  QPoint(W/4,5*H/12)  ]
##      painter.drawConvexPolygon(QPolygon(points))
####      painter.drawConvexPolygon(points,4)     #错误，不能用

   ##      points=[QPoint(5*W/12,H/4),
   ##              QPoint(3*W/4,5*H/12),
   ##              QPoint(4*W/12,5*H/7)]
   ####              QPoint(2*W/4,5*H/12) ]      #列表数据
   ##      
   ##      painter.drawPolyline(QPolygon(points))
   
      
   #### 设置字体
   ##      font=QFont()
   ##      font.setPointSize(2)
   ##      font.setBold(True)
   ##      painter.setFont(font)
   ##
   ####绘图
   ####      Line=QLine(W/8,H/2,7*W/8,H/2)
   ####      painter.drawLine(Line)
   ######      painter.drawText(W/8,H/2+25,"Qt.DashDotDotLine")
   ##
   ##
##      rect=QRect(W/4,H/4,W/2,H/2)
##      painter.drawRect(self.rect())    #只填充定义的渐变区域
##      painter.drawRect(rect)    #只填充定义的渐变区域
   ####      startAngle = 45 * 16  #起始90°
   ####      spanAngle = 135 * 16  #旋转90°
   ####      painter.drawArc(rect, startAngle, spanAngle)
   ##
   ##      points=[QPoint(5*W/12,H/4),
   ##              QPoint(3*W/4,5*H/12),
   ##              QPoint(4*W/12,5*H/7)]
   ####              QPoint(2*W/4,5*H/12) ]      #列表数据
   ##      
   ##      painter.drawPolyline(QPolygon(points))
   ####      painter.drawPolyline(points)  #错误，必须使用QPolygon()
      
      
##  =================自定义功能函数=================================
##   def __draw01_filledRect(self):#绘制一个填充矩形
##      painter=QPainter(self)
##      painter.setRenderHint(QPainter.Antialiasing)
##      painter.setRenderHint(QPainter.TextAntialiasing)
##
##      W=self.width()    #绘图区宽度
##      H=self.height()   #绘图区高度
##      rect=QRect(W/4,H/4,W/2,H/2)   #中间区域矩形框
##
###设置画笔
##      pen=QPen()
##      pen.setWidth(2)   #线宽
##      pen.setColor(Qt.red)   #划线颜色
#### Qt::NoPen,Qt::SolidLine, Qt::DashLine, Qt::DotLine,Qt::DashDotLine,Qt::DashDotDotLine,Qt::CustomDashLine
##      pen.setStyle(Qt.SolidLine) #线的类型，实线、虚线等
#### Qt::FlatCap, Qt::SquareCap,Qt::RoundCap
##      pen.setCapStyle(Qt.SquareCap)   #线端点样式
#### Qt::MiterJoin,Qt::BevelJoin,Qt::RoundJoin,Qt::SvgMiterJoin
##      pen.setJoinStyle(Qt.BevelJoin) #线的连接点样式
##      painter.setPen(pen)
##      
####设置画刷
####    QPixmap texturePixmap(":images/images/texture.jpg");
##      brush=QBrush()
##      brush.setColor(Qt.yellow)     #画刷颜色
##      brush.setStyle(Qt.SolidPattern)  #填充样式
####    brush.setStyle(Qt.TexturePattern)   #画刷填充样式
####    brush.setTexture(texturePixmap)  #设置材质图片
##      painter.setBrush(brush)
##
####绘图
##      painter.drawRect(rect)    #只填充定义的渐变区域


   def __draw02_textureBrush(self):  #在窗口上绘图
      painter=QPainter(self)
      painter.setRenderHint(QPainter.Antialiasing)
      painter.setRenderHint(QPainter.TextAntialiasing)

#设置画笔
      pen=QPen()
      pen.setWidth(3)   #线宽
      pen.setColor(Qt.red)   #划线颜色
      
## Qt::NoPen,Qt::SolidLine, Qt::DashLine, Qt::DotLine,Qt::DashDotLine,Qt::DashDotDotLine,Qt::CustomDashLine
      pen.setStyle(Qt.SolidLine) #线的类型，实线、虚线等
      
## Qt::FlatCap, Qt::SquareCap,Qt::RoundCap
      pen.setCapStyle(Qt.RoundCap)   #线端点样式
      
## Qt::MiterJoin,Qt::BevelJoin,Qt::RoundJoin,Qt::SvgMiterJoin
      pen.setJoinStyle(Qt.RoundJoin) #线的连接点样式
      
      painter.setPen(pen)
      
##设置画刷
      texturePixmap=QPixmap("texture.jpg")
      brush=QBrush()
##      brush.setColor(Qt.yellow)     #画刷颜色
##      brush.setStyle(Qt.SolidPattern)  #填充样式
      brush.setStyle(Qt.TexturePattern)   #画刷填充样式
      brush.setTexture(texturePixmap)  #设置材质图片
      painter.setBrush(brush)

##绘图
      W=self.width()    #绘图区宽度
      H=self.height()   #绘图区高度

      rect=QRect(W/4,H/4,W/2,H/2)
      painter.drawRect(rect)

     
   def __draw03_gradient(self):  #在窗口上绘图
      painter=QPainter(self)
   ##      painter.setRenderHint(QPainter.Antialiasing)
   ##      painter.setRenderHint(QPainter.TextAntialiasing)

   #设置画笔
      pen=QPen()
   ##      pen.setWidth(2)   #线宽
   ##      pen.setColor(Qt.red)   #划线颜色
      pen.setStyle(Qt.NoPen) #线的类型，实线、虚线等
      
      painter.setPen(pen)

      W=self.width()    #绘图区宽度
      H=self.height()   #绘图区高度

   ##      radialGrad=QRadialGradient(W/4,H/2,W/10,W/5,H/2)
   ##      radialGrad=QRadialGradient(W/2,H/2,W/2,3*W/4,H/2)    #原点就是焦点
      radialGrad=QRadialGradient(W/2,H/2,W/8,W/2,H/2)
      radialGrad.setColorAt(0,Qt.yellow)  #yellow
      radialGrad.setColorAt(1,Qt.blue)   #green
   ## PadSpread,  RepeatSpread,  ReflectSpread
      radialGrad.setSpread(QGradient.ReflectSpread)
      painter.setBrush(radialGrad)

   ##      rect=QRect(W/4,H/4,W/2,H/2)

   ####      linearGrad=QLinearGradient(rect.left(), rect.top(), rect.right(), rect.top())#从左到右
   ##      linearGrad=QLinearGradient(rect.left(), rect.top(), rect.right(), rect.bottom())#从左到右
   ##
   ##      linearGrad.setColorAt(0,Qt.blue) #起点颜色
   ##      linearGrad.setColorAt(0.5,Qt.white)    #中间点颜色
   ##      linearGrad.setColorAt(1,Qt.blue)     #终点颜色
   ##   ##      linearGrad.setSpread(QGradient.ReflectSpread)   #展布模式
   ##      painter.setBrush(linearGrad)

   ##      coniGrad=QConicalGradient(W/2, H/2, 0)
   ##      coniGrad.setColorAt(0,     Qt.yellow)
   ##      coniGrad.setColorAt(0.5,   Qt.blue)
   ##      coniGrad.setColorAt(1,     Qt.green)  #green
   ##      painter.setBrush(coniGrad)
      
      painter.drawRect(self.rect())   #填充整个窗口
   ##      painter.drawEllipse(rect)    #只填充定义的渐变区域
   ##      painter.drawRect(rect)    #只填充定义的渐变区域
        
##  ==========由connectSlotsByName() 自动连接的槽函数===============        
    
        
        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyWidget()            #创建窗体
   form.show()
   sys.exit(app.exec_())
