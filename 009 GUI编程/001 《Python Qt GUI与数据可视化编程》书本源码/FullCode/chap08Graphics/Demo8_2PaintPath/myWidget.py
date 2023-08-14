import sys, math

from PyQt5.QtWidgets import  QApplication, QWidget

from PyQt5.QtCore import  Qt,QRect,QPoint

##from PyQt5.QtWidgets import  

from PyQt5.QtGui import (QPainter, QPen, QBrush,QPalette,
                        QFont,QPainterPath)

##from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)      #调用父类构造函数，创建窗体
      
      self.setPalette(QPalette(Qt.white))    #设置窗口为白色背景
      self.setAutoFillBackground(True)
      self.resize(600, 300)
      self.setWindowTitle("Demo8_2, QPainter坐标变换")


##  ==================event处理函数=================================
   def paintEvent(self,event):  #在窗口上绘图
      painter=QPainter(self)
      painter.setRenderHint(QPainter.Antialiasing)
      painter.setRenderHint(QPainter.TextAntialiasing)


##生成五角星的5个顶点的,假设原点在五角星中心
      R=100.0        #半径
      Pi=3.14159     #常数
      deg=Pi*72.0/180   
      points=[QPoint(R,0),
              QPoint(R*math.cos(deg),    -R*math.sin(deg)),
              QPoint(R*math.cos(2*deg),  -R*math.sin(2*deg)),
              QPoint(R*math.cos(3*deg),  -R*math.sin(3*deg)),
              QPoint(R*math.cos(4*deg),  -R*math.sin(4*deg)) ]   #元组数据

      font=painter.font()
      font.setPointSize(14)
      font.setBold(False)
      painter.setFont(font)

   #设置画笔
      pen=QPen()
      pen.setWidth(2)   #线宽
      pen.setColor(Qt.blue)      #划线颜色
      pen.setStyle(Qt.SolidLine) #线的类型，实线、虚线等
      pen.setCapStyle(Qt.FlatCap)      #线端点样式
      pen.setJoinStyle(Qt.BevelJoin)   #线的连接点样式
      painter.setPen(pen)
      
##设置画刷
      brush=QBrush()
      brush.setColor(Qt.yellow)        #画刷颜色
      brush.setStyle(Qt.SolidPattern)  #填充样式
      painter.setBrush(brush)

##设计绘制五角星的PainterPath，以便重复使用
      starPath=QPainterPath()
      starPath.moveTo(points[0])
      starPath.lineTo(points[2])
      starPath.lineTo(points[4])
      starPath.lineTo(points[1])
      starPath.lineTo(points[3])
      starPath.closeSubpath()    #闭合路径，最后一个点与第一个点相连

      
      starPath.addText(points[0],font,"0") #显示端点编号
      starPath.addText(points[1],font,"1")
      starPath.addText(points[2],font,"2")
      starPath.addText(points[3],font,"3")
      starPath.addText(points[4],font,"4")

##绘图
      painter.save()    #保存坐标状态
      painter.translate(100,120)
      painter.drawPath(starPath)    #画星星
      painter.drawText(0,0,"S1")
      painter.restore()    #恢复坐标状态

      painter.translate(300,120)    #平移
      painter.scale(0.8,0.8)        #缩放
      painter.rotate(90)         #顺时针旋转
      painter.drawPath(starPath) #画星星
      painter.drawText(0,0,"S2")

      painter.resetTransform()      #复位所有坐标变换
      painter.translate(500,120)    #平移
      painter.rotate(-145)          #逆时针旋转
      painter.drawPath(starPath)    #画星星
      painter.drawText(0,0,"S3")    

        
##  ==========由connectSlotsByName() 自动连接的槽函数===============        
    
        
        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   form=QmyWidget()                 #创建窗体
   form.show()
   sys.exit(app.exec_())
