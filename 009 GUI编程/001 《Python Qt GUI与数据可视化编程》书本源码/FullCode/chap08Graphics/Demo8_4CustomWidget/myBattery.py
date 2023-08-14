# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import   QApplication, QWidget

from PyQt5.QtCore import  pyqtSignal,Qt,QRect

##from PyQt5.QtWidgets import  

from PyQt5.QtGui import QPainter,QPen, QBrush,QFontMetrics

##from PyQt5.QtSql import 

##from PyQt5.QtMultimedia import

##from PyQt5.QtMultimediaWidgets import


class QmyBattery(QWidget):

   powerLevelChanged=pyqtSignal(int)  ##电量变化时发射的信号

   def __init__(self, parent=None):
      super().__init__(parent)

      self.colorBack=Qt.white      #背景颜色
      self.colorBorder=Qt.black    #电池边框颜色
      self.colorPower=Qt.green     #电量柱颜色
      self.colorWarning=Qt.red     #电量短缺时的颜色

      self.__powerLevel=65       #电量0-100
      self.__warnLevel=20        #电量低警示阈值
      
##  ===========接口函数====================
   def setPowerLevel(self,power):   ##设置电量
      self.__powerLevel=power
      self.powerLevelChanged.emit(power)  #发射信号
      self.repaint()  #重绘
      
   def powerLevel(self):   #返回电量
      return self.__powerLevel

   def setWarnLevel(self,warn):   #设置低电量阈值
      self.__warnLevel=warn
      self.repaint()  #重绘
      
   def warnLevel(self):   #返回低电量阈值
      return self.__warnLevel
   

##  ==============自定义功能函数========================


##  ==============event处理函数==========================
   def paintEvent(self,event):  ##在窗口上绘图
      painter=QPainter(self)
      painter.setRenderHint(QPainter.Antialiasing)
      painter.setRenderHint(QPainter.TextAntialiasing)

      rect=QRect(0,0,self.width(),self.height()) #viewport矩形区
      painter.setViewport(rect)     #设置Viewport
      painter.setWindow(0,0,120,50) #设置窗口大小，逻辑坐标

##绘制电池边框
      pen=QPen()
      pen.setWidth(2)   #线宽
      pen.setColor(self.colorBorder)    #划线颜色
      pen.setStyle(Qt.SolidLine)    #线的类型，实线、虚线等
      pen.setCapStyle(Qt.FlatCap)   #线端点样式
      pen.setJoinStyle(Qt.BevelJoin)#线的连接点样式
      painter.setPen(pen)


      brush=QBrush()  #设置画刷
      brush.setColor(self.colorBack)   #画刷颜色
      brush.setStyle(Qt.SolidPattern)  #画刷填充样式
      painter.setBrush(brush)

      rect.setRect(1,1,109,48)
      painter.drawRect(rect)    #绘制电池边框

      brush.setColor(self.colorBorder) #画刷颜色
      painter.setBrush(brush)
      rect.setRect(110,15,10,20)
      painter.drawRect(rect)    #画电池正极头

##画电池柱
      if self.__powerLevel>self.__warnLevel: #正常颜色电量柱
         brush.setColor(self.colorPower)     #画刷颜色
         pen.setColor(self.colorPower)       #划线颜色
      else:   ##电量低电量柱
         brush.setColor(self.colorWarning)   #画刷颜色
         pen.setColor(self.colorWarning)     #划线颜色
         
      painter.setBrush(brush)
      painter.setPen(pen)

      if self.__powerLevel>0:
         rect.setRect(5,5,self.__powerLevel,40)
         painter.drawRect(rect)   #画电池柱

##绘制电量百分比文字
      textSize=QFontMetrics(self.font())
      powStr="%d%%"%self.__powerLevel
      textRect=QRect(textSize.boundingRect(powStr))  #得到字符串的rect

      painter.setFont(self.font())
      pen.setColor(self.colorBorder)   #划线颜色
      painter.setPen(pen)

      painter.drawText(55-textRect.width()/2,
               23+textRect.height()/2,powStr)
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
        
        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序

   form=QmyBattery()               #创建窗体
   form.show()

   sys.exit(app.exec_())
