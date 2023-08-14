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
      pass

      
##  ===========接口函数====================
   def setPowerLevel(self,power):   ##设置电量
      pass
      
   def powerLevel(self):   ##返回电量
      pass

   def setWarnLevel(self,warn):   ##设置低电量阈值
      pass
      
   def warnLevel(self):   ##返回低电量阈值
      pass
   

##  ==============自定义功能函数========================


##  ==============event处理函数==========================
   def paintEvent(self,event):  ##在窗口上绘图
      pass
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
        
        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyBattery()               #创建窗体
   form.show()
   sys.exit(app.exec_())
