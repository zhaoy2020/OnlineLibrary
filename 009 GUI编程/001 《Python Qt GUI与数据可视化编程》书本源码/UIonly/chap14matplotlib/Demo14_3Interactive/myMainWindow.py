# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QLabel

from PyQt5.QtCore import  pyqtSlot,Qt

from PyQt5.QtChart import QChart,QLineSeries,QValueAxis

import numpy as np

import matplotlib as mpl

from  matplotlib.backends.backend_qt5agg import (FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar)

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo14_3, 交互操作")
      pass
   

##  ==============自定义功能函数========================
   def __createFigure(self):  ##创建绘图系统
      pass
      

   def __changeActionLanguage(self):
      pass


   def __drawScatters(self,N=15):
      pass

      
   def __drawFig1X2(self):  #初始化绘图
      pass


##  ==============event处理函数==========================


##  ==========由connectSlotsByName()自动连接的槽函数============        
   @pyqtSlot()   ## 紧凑布局
   def on_actTightLayout_triggered(self):
      self.__fig.tight_layout()     # 对所有子图 进行一次tight_layout
      self.__fig.canvas.draw()


   @pyqtSlot()   ## 设置鼠标光标
   def on_actSetCursor_triggered(self):
      self.__fig.canvas.setCursor(Qt.CrossCursor)

      
   @pyqtSlot()   ## 重新绘制散点图
   def on_actScatterAgain_triggered(self):
      self.__axScatter.clear()   #清除子图
      self.__drawScatters(N=15)
      self.__fig.canvas.draw()   #刷新
      

##  =================自定义槽函数==========
#event类型 matplotlib.backend_bases.MouseEvent
   def do_canvas_mouseMove(self,event):
      pass


## event类型：matplotlib.backend_bases.LocationEvent      
   def do_axes_mouseEnter(self,event):
      pass

      
   def do_axes_mouseLeave(self,event):
      pass


##event 类型： matplotlib.backend_bases.PickEvent
   def do_series_pick(self,event):
      pass

     
#event类型 matplotlib.backend_bases.MouseEvent
   def do_scrollZoom(self,event): #通过鼠标滚轮缩放
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
