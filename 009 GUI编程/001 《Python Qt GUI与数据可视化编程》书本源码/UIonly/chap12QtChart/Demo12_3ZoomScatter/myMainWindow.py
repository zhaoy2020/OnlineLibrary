# -*- coding: utf-8 -*-

import sys, math,random

from PyQt5.QtWidgets import  QApplication, QMainWindow,QWidget,QLabel

from PyQt5.QtCore import  Qt,pyqtSlot

from PyQt5.QtGui import QPen,QPainter,QBrush,QColor

from PyQt5.QtChart import (QChart,QLineSeries,QSplineSeries,
                     QScatterSeries,QValueAxis,QLegendMarker)

from ui_MainWindow import Ui_MainWindow

from myChartView import QmyChartView

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo12_3, QScatterSeries、QSplineSeries、自定义QChartView")
      pass
      

##  ==============自定义功能函数========================
   def __buildStatusBar(self):
      pass
      
   def __createChart(self):  ##创建图表
      pass

   def __prepareData(self):  ##为序列设置数据
      pass


##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
   @pyqtSlot()  ##放大
   def on_actZoomIn_triggered(self):
      pass
      
   @pyqtSlot()  ##缩小
   def on_actZoomOut_triggered(self):
      pass

   @pyqtSlot()  ##复位原始大小
   def on_actZoomReset_triggered(self):
      pass
      
           
##  =============自定义槽函数===============================        
   def do_LegendMarkerClicked(self):  ##点击图例小方块
      pass

   def do_chartView_mouseMove(self,point):   ##鼠标移动
      pass

   def do_series_hovered(self,point,state):  ##序列的hovered信号
      pass

   def do_series_clicked(self,point):     ##序列的click信号
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
