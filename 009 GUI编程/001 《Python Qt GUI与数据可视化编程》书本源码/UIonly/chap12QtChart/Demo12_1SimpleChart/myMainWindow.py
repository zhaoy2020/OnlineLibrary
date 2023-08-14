# -*- coding: utf-8 -*-

import sys, math

from PyQt5.QtWidgets import  QApplication, QMainWindow

##from PyQt5.QtCore import  pyqtSlot,pyqtSignal,Qt

##from PyQt5.QtWidgets import  

##from PyQt5.QtGui import

##from PyQt5.QtSql import 

##from PyQt5.QtMultimedia import

##from PyQt5.QtMultimediaWidgets import

from PyQt5.QtChart import QChartView,QChart,QLineSeries,QValueAxis

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)
      self.setWindowTitle("Demo12_1, QChart基本绘图")
      self.resize(580,420)

#创建chart和chartView
      chart = QChart()   #创建 Chart
      chart.setTitle("简单函数曲线")

      chartView=QChartView(self)   #创建 ChartView
      chartView.setChart(chart)    #Chart添加到ChartView
      self.setCentralWidget(chartView)

#创建曲线序列
      series0 = QLineSeries()
      series1 = QLineSeries()
      series0.setName("Sin曲线")
      series1.setName("Cos曲线")
      chart.addSeries(series0)    #序列添加到图表
      chart.addSeries(series1)

#序列添加数值
      t=0
      intv=0.1
      pointCount=100
      for i in range(pointCount):
         y1=math.cos(t) 
         series0.append(t,y1)
         y2=1.5*math.sin(t+20)
         series1.append(t,y2)
         t=t+intv

##创建坐标轴
      axisX = QValueAxis()   #X 轴
      axisX.setRange(0, 10)  #设置坐标轴范围
      axisX.setTitleText("time(secs)") #标题
##    axisX.setLabelFormat("%.1f")     #标签格式
##    axisX.setTickCount(11)           #主分隔个数
##    axisX.setMinorTickCount(4)
##    axisX.setGridLineVisible(false)

      axisY = QValueAxis()   #Y 轴
      axisY.setRange(-2, 2)
      axisY.setTitleText("value")
##    axisY.setTickCount(5)
##    axisY.setMinorTickCount(4)
##    axisY.setLabelFormat("%.2f")     #标签格式
##    axisY.setGridLineVisible(false)

#为序列设置坐标轴
      chart.setAxisX(axisX, series0)   #为序列设置坐标轴
      chart.setAxisY(axisY, series0)

      chart.setAxisX(axisX, series1)   #为序列设置坐标轴
      chart.setAxisY(axisY, series1)
      

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
