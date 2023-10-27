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
      self.__buildStatusBar()    #创建状态栏QLabel组件

      self.chartView = QmyChartView(self)
      self.chartView.setRenderHint(QPainter.Antialiasing)
      self.chartView.setCursor(Qt.CrossCursor)  #设置鼠标指针为十字星
      self.setCentralWidget(self.chartView)     #填充满工作区
      self.chartView.mouseMove.connect(self.do_chartView_mouseMove)

      self.__createChart()    #创建图表
      self.__prepareData()    #为序列设置数据
      

##  ==============自定义功能函数========================
   def __buildStatusBar(self):
      self.__labChartXY = QLabel("Chart X=,  Y=  ")   #图表坐标
      self.__labChartXY.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.__labChartXY)

      self.__labHoverXY = QLabel("Hovered X=,  Y=  ") #序列hover点坐标
      self.__labHoverXY.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.__labHoverXY)

      self.__labClickXY = QLabel("Clicked X=,  Y=  ") #序列click点坐标
      self.__labClickXY.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.__labClickXY)
      

   def __createChart(self):  ##创建图表
      chart = QChart()  #创建 Chart
   ##      chart.setTitle("简单函数曲线")
      chart.legend().setVisible(True)
      self.chartView.setChart(chart)   #Chart添加到chartView

      pen=QPen()
      pen.setWidth(2)

   ##========LineSeries折线 和 ScatterSeries散点
      seriesLine = QLineSeries()
      seriesLine.setName("LineSeries折线")
      seriesLine.setPointsVisible(False)     #数据点不可见
      pen.setColor(Qt.red)
      seriesLine.setPen(pen)
      seriesLine.hovered.connect(self.do_series_hovered)    #信号 hovered
      seriesLine.clicked.connect(self.do_series_clicked)    #信号 clicked
      chart.addSeries(seriesLine)   #添加到chart

      seriesLinePoint = QScatterSeries()    #散点序列
      seriesLinePoint.setName("ScatterSeries散点")
      shape=QScatterSeries.MarkerShapeCircle  #MarkerShapeRectangle
      seriesLinePoint.setMarkerShape(shape)   #散点形状，只有2种
      seriesLinePoint.setBorderColor(Qt.yellow)
      seriesLinePoint.setBrush(QBrush(Qt.red))
      seriesLinePoint.setMarkerSize(10)      #散点大小
      seriesLinePoint.hovered.connect(self.do_series_hovered)  #信号 hovered
      seriesLinePoint.clicked.connect(self.do_series_clicked)  #信号 clicked
      chart.addSeries(seriesLinePoint)    #添加到chart

   ##======== SplineSeries 曲线和 QScatterSeries 散点
      seriesSpLine = QSplineSeries()
      seriesSpLine.setName("SplineSeries曲线")
      seriesSpLine.setPointsVisible(False)    #数据点不可见
      pen.setColor(Qt.blue)
      seriesSpLine.setPen(pen)
      seriesSpLine.hovered.connect(self.do_series_hovered)  #信号 hovered
      seriesSpLine.clicked.connect(self.do_series_clicked)  #信号 clicked


      seriesSpPoint = QScatterSeries()    #散点序列
      seriesSpPoint.setName("QScatterSeries")
      shape=QScatterSeries.MarkerShapeRectangle  #MarkerShapeCircle
      seriesSpPoint.setMarkerShape(shape) #散点形状，只有2种
      seriesSpPoint.setBorderColor(Qt.green)
      seriesSpPoint.setBrush(QBrush(Qt.blue))
      seriesSpPoint.setMarkerSize(10)     #散点大小
      seriesSpPoint.hovered.connect(self.do_series_hovered) #信号 hovered
      seriesSpPoint.clicked.connect(self.do_series_clicked) #信号 clicked

      chart.addSeries(seriesSpLine)
      chart.addSeries(seriesSpPoint)

   ##  创建缺省坐标轴
      chart.createDefaultAxes()  #创建缺省坐标轴并与序列关联
      chart.axisX().setTitleText("time(secs)")
      chart.axisX().setRange(0,10)
      chart.axisX().applyNiceNumbers()

      chart.axisY().setTitleText("value")
      chart.axisY().setRange(-2,2)
      chart.axisY().applyNiceNumbers()

      for marker in chart.legend().markers():  #QLegendMarker类型列表
         marker.clicked.connect(self.do_LegendMarkerClicked)


   def __prepareData(self):  ##为序列设置数据
      series0=self.chartView.chart().series()[0]  # QLineSeries
      series1=self.chartView.chart().series()[1]  # QScatterSeries

      series2=self.chartView.chart().series()[2]  # QSplineSeries
      series3=self.chartView.chart().series()[3]  # QScatterSeries

      series0.clear()
      series1.clear()
      series2.clear()
      series3.clear()

      t=0
      intv=0.5
      pointCount=20  #数据点个数较少，比较QLineSeries和QSplineSeries的差别
      for i in range(pointCount):
        rd=random.randint(-5,5)  #随机数,-5~+5
        y1=math.sin(2*t)+rd/50.0  
        series0.append(t,y1)     # QLineSeries
        series1.append(t,y1)     # QScatterSeries

        rd=random.randint(-5,5)  #随机数,-5~+5
        y2=1.5*math.sin(2*t+20)+rd/50.0
        series2.append(t,y2)     # QSplineSeries
        series3.append(t,y2)     # QScatterSeries

        t=t+intv

##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
   @pyqtSlot()  ##放大
   def on_actZoomIn_triggered(self):
      self.chartView.chart().zoom(1.2)
      
   @pyqtSlot()  ##缩小
   def on_actZoomOut_triggered(self):
      self.chartView.chart().zoom(0.8)

   @pyqtSlot()  ##复位原始大小
   def on_actZoomReset_triggered(self):
      self.chartView.chart().zoomReset()
      
           
##  =============自定义槽函数===============================        
   def do_LegendMarkerClicked(self):  ##点击图例小方块
      marker =self.sender()   #QLegendMarker类型
      if (marker.type() != QLegendMarker.LegendMarkerTypeXY):
         return 
      
      marker.series().setVisible(not marker.series().isVisible())
      marker.setVisible(True)
      alpha = 1.0
      if not marker.series().isVisible():
         alpha = 0.5

      brush = marker.labelBrush()  # QBrush
      color = brush.color()        # QColor
      color.setAlphaF(alpha)
      brush.setColor(color)
      marker.setLabelBrush(brush)

      brush = marker.brush()
      color = brush.color()
      color.setAlphaF(alpha)
      brush.setColor(color)
      marker.setBrush(brush)

      pen = marker.pen()   #QPen
      color = pen.color()
      color.setAlphaF(alpha)
      pen.setColor(color)
      marker.setPen(pen)

   def do_chartView_mouseMove(self,point):   ##鼠标移动
      pt=self.chartView.chart().mapToValue(point)  #QPointF 转换为图表的数值
      hint="Chart X=%.2f,Y=%.2f"%(pt.x(),pt.y())
      self.__labChartXY.setText(hint)  #状态栏显示

   def do_series_hovered(self,point,state):  ##序列的hovered信号
      if state:
         hint="Hovered X=%.2f,Y=%.2f"%(point.x(),point.y())
         self.__labHoverXY.setText(hint)
      else:
         self.__labHoverXY.setText("Series X=, Y=")

   def do_series_clicked(self,point):     ##序列的click信号
      hint="Clicked X=%.2f,Y=%.2f"%(point.x(),point.y())
      self.__labClickXY.setText(hint)

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
