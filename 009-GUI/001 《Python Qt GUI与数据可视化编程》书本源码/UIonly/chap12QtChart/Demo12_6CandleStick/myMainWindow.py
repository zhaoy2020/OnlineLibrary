# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QLabel,QFileDialog,QMessageBox

from PyQt5.QtCore import  Qt,pyqtSlot,QDir,QFileInfo,QDateTime,QDate 

from PyQt5.QtGui import QPen,QPainter,QBrush,QStandardItemModel,QStandardItem

from PyQt5.QtChart import (QChart,QLineSeries,QValueAxis,QCandlestickSet,
                     QLegendMarker, QDateTimeAxis,QCandlestickSeries)

from ui_MainWindow import Ui_MainWindow


class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面

      self.setWindowTitle("Demo12_6, 蜡烛图、日期时间坐标轴")
      pass
      

##  ==============自定义功能函数========================
   def __buildStatusBar(self):
      self.__labChartXY = QLabel("Chart Y= ")    ##状态栏显示鼠标点的坐标
      self.__labChartXY.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.__labChartXY)

      self.__labHoverXY = QLabel("Hovered candle")   
      self.__labHoverXY.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.__labHoverXY)
      
      self.__labClickXY = QLabel("Clicked candle")   
##      self.__labClickXY.setMinimumWidth(200)
      self.ui.statusBar.addPermanentWidget(self.__labClickXY)


   def __iniChart(self):
      pass


   def __loadData(self,allLines):   ##从字符串列表读取数据构建数据模型
      pass

         
   def __drawChart(self):  ##绘制图表
      pass


##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
   @pyqtSlot()   ##“打开文件”按钮
   def on_actOpen_triggered(self):
      curPath=QDir.currentPath()  
      filename,flt=QFileDialog.getOpenFileName(self,"打开一个文件",curPath,
                  "股票数据文件(*.txt);;所有文件(*.*)")
      if (filename==""):
         return

      aFile=open(filename,'r')
      allLines=aFile.readlines()  #读取所有行,list类型,每行末尾带有 \n
      aFile.close()
      fileInfo=QFileInfo(filename)
      QDir.setCurrent(fileInfo.absolutePath())
      self.ui.tabWidget.setTabText(0,fileInfo.baseName())
      
      self.__loadData(allLines)  # 载入数据到数据模型
      self.__drawChart()         # 绘制图表
      self.ui.tab_Setup.setEnabled(True)

   @pyqtSlot()
   def on_actZoomIn_triggered(self):
      self.ui.chartView.chart().zoom(1.2)
      
   @pyqtSlot()
   def on_actZoomOut_triggered(self):
      self.ui.chartView.chart().zoom(0.8)

   @pyqtSlot()
   def on_actZoomReset_triggered(self):
      self.ui.chartView.chart().zoomReset()
      
##图表外观控制
   @pyqtSlot(int)    ##主题
   def on_comboTheme_currentIndexChanged(self,index):
      self.ui.chartView.chart().setTheme(QChart.ChartTheme(index))
      
   @pyqtSlot(bool)   ##显示图例
   def on_chkBox_Legend_clicked(self,checked):
      self.ui.chartView.chart().legend().setVisible(checked)
      
   ## 蜡烛图     
   @pyqtSlot(bool)   ##capsVisible
   def on_chkBox_Caps_clicked(self,checked):
      pass

   @pyqtSlot(bool)   ##bodyOutlineVisible
   def on_chkBox_Outline_clicked(self,checked):
      pass

   
## Y轴--QValueAxis 
   @pyqtSlot()    ##设置坐标范围
   def on_btnY_SetRange_clicked(self):
      self.__axisY.setRange(self.ui.spinY_Min.value(),
                              self.ui.spinY_Max.value())
   
   @pyqtSlot(int)  ##分度数
   def on_spinY_Ticks_valueChanged(self,arg1):
      self.__axisY.setTickCount(arg1)


   ## X轴--QDateTimeAxis
   @pyqtSlot(str)    ##标签格式
   def on_comboDateFormat_currentIndexChanged(self,arg1):
      self.__axisX.setFormat(arg1)
      
   @pyqtSlot(int)    ##分度数
   def on_btnX_Ticks_valueChanged(self,arg1):
      self.__axisX.setTickCount(arg1)

           
##  =============自定义槽函数===============================        
   def do_LegendMarkerClicked(self):
      marker =self.sender()   #QLegendMarker
      
      marker.series().setVisible(not marker.series().isVisible())
      marker.setVisible(True)
      alpha = 1.0
      if not marker.series().isVisible():
         alpha = 0.5

      brush = marker.labelBrush()   #QBrush
      color = brush.color()         #QColor
      color.setAlphaF(alpha)
      brush.setColor(color)
      marker.setLabelBrush(brush)

      brush = marker.brush()
      color = brush.color()
      color.setAlphaF(alpha)
      brush.setColor(color)
      marker.setBrush(brush)

      pen = marker.pen()  #QPen
      color = pen.color()
      color.setAlphaF(alpha)
      pen.setColor(color)
      marker.setPen(pen)

      if marker.type()==QLegendMarker.LegendMarkerTypeCandlestick:
         QMessageBox.information(self,"提示","蜡烛图序列无法隐藏")

   def do_chartView_mouseMove(self,point):
      pt=self.chart.mapToValue(point)  #QPointF 转换为图表的数值
      self.__labChartXY.setText("Chart Y=%.2f"%(pt.y()))    #状态栏显示


   def do_candleClicked(self,dataSet):
      pass

         
   def do_candleHovered(self,status,dataSet):
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
