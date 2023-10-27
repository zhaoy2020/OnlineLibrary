# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QLabel,QFileDialog

from PyQt5.QtCore import  Qt,pyqtSlot,QDir,QFileInfo

from PyQt5.QtGui import QPen,QPainter

from PyQt5.QtChart import QChart,QLineSeries,QValueAxis,QAreaSeries

from ui_MainWindow import Ui_MainWindow

##from myChartView import QmyChartView

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo12_7, QAreaSeries绘制填充图")
      pass
      

##  ==============自定义功能函数========================
   def __buildStatusBar(self):
      self.__labFileName = QLabel("数据文件")     
      self.__labFileName.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.__labFileName)

      self.__labChartXY = QLabel("Chart, X=, Y=") 
      self.__labChartXY.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.__labChartXY)
      
      self.__labAreaXY = QLabel("AreaSeries，X=, Y=") 
      self.__labAreaXY.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.__labAreaXY)

   def __iniChart(self):
      pass

##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
##=====工具栏 按钮功能
   @pyqtSlot()    ##“打开文件”按钮
   def on_actOpen_triggered(self):
      pass


   @pyqtSlot()
   def on_actZoomIn_triggered(self):
      self.ui.chartView.chart().zoom(1.2)
      
   @pyqtSlot()
   def on_actZoomOut_triggered(self):
      self.ui.chartView.chart().zoom(0.8)

   @pyqtSlot()
   def on_actZoomReset_triggered(self):
      self.ui.chartView.chart().zoomReset()
      
##   Y轴设置      
   @pyqtSlot()    ##设置坐标范围
   def on_btnY_SetRange_clicked(self):
      self.__axisY.setRange(self.ui.spinY_Min.value(),
                            self.ui.spinY_Max.value())
   @pyqtSlot(int)    ##分度数
   def on_spinY_Ticks_valueChanged(self,arg1):
      self.__axisY.setTickCount(arg1)

   @pyqtSlot(bool)  ##显示网格线
   def on_chkBoxY_GridLine_clicked(self,checked):
      self.__axisY.setGridLineVisible(checked)
      
##    X轴设置
   @pyqtSlot()       ##设置坐标范围
   def on_btnX_SetRange_clicked(self):
      self.__axisX.setRange(self.ui.spinX_Min.value(),
                            self.ui.spinX_Max.value())
   @pyqtSlot(int)    ##分度数
   def on_spinX_Ticks_valueChanged(self,arg1):
      self.__axisX.setTickCount(arg1)

   @pyqtSlot(bool)   ##显示网格线
   def on_chkBoxX_GridLine_clicked(self,checked):
      self.__axisX.setGridLineVisible(checked)
      
           
##  =============自定义槽函数===============================        
   @pyqtSlot()    ##绘制原始波形曲线
   def do_redrawWave(self):
      pass


   @pyqtSlot()    ##绘制3种填充曲线
   def do_redrawFill(self):
      pass


   def do_chartView_mouseMove(self,point):
      pt=self.ui.chartView.chart().mapToValue(point)  #QPointF 转换为图表的数值
      self.__labChartXY.setText("Chart X=%.2f,Y=%.2f"%(pt.x(),pt.y()))  #状态栏显示

   def do_area_clicked(self,point):
      self.__labAreaXY.setText("AreaSeries X=%.2f,Y=%.2f"
                               %(point.x(),point.y()))   #状态栏显示
      


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
