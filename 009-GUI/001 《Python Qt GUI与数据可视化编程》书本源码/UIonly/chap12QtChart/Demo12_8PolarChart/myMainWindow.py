# -*- coding: utf-8 -*-

import sys, math

from PyQt5.QtWidgets import  QApplication, QMainWindow,QLabel

from PyQt5.QtCore import  Qt,pyqtSlot

from PyQt5.QtGui import QPen,QPainter

##from PyQt5.QtSql import 

from PyQt5.QtChart import QSplineSeries,QValueAxis,QPolarChart,QChart

from ui_MainWindow import Ui_MainWindow


class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    # 调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     # 创建UI对象
      self.ui.setupUi(self)       # 构造UI界面

      self.setWindowTitle("Demo12_8, 极坐标图")
      
      self.__iniChart()  # 创建self.chart
      self.__drawRose()


##  ==============自定义功能函数========================

   def __iniChart(self):  #图表初始化
      pass
   

   def __drawRose(self):
      pass


##  ==============event处理函数==========================
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
   ## 工具栏按钮
   @pyqtSlot()   ##重画曲线
   def on_actRedraw_triggered(self):
      self.ui.spinAngle_Min.setValue(0)
      self.ui.spinAngle_Max.setValue(360)
      self.__drawRose()
      
   @pyqtSlot()    ##放大
   def on_actZoomIn_triggered(self):
      self.ui.chartView.chart().zoom(1.2)
      
   @pyqtSlot()    ##缩小
   def on_actZoomOut_triggered(self):
      self.ui.chartView.chart().zoom(0.8)

   @pyqtSlot()    ##恢复原始大小
   def on_actZoomReset_triggered(self):
      self.ui.chartView.chart().zoomReset()
      
   ##图表外观控制
   @pyqtSlot(int)  ##主题
   def on_comboTheme_currentIndexChanged(self,index):
      self.ui.chartView.chart().setTheme(QChart.ChartTheme(index))

   @pyqtSlot(bool)  ##显示数据点
   def on_chkBox_ShowPoints_clicked(self,checked):
      series=self.ui.chartView.chart().series()[0]
      series.setPointsVisible(checked)
      
   ## 角度坐标轴设置      
   @pyqtSlot(int)    ##设置坐标最小值
   def on_spinAngle_Min_valueChanged(self, arg1):
      self.__axisAngle.setMin(arg1)

   @pyqtSlot(int)    ##设置坐标最大值
   def on_spinAngle_Max_valueChanged(self, arg1):
      self.__axisAngle.setMax(arg1)

   @pyqtSlot(int)    ##分度数
   def on_spinAngle_Ticks_valueChanged(self,arg1):
      self.__axisAngle.setTickCount(arg1)

   ## 径向坐标轴设置     
   @pyqtSlot(int)    ##设置坐标范围
   def on_spinRadial_Max_valueChanged(self,arg1):
      self.__axisRadial.setMax(arg1)

   @pyqtSlot(int)    ##分度数
   def on_spinRadial_Ticks_valueChanged(self,arg1):
      self.__axisRadial.setTickCount(arg1)


   @pyqtSlot(int)    ##花瓣个数
   def on_spinCount_valueChanged(self,value):
      self.__drawRose()

   ## 旋转
   @pyqtSlot()    ##两个相邻点的角度差如果大于180度，就不会直连，而是连接到中心
   def on_btnRotate_clicked(self):
      pass

           
##  =============自定义槽函数===============================        
   def do_series_hovered(self,point,state):
      pass


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
