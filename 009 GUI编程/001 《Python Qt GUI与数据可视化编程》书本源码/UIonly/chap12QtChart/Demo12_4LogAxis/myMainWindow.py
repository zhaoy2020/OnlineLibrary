# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QWidget,QLabel,QFileDialog

from PyQt5.QtCore import  Qt,pyqtSlot,QDir,QFileInfo

from PyQt5.QtGui import QPen,QPainter,QBrush,QColor

from PyQt5.QtChart import QChart,QLineSeries,QValueAxis,QLegendMarker,QLogValueAxis

from ui_MainWindow import Ui_MainWindow

##from myChartView import QmyChartView

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo12_4, 对数坐标轴和多坐标轴")
      pass
      

##  ==============自定义功能函数========================
   def __buildStatusBar(self):   ##构建状态栏
      self.__labMagXY = QLabel("幅频曲线，") 
      self.__labMagXY.setMinimumWidth(250)
      self.ui.statusBar.addWidget(self.__labMagXY)

      self.__labPhaseXY = QLabel("相频曲线，")
      self.__labPhaseXY.setMinimumWidth(250)
      self.ui.statusBar.addWidget(self.__labPhaseXY)
      

   def __iniChart(self):  ##图表初始化
      pass


   def __loadData(self,allLines):  ##从字符串列表读取数据
      pass

         
   def __drawBode(self):
      pass


##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
   @pyqtSlot()    ##打开数据
   def on_actOpen_triggered(self):     
      curPath=QDir.currentPath()    #获取当前路径
      filename,flt=QFileDialog.getOpenFileName(self,"打开一个文件",curPath,
                  "频率响应数据文件(*.txt);;所有文件(*.*)")
      if (filename==""):
         return

      aFile=open(filename,'r')
      allLines=aFile.readlines()  #读取所有行,list类型,每行末尾带有 \n
      aFile.close()
      fileInfo=QFileInfo(filename)
      QDir.setCurrent(fileInfo.absolutePath())
      
      self.__loadData(allLines)   #解析数据
      self.__drawBode()    #绘制幅频曲线和相频曲线
      self.ui.frameSetup.setEnabled(True)
      

   @pyqtSlot()    ##放大
   def on_actZoomIn_triggered(self):
      self.ui.chartView.chart().zoom(1.2)
      
   @pyqtSlot()    ##缩小
   def on_actZoomOut_triggered(self):
      self.ui.chartView.chart().zoom(0.8)

   @pyqtSlot()    ##原始大小
   def on_actZoomReset_triggered(self):
      self.ui.chartView.chart().zoomReset()
      
##图表外观控制
   @pyqtSlot(int)    ##主题
   def on_comboTheme_currentIndexChanged(self,index):
      self.ui.chartView.chart().setTheme(QChart.ChartTheme(index))
      
   @pyqtSlot(bool)   ##显示图例
   def on_chkBox_Legend_clicked(self,checked):
      self.ui.chartView.chart().legend().setVisible(checked)
      
## 幅频曲线设置      
   @pyqtSlot()       ##设置坐标范围
   def on_btnMag_SetRange_clicked(self):
      pass

   @pyqtSlot(int)    ##分度数
   def on_spinMag_Ticks_valueChanged(self,arg1):
      pass

   @pyqtSlot(bool)   ##显示数据点
   def on_chkBoxMag_Point_clicked(self,checked):
      pass
   
## 相频曲线设置      
   @pyqtSlot()       ##设置坐标范围
   def on_btnPh_SetRange_clicked(self):
      pass

   @pyqtSlot(int)    ##分度数
   def on_spinPh_Ticks_valueChanged(self,arg1):
      pass

   @pyqtSlot(bool)   ##显示数据点
   def on_chkBoxPh_Point_clicked(self,checked):
      pass

## 频率坐标轴
   @pyqtSlot()       ##设置坐标范围
   def on_btnX_SetRange_clicked(self):
      pass

   @pyqtSlot(int)    ##次分度数
   def on_spinX_MinorTicks_valueChanged(self,arg1):
      pass
      

           
##  =============自定义槽函数===============================        
   def do_LegendMarkerClicked(self):
      marker =self.sender()   #QLegendMarker
      if (marker.type() != QLegendMarker.LegendMarkerTypeXY):
         return
      
      marker.series().setVisible(not marker.series().isVisible())
      marker.setVisible(True)
      alpha = 1.0
      if not marker.series().isVisible():
         alpha = 0.5

      brush = marker.labelBrush()  #QBrush
      color = brush.color()        #QColor
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


   def do_seriesMag_hovered(self,point,state):
      pass

         
   def do_seriesPhase_hovered(self,point,state):
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
