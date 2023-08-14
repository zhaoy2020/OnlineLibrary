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
      self.__buildStatusBar()
      self.ui.chartView.setRenderHint(QPainter.Antialiasing)
      self.ui.chartView.setCursor(Qt.CrossCursor)  #设置鼠标指针为十字星
      self.ui.chartView.mouseMove.connect(self.do_chartView_mouseMove)

      self.__colorLine=Qt.darkBlue      #曲线颜色,darkBlue
      self.__colorFill=Qt.darkBlue      #填充颜色 gray
      self.__iniChart()       #创建self.chart
##  "填充类型"4个RadioButton关联槽函数
      self.ui.radioFill_Pos.clicked.connect(self.do_redrawFill)   #positive
      self.ui.radioFill_Neg.clicked.connect(self.do_redrawFill)   #negative
      self.ui.radioFill_Both.clicked.connect(self.do_redrawFill)  #both
      self.ui.radioFill_None.clicked.connect(self.do_redrawWave)  #wiggle
      

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
      self.chart = QChart()    #创建 chart
      self.chart.setTitle("地震波形")
      self.chart.legend().setVisible(False)  #不显示图例
      self.ui.chartView.setChart(self.chart) #chart添加到chartView
## 创建坐标轴X
      self.__axisX = QValueAxis()   # bottom 轴是 QValueAxis
      self.__axisX.setTitleText("时间(秒)")
      self.__axisX.setRange(0, 10)
      self.__axisX.setTickCount(10)
      self.__axisX.setLabelFormat("%.2f")  #标签格式
      self.chart.addAxis(self.__axisX,Qt.AlignBottom)
## 创建坐标轴Y
      self.__axisY = QValueAxis()   # left 轴是 QValueAxis
      self.__axisY.setTitleText("幅度")
      self.__axisY.setRange(-5,5)
      self.__axisY.setTickCount(5)
      self.__axisY.setLabelFormat("%.2f")  #标签格式
      self.chart.addAxis(self.__axisY,Qt.AlignLeft)

##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
##=====工具栏 按钮功能
   @pyqtSlot()    ##“打开文件”按钮
   def on_actOpen_triggered(self):
      curPath=QDir.currentPath()    #获取当前路径
      filename,flt=QFileDialog.getOpenFileName(self,"打开一个文件",curPath,
                  "地震数据文件(*.txt);;所有文件(*.*)")
      if (filename==""):
         return

      aFile=open(filename,'r')
      allLines=aFile.readlines()  #读取所有行,list类型,每行末尾带有 \n
      aFile.close()
      fileInfo=QFileInfo(filename)
      QDir.setCurrent(fileInfo.absolutePath())
      self.__labFileName.setText("数据文件："+fileInfo.fileName())
      
      rowCnt=len(allLines)    #行数，即数据点数
      self.__vectData=[0]*rowCnt   #列表
      for i in range(rowCnt):      
         lineText=allLines[i].strip()  #字符串表示的数字
         self.__vectData[i]=float(lineText)
      minV=min(self.__vectData)  #最小值
      self.ui.spinY_Min.setValue(minV)
      maxV=max(self.__vectData)  #最大值
      self.ui.spinY_Max.setValue(maxV)
      
      if self.ui.radioFill_None.isChecked():
         self.do_redrawWave()    #绘制波形曲线
      else:
         self.do_redrawFill()    #绘制有填充的波形
      self.ui.frameSetup.setEnabled(True)

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
      self.chart.removeAllSeries()  # 删除所有序列
      pen=QPen(self.__colorLine)    # 曲线颜色
      pen.setWidth(2)
      seriesWave = QLineSeries() 
      seriesWave.setUseOpenGL(True) 
      seriesWave.setPen(pen)

      vx=0
      intv=0.001  #1000Hz采样
      pointCount=len(self.__vectData)
      for i in range(pointCount):
         value=self.__vectData[i]
         seriesWave.append(vx,value)
         vx =vx+ intv
      self.__axisX.setRange(0,vx)

      self.chart.addSeries(seriesWave)
      seriesWave.attachAxis(self.__axisX)
      seriesWave.attachAxis(self.__axisY)
      

   @pyqtSlot()    ##绘制3种填充曲线
   def do_redrawFill(self):
      self.chart.removeAllSeries()  #删除所有序列
      pen=QPen(self.__colorLine)    #线条颜色
      pen.setWidth(2)

      seriesFullWave = QLineSeries()      #全波形
      seriesFullWave.setUseOpenGL(True)
      seriesFullWave.setPen(pen)

      seriesPositive = QLineSeries()      #正半部分曲线
      seriesPositive.setUseOpenGL(True)
      seriesPositive.setVisible(False)    #不显示

      seriesNegative = QLineSeries()      #负半部分曲线
      seriesNegative.setUseOpenGL(True)
      seriesNegative.setVisible(False)    #不显示即可

      seriesZero = QLineSeries()          #零均值线
      seriesZero.setUseOpenGL(True)
      seriesZero.setVisible(False)        #不显示即可

   ## 填充数据
      vx=0
      intv=0.001   #1000Hz采样，数据点间隔时间
      pointCount=len(self.__vectData)
      for i in range(pointCount):
         value=self.__vectData[i]
         seriesFullWave.append(vx,value)  #完整波形
         seriesZero.append(vx,0)          #零值线
         if value>0:
            seriesPositive.append(vx,value)  #正半部分波形
            seriesNegative.append(vx,0)
         else:
            seriesPositive.append(vx,0)
            seriesNegative.append(vx,value) #负半部分波形
         vx =vx+intv

      self.__axisX.setRange(0,vx)

   ##  创建QAreaSeries序列，设置上、下界的QLineSeries对象
      pen.setStyle(Qt.NoPen)  #无线条,隐藏填充区域的边线
      if self.ui.radioFill_Pos.isChecked():     #positive fill
         series = QAreaSeries(seriesPositive, seriesZero) #QAreaSeries
         series.setColor(self.__colorFill)      #填充色
         series.setPen(pen)   #不显示线条
         self.chart.addSeries(series)
         series.attachAxis(self.__axisX)
         series.attachAxis(self.__axisY)
         
      elif self.ui.radioFill_Neg.isChecked():  #negative fill
         series = QAreaSeries(seriesZero,seriesNegative)
         series.setColor(self.__colorFill)
         series.setPen(pen)   #不显示线条
         self.chart.addSeries(series)
         series.attachAxis(self.__axisX)
         series.attachAxis(self.__axisY)

      elif self.ui.radioFill_Both.isChecked():  #both fill
         series = QAreaSeries(seriesZero,seriesFullWave)
         series.setColor(self.__colorFill)
         series.setPen(pen)   #不显示线条
         self.chart.addSeries(series)
         series.attachAxis(self.__axisX)
         series.attachAxis(self.__axisY)

      series.clicked.connect(self.do_area_clicked)  #关联槽函数
         
   ## 构建QAreaSeries的两个QLineSeries序列必须添加到chart里，否则程序崩溃
      self.chart.addSeries(seriesZero)       #隐藏
      self.chart.addSeries(seriesPositive)   #隐藏
      self.chart.addSeries(seriesNegative)   #隐藏
      self.chart.addSeries(seriesFullWave)   #全波形曲线，显示

      seriesPositive.attachAxis(self.__axisX)
      seriesPositive.attachAxis(self.__axisY)

      seriesNegative.attachAxis(self.__axisX)
      seriesNegative.attachAxis(self.__axisY)

      seriesZero.attachAxis(self.__axisX)
      seriesZero.attachAxis(self.__axisY)
      
      seriesFullWave.attachAxis(self.__axisX)
      seriesFullWave.attachAxis(self.__axisY)

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
