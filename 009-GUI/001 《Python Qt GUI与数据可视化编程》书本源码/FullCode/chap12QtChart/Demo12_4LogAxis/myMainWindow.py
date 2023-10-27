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
      self.__buildStatusBar()
      self.ui.frameSetup.setEnabled(False) #禁用控制面板

## 创建QmyChartView对象，并添加到窗口上
      self.ui.chartView.setRenderHint(QPainter.Antialiasing)
      self.ui.chartView.setCursor(Qt.CrossCursor)  #设置鼠标指针为十字星
      self.__iniChart()  # 创建self.chart
      

##  ==============自定义功能函数========================
   def __buildStatusBar(self):   ##构建状态栏
      self.__labMagXY = QLabel("幅频曲线，") 
      self.__labMagXY.setMinimumWidth(250)
      self.ui.statusBar.addWidget(self.__labMagXY)

      self.__labPhaseXY = QLabel("相频曲线，")
      self.__labPhaseXY.setMinimumWidth(250)
      self.ui.statusBar.addWidget(self.__labPhaseXY)
      

   def __iniChart(self):  ##图表初始化
      self.chart = QChart()   #创建 chart
      self.chart.setTitle("二阶系统频率特性")
      self.chart.legend().setVisible(True)
      self.ui.chartView.setChart(self.chart) #chart添加到chartView

      ##2. 创建坐标轴
      ## bottom 轴是 QLogValueAxis
      self.__axisFreq = QLogValueAxis()
      self.__axisFreq.setLabelFormat("%.1f")    #标签格式
      self.__axisFreq.setTitleText("角频率(rad/sec)") 
      self.__axisFreq.setRange(0.1,100) 
      self.__axisFreq.setMinorTickCount(9)
      self.chart.addAxis(self.__axisFreq,Qt.AlignBottom)

      ## left 轴是 QValueAxis
      self.__axisMag = QValueAxis()
      self.__axisMag.setTitleText("幅度(dB)")
      self.__axisMag.setRange(-40, 10)
      self.__axisMag.setTickCount(6)
      self.__axisMag.setLabelFormat("%.1f")  #标签格式
      self.chart.addAxis(self.__axisMag,Qt.AlignLeft)

      ## right 轴是 QValueAxis
      self.__axisPhase = QValueAxis()
      self.__axisPhase.setTitleText("相位(度)")
      self.__axisPhase.setRange(-200,0)
      self.__axisPhase.setTickCount(6)
      self.__axisPhase.setLabelFormat("%.0f")   #标签格式
      self.chart.addAxis(self.__axisPhase,Qt.AlignRight)

   def __loadData(self,allLines):  ##从字符串列表读取数据
      rowCnt=len(allLines)    #文本行数
      self.__vectW=[0]*rowCnt
      self.__vectMag=[0]*rowCnt
      self.__VectPhase=[0]*rowCnt
      
      for i in range(rowCnt):  
         lineText=allLines[i].strip()  #一行的文字，必须去掉末尾的\n
         strList=lineText.split()      #分割为字符串列表 
         self.__vectW[i]=float(strList[0])      #频率
         self.__vectMag[i]=float(strList[1])    #幅度
         self.__VectPhase[i]=float(strList[2])  #相位
         
   def __drawBode(self):
      self.chart.removeAllSeries()  #删除所有序列
   ## 创建序列
      pen=QPen(Qt.red)
      pen.setWidth(2)

      seriesMag = QLineSeries()     #幅频曲线序列
      seriesMag.setName("幅频曲线")
      seriesMag.setPen(pen)
      seriesMag.setPointsVisible(False)
      seriesMag.hovered.connect(self.do_seriesMag_hovered)

      seriesPhase = QLineSeries()   #相频曲线序列
      pen.setColor(Qt.blue)
      seriesPhase.setName("相频曲线")
      seriesPhase.setPen(pen)
      seriesPhase.setPointsVisible(True)
      seriesPhase.hovered.connect(self.do_seriesPhase_hovered)

   ## 为序列添加数据点
      count=len(self.__vectW)       #数据点数
      for i in range(count):
         seriesMag.append(self.__vectW[i],self.__vectMag[i])
         seriesPhase.append(self.__vectW[i],self.__VectPhase[i])

   ##设置坐标轴范围
      minMag=min(self.__vectMag)
      maxMag=max(self.__vectMag)
      minPh=min(self.__VectPhase)
      maxPh=max(self.__VectPhase)
      self.__axisMag.setRange(minMag,maxMag)
      self.__axisPhase.setRange(minPh,maxPh)

   ##序列添加到chart，并指定坐标轴
      self.chart.addSeries(seriesMag)
      seriesMag.attachAxis(self.__axisFreq)
      seriesMag.attachAxis(self.__axisMag)

      self.chart.addSeries(seriesPhase)
      seriesPhase.attachAxis(self.__axisFreq)
      seriesPhase.attachAxis(self.__axisPhase)

      for marker in self.chart.legend().markers():  #QLegendMarker类型列表
         marker.clicked.connect(self.do_LegendMarkerClicked)
      

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
      self.__axisMag.setRange(self.ui.spinMag_Min.value(),
                              self.ui.spinMag_Max.value())
      
   @pyqtSlot(int)    ##分度数
   def on_spinMag_Ticks_valueChanged(self,arg1):
      self.__axisMag.setTickCount(arg1)
      

   @pyqtSlot(bool)   ##显示数据点
   def on_chkBoxMag_Point_clicked(self,checked):
      seriesMag =self.chart.series()[0]
      seriesMag.setPointsVisible(checked)
   
## 相频曲线设置      
   @pyqtSlot()       ##设置坐标范围
   def on_btnPh_SetRange_clicked(self):
      self.__axisPhase.setRange(self.ui.spinPh_Min.value(),
                              self.ui.spinPh_Max.value())

      
   @pyqtSlot(int)    ##分度数
   def on_spinPh_Ticks_valueChanged(self,arg1):
      self.__axisPhase.setTickCount(arg1)
      

   @pyqtSlot(bool)   ##显示数据点
   def on_chkBoxPh_Point_clicked(self,checked):
      seriesPhase =self.chart.series()[1]
      seriesPhase.setPointsVisible(checked)
      

## 频率坐标轴
   @pyqtSlot()       ##设置坐标范围
   def on_btnX_SetRange_clicked(self):
      self.__axisFreq.setRange(self.ui.spinX_Min.value(),
                              self.ui.spinX_Max.value())
      
      
   @pyqtSlot(int)    ##次分度数
   def on_spinX_MinorTicks_valueChanged(self,arg1):
      self.__axisFreq.setMinorTickCount(arg1)
      

           
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
      if state:
         hint="幅频曲线：频率=%.1f, 幅度=%.1f dB"%(point.x(),point.y())
         self.__labMagXY.setText(hint)
##      else:
##         self.__labMagXY.setText("幅频曲线")
         
   def do_seriesPhase_hovered(self,point,state):
      if state:
         hint="相频曲线：频率=%.1f, 相位=%.1f 度"%(point.x(),point.y())
         self.__labPhaseXY.setText(hint)
##      else:
##         self.__labPhaseXY.setText("相频曲线")

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
