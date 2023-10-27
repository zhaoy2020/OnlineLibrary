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
      self.__buildStatusBar()

      self.ui.chartView.setRenderHint(QPainter.Antialiasing)
      self.ui.chartView.setCursor(Qt.CrossCursor)  #设置鼠标指针为十字星

## 初始化Model/View结构
      self.itemModel=QStandardItemModel(self)
      self.ui.tableView.setModel(self.itemModel)
      self.ui.tableView.setAlternatingRowColors(True)
      self.ui.tableView.horizontalHeader().setDefaultSectionSize(80)
      self.ui.tableView.verticalHeader().setDefaultSectionSize(24)

      self.__iniChart()  # 初始化图表
      self.ui.chartView.mouseMove.connect(self.do_chartView_mouseMove)
      

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
      self.chart = QChart() #创建 Chart
      self.chart.setTitle("股票日线图")
      self.chart.setTheme(QChart.ChartThemeBlueCerulean) #ChartThemeBlueCerulean, ChartThemeQt,ChartThemeBlueNcs,ChartThemeDark
      self.ui.chartView.setChart(self.chart) #Chart添加到ChartView

   ## X 轴是QDateTimeAxis
      self.__axisX = QDateTimeAxis()
      dateFormat=self.ui.comboDateFormat.currentText()  #如"MM-dd"
      self.__axisX.setFormat(dateFormat)  #标签格式
      self.__axisX.setTickCount(10)       #主分隔个数
      self.__axisX.setTitleText("日期")   #标题
      
      dateMin=QDateTime.fromString("2018-01-01","yyyy-MM-dd")
      self.__axisX.setMin(dateMin)
      dateMax=dateMin.addDays(150)
      self.__axisX.setMax(dateMax)
      self.chart.addAxis(self.__axisX,Qt.AlignBottom)

   ## Y 轴是 QValueAxis
      self.__axisY = QValueAxis()
      self.__axisY.setTitleText("Value")
      self.__axisY.setRange(0, 20)
      self.__axisY.setTickCount(5)
      self.__axisY.setLabelFormat("%.2f") #标签格式
      self.chart.addAxis(self.__axisY,Qt.AlignLeft)


   def __loadData(self,allLines):   ##从字符串列表读取数据构建数据模型
      rowCount=len(allLines)    #文本行数，第1行是标题
      self.itemModel.setRowCount(rowCount-1) #实际数据行数
   ## 设置表头
      header=allLines[0].strip()   #第1行是表头
      headerList=header.split()
      self.itemModel.setHorizontalHeaderLabels(headerList) #设置表头文字

      colCount=len(headerList) #列数
      self.itemModel.setColumnCount(colCount) #数据列数

   ## 设置模型数据
      for i in range(rowCount-1):
         lineText=allLines[i+1].strip()   #获取 数据区 的一行
         tmpList=lineText.split() 
         for j in range(colCount):
            item=QStandardItem(tmpList[j])  #创建item
            item.setTextAlignment(Qt.AlignHCenter)
            self.itemModel.setItem(i,j,item)  #为模型的某个行列位置设置Item
         
   def __drawChart(self):  ##绘制图表
      self.chart.removeAllSeries()#删除所有序列
      self.chart.setTitle("股票日线图--"+self.ui.tabWidget.tabText(0))

   ## 1. 创建蜡烛图
      seriesCandle =  QCandlestickSeries()
      seriesCandle.setName("蜡烛图")
      seriesCandle.setIncreasingColor(Qt.red)         #暴涨
      seriesCandle.setDecreasingColor(Qt.darkGreen)   #暴跌

      visible=self.ui.chkBox_Outline.isChecked()
      seriesCandle.setBodyOutlineVisible(visible)
      seriesCandle.setCapsVisible(self.ui.chkBox_Caps.isChecked())

      self.chart.addSeries(seriesCandle)
      seriesCandle.attachAxis(self.__axisX)
      seriesCandle.attachAxis(self.__axisY)

      seriesCandle.clicked.connect(self.do_candleClicked)
      seriesCandle.hovered.connect(self.do_candleHovered)


   ## 2. 创建MA曲线
      pen=QPen()
      pen.setWidth(2)

      seriesMA1 =  QLineSeries()    #不能使用QSplineSeries
      seriesMA1.setName("MA5")
      pen.setColor(Qt.magenta)
      seriesMA1.setPen(pen)
      self.chart.addSeries(seriesMA1)
      seriesMA1.attachAxis(self.__axisX)
      seriesMA1.attachAxis(self.__axisY)

      seriesMA2 =  QLineSeries()
      seriesMA2.setName("MA10")
      pen.setColor(Qt.yellow)
      seriesMA2.setPen(pen)
      self.chart.addSeries(seriesMA2)
      seriesMA2.attachAxis(self.__axisX)
      seriesMA2.attachAxis(self.__axisY)

      seriesMA3 =  QLineSeries()
      seriesMA3.setName("MA20")
      pen.setColor(Qt.cyan)
      seriesMA3.setPen(pen)
      self.chart.addSeries(seriesMA3)
      seriesMA3.attachAxis(self.__axisX)
      seriesMA3.attachAxis(self.__axisY)

      seriesMA4 =  QLineSeries()
      seriesMA4.setName("MA60")
      pen.setColor(Qt.green)#green
      seriesMA4.setPen(pen)
      self.chart.addSeries(seriesMA4)
      seriesMA4.attachAxis(self.__axisX)
      seriesMA4.attachAxis(self.__axisY)

   ## 3. 填充数据到序列
      dataRowCount=self.itemModel.rowCount()    #数据点个数
      for i in range(dataRowCount):
         dateStr=self.itemModel.item(i,0).text()    #日期字符串，如"2017/02/03"
         dateValue=QDate.fromString(dateStr,"yyyy/MM/dd")   #QDate
         dtValue=QDateTime(dateValue)           #日期时间 QDateTime
         timeStamp=dtValue.toMSecsSinceEpoch()  #毫秒数

         oneCandle= QCandlestickSet()   #QCandlestickSet
         oneCandle.setOpen(float(self.itemModel.item(i,1).text())) #开盘
         oneCandle.setHigh(float(self.itemModel.item(i,2).text())) #最高
         oneCandle.setLow(float(self.itemModel.item(i,3).text()))  #最低
         oneCandle.setClose(float(self.itemModel.item(i,4).text()))#收盘
         oneCandle.setTimestamp(timeStamp)   #时间戳
         seriesCandle.append(oneCandle)      #添加到序列

         M1=float(self.itemModel.item(i,5).text())
         M2=float(self.itemModel.item(i,6).text())
         M3=float(self.itemModel.item(i,7).text())
         M4=float(self.itemModel.item(i,8).text())

         seriesMA1.append(timeStamp,M1)
         seriesMA2.append(timeStamp,M2)
         seriesMA3.append(timeStamp,M3)
         seriesMA4.append(timeStamp,M4)

   ## 4. 设置坐标轴范围
      minDateStr=self.itemModel.item(0,0).text()   #日期字符串，如"2017/02/03"
      minDate=QDate.fromString(minDateStr,"yyyy/MM/dd")  #QDate
      minDateTime=QDateTime(minDate)  #最小日期时间,QDateTime

      maxDateStr=self.itemModel.item(dataRowCount-1,0).text()  #日期字符串，如"2017/05/03"
      maxDate=QDate.fromString(maxDateStr,"yyyy/MM/dd")
      maxDateTime=QDateTime(maxDate)  #最大日期时间
      
      self.__axisX.setRange(minDateTime,maxDateTime)     #日期时间范围
      dateFormat=self.ui.comboDateFormat.currentText()   #格式，如"MM-dd"
      self.__axisX.setFormat(dateFormat)  #标签格式

      self.__axisY.applyNiceNumbers()     #自动

      for marker in self.chart.legend().markers():  #QLegendMarker类型列表
         marker.clicked.connect(self.do_LegendMarkerClicked)

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
      seriesCandle =self.chart.series()[0]
      seriesCandle.setCapsVisible(checked)

   @pyqtSlot(bool)   ##bodyOutlineVisible
   def on_chkBox_Outline_clicked(self,checked):
      seriesCandle =self.chart.series()[0]
      seriesCandle.setBodyOutlineVisible(checked)

   
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
      valOpen=dataSet.open()
      valClose=dataSet.close()
      valHigh=dataSet.high()
      valLow=dataSet.low()
      price="开盘%.2f, 收盘%.2f, 最高%.2f, 最低%.2f"%(
                valOpen,valClose,valHigh,valLow)

      timeStamp=dataSet.timestamp()  #时间戳数据
      dt=QDateTime.fromMSecsSinceEpoch(timeStamp)
      dateStr=dt.toString("yyyy-MM-dd, ")

      self.__labClickXY.setText(dateStr+price)

         
   def do_candleHovered(self,status,dataSet):
      if status==False:
         self.__labHoverXY.setText("Hovered candle")
         return

      valOpen=dataSet.open()
      valClose=dataSet.close()
      valHigh=dataSet.high()
      valLow=dataSet.low()
      price="开盘%.2f, 收盘%.2f, 最高%.2f, 最低%.2f"%(
             valOpen,valClose,valHigh,valLow)

      timeStamp=dataSet.timestamp()  #时间戳数据
      dt=QDateTime.fromMSecsSinceEpoch(timeStamp)
      dateStr=dt.toString("yyyy-MM-dd, ")

      self.__labHoverXY.setText(dateStr+price)

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
