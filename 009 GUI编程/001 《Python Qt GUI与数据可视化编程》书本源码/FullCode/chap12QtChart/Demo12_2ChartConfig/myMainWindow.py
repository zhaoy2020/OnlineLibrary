# -*- coding: utf-8 -*-

import sys, math, random

from PyQt5.QtWidgets import  (QApplication, QMainWindow,QDialog,
                           QFontDialog,QColorDialog)

from PyQt5.QtCore import  pyqtSlot,Qt,QMargins

from PyQt5.QtGui import QPainter,QPen,QColor,QBrush,QFont

from PyQt5.QtChart import QChart,QLineSeries,QValueAxis

from myDialogPen import QmyDialogPen

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo12_2, QChart绘制折线图")
      self.setCentralWidget(self.ui.splitter)

      self.__chart=None       #图表
      self.__curSeries=None   #当前序列
      self.__curAxis=None     #当前坐标轴
      
      self.__createChart()
      self.__prepareData()
      self.__updateFromChart()

##  ==============自定义功能函数========================
   def __createChart(self):
      self.__chart = QChart()
      self.__chart.setTitle("简单函数曲线")
      self.ui.chartView.setChart(self.__chart)
      self.ui.chartView.setRenderHint(QPainter.Antialiasing)

      series0 =  QLineSeries()
      series0.setName("Sin曲线")
      series1 =  QLineSeries()
      series1.setName("Cos曲线")
      self.__curSeries=series0   #当前序列

      pen=QPen(Qt.red)
      pen.setStyle(Qt.DotLine)   #SolidLine, DashLine, DotLine, DashDotLine
      pen.setWidth(2)
      series0.setPen(pen)        #序列的线条设置

      pen.setStyle(Qt.SolidLine) #SolidLine, DashLine, DotLine, DashDotLine
      pen.setColor(Qt.blue)
      series1.setPen(pen)        #序列的线条设置

      self.__chart.addSeries(series0)
      self.__chart.addSeries(series1)

      axisX = QValueAxis()
      self.__curAxis=axisX       #当前坐标轴
      axisX.setRange(0, 10)      #设置坐标轴范围
      axisX.setLabelFormat("%.1f")  #标签格式
      axisX.setTickCount(11)        #主分隔个数
      axisX.setMinorTickCount(4)
      axisX.setTitleText("time(secs)")  #标题
      axisX.setGridLineVisible(True)
      axisX.setMinorGridLineVisible(False)

      axisY = QValueAxis()
      axisY.setRange(-2, 2)
      axisY.setLabelFormat("%.2f")     #标签格式
      axisY.setTickCount(5)
      axisY.setMinorTickCount(4)
      axisY.setTitleText("value")
      axisY.setGridLineVisible(True)
      axisY.setMinorGridLineVisible(False)

   ##      self.__chart.setAxisX(axisX, series0) #添加X坐标轴
   ##      self.__chart.setAxisX(axisX, series1) #添加X坐标轴
   ##      self.__chart.setAxisY(axisY, series0) #添加Y坐标轴
   ##      self.__chart.setAxisY(axisY, series1) #添加Y坐标轴

   ##另一种实现设置坐标轴的方法
      self.__chart.addAxis(axisX,Qt.AlignBottom) #坐标轴添加到图表，并指定方向
      self.__chart.addAxis(axisY,Qt.AlignLeft)

      series0.attachAxis(axisX)  #序列 series0 附加坐标轴
      series0.attachAxis(axisY)

      series1.attachAxis(axisX)  #序列 series1 附加坐标轴
      series1.attachAxis(axisY)

      

   def __prepareData(self):  ##为序列设置数据点
      chart=self.ui.chartView.chart()  #获取chartView中的QChart对象
      series0=chart.series()[0]        #获取第1个序列，QLineSeries
      series0.clear() 

      series1=chart.series()[1]   #获取第2个序列，QLineSeries
      series1.clear()
      
      t,y1,y2=0.0,0.0,0.0
      intv=0.1
      pointCount=100
      for i in range(pointCount):
         rd=random.randint(-5,5)     #随机数,-5~+5
         y1=math.sin(t)+rd/50.0
         series0.append(t,y1)        #序列添加数据点

         rd=random.randint(-5,5)     #随机数,-5~+5
         y2=math.cos(t)+rd/50.0
         series1.append(t,y2)        #序列添加数据点
         t=t+intv

   def __updateFromChart(self):
      self.ui.editTitle.setText(self.__chart.title())    #图表标题
      mg=self.__chart.margins()     #边距, QMargins
      self.ui.spinMarginLeft.setValue(mg.left())
      self.ui.spinMarginRight.setValue(mg.right())
      self.ui.spinMarginTop.setValue(mg.top())
      self.ui.spinMarginBottom.setValue(mg.bottom())
      
      
##  ==============event处理函数==========================


##  ==========由connectSlotsByName()自动连接的槽函数============        
      
##========工具栏上的几个按钮的Actions==============
   @pyqtSlot()  ## "刷新绘图" 工具栏按钮
   def on_actDraw_triggered(self):
      self.__prepareData()


##=====ToolBox 第1组：==="图表设置" 分组里的功能================
##=======1.1 标题========
   @pyqtSlot()   ##设置标题文字
   def on_btnTitleSetText_clicked(self):
      text=self.ui.editTitle.text()
      self.__chart.setTitle(text)
        
   @pyqtSlot()   ##设置标题文字颜色
   def on_btnTitleColor_clicked(self):
      color=self.__chart.titleBrush().color()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.__chart.setTitleBrush(QBrush(color))

   @pyqtSlot()   ##设置标题字体
   def on_btnTitleFont_clicked(self):
      iniFont=self.__chart.titleFont()  #QFont
      font,ok=QFontDialog.getFont(iniFont)
      if ok:
         self.__chart.setTitleFont(font)

##=======1.2 图例==========
   @pyqtSlot(bool)   ##图例是否可见
   def on_groupBox_Legend_clicked(self,checked):
      self.__chart.legend().setVisible(checked)

   @pyqtSlot()   ##图例的位置, 上
   def on_radioButton_clicked(self):
      self.__chart.legend().setAlignment(Qt.AlignTop)
         
   @pyqtSlot()   ##图例的位置，下
   def on_radioButton_2_clicked(self):
      self.__chart.legend().setAlignment(Qt.AlignBottom)
      
   @pyqtSlot()   ##图例的位置，左
   def on_radioButton_3_clicked(self):
      self.__chart.legend().setAlignment(Qt.AlignLeft)

   @pyqtSlot()   ##图例的位置，右
   def on_radioButton_4_clicked(self):
      self.__chart.legend().setAlignment(Qt.AlignRight)

   @pyqtSlot()   ##图例的文字颜色
   def on_btnLegendlabelColor_clicked(self):
      color=self.__chart.legend().labelColor()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.__chart.legend().setLabelColor(color)

   @pyqtSlot()   ##图例的字体
   def on_btnLegendFont_clicked(self):
      iniFont=self.__chart.legend().font()
      font,ok=QFontDialog.getFont(iniFont)
      if ok:
         self.__chart.legend().setFont(font)

##=======1.3 边距========
   @pyqtSlot()   ##设置图表的4个边距
   def on_btnSetMargin_clicked(self):
      mgs=QMargins()
      mgs.setLeft(self.ui.spinMarginLeft.value())
      mgs.setRight(self.ui.spinMarginRight.value())
      mgs.setTop(self.ui.spinMarginTop.value())
      mgs.setBottom(self.ui.spinMarginBottom.value())
      self.__chart.setMargins(mgs)

##=======1.4 动画效果========        
   @pyqtSlot(int)   ##动画效果
   def on_comboAnimation_currentIndexChanged(self,index):
      animation=QChart.AnimationOptions(index)
      self.__chart.setAnimationOptions(animation)

   @pyqtSlot(int)   ##图表的主题
   def on_comboTheme_currentIndexChanged(self,index):
      self.__chart.setTheme(QChart.ChartTheme(index))


##=====ToolBox 第2组：==="曲线设置" 分组里的功能================
##=======2.1 选择操作序列========
   @pyqtSlot()   ##获取当前数据序列,sin
   def on_radioSeries0_clicked(self):
      if self.ui.radioSeries0.isChecked():
         self.__curSeries=self.__chart.series()[0]
      else:
         self.__curSeries=self.__chart.series()[1]
      ##获取序列的属性值，并显示到界面上
      self.ui.editSeriesName.setText(self.__curSeries.name())
      self.ui.groupBox_Series.setChecked(self.__curSeries.isVisible())  #序列是否显示
      self.ui.chkBoxPointVisible.setChecked(self.__curSeries.pointsVisible())    #数据点是否显示
      self.ui.chkkBoxUseOpenGL.setChecked(self.__curSeries.useOpenGL()) #使用openGL

      self.ui.sliderOpacity.setValue(self.__curSeries.opacity()*10)  #透明度

      visible=self.__curSeries.pointLabelsVisible()   #数据点标签可见性
      self.ui.groupBox_PointLabel.setChecked(visible)

   @pyqtSlot()   ##获取当前数据序列,cos
   def on_radioSeries1_clicked(self):
      self.on_radioSeries0_clicked()

      
##======2.2 序列曲线 设置======== 
   @pyqtSlot(bool)   ##序列是否可见
   def on_groupBox_Series_clicked(self,checked):
      self.__curSeries.setVisible(checked)

   @pyqtSlot()   ##设置序列名称
   def on_btnSeriesName_clicked(self):
      seriesName=self.ui.editSeriesName.text()
      self.__curSeries.setName(seriesName)
      if self.ui.radioSeries0.isChecked():
         self.ui.radioSeries0.setText(seriesName)
      else:
         self.ui.radioSeries1.setText(seriesName)

   @pyqtSlot()   ##序列的曲线颜色
   def on_btnSeriesColor_clicked(self):
      color=self.__curSeries.color()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.__curSeries.setColor(color)

   @pyqtSlot()   ##序列曲线的Pen设置
   def on_btnSeriesPen_clicked(self):
      iniPen=self.__curSeries.pen()
      pen,ok=QmyDialogPen.staticGetPen(iniPen)
      if ok:
         self.__curSeries.setPen(pen)

   @pyqtSlot(bool)   ##序列的数据点是否可见,数据点形状是固定的
   def on_chkBoxPointVisible_clicked(self,checked):
      self.__curSeries.setPointsVisible(checked)

   @pyqtSlot(bool)   ##使用openGL加速后，不能设置线型，不能显示数据点
   def on_chkkBoxUseOpenGL_clicked(self,checked):
      self.__curSeries.setUseOpenGL(checked)

   @pyqtSlot(int)   ##序列的透明度
   def on_sliderOpacity_sliderMoved(self,position):
      self.__curSeries.setOpacity(position/10.0)

##=======2.3 数据点标签 ========     
   @pyqtSlot(bool)   ##数据点标签 groupBox
   def on_groupBox_PointLabel_clicked(self,checked):
      self.__curSeries.setPointLabelsVisible(checked)

   @pyqtSlot()   ##序列数据点标签颜色
   def on_btnSeriesLabColor_clicked(self):
      color=self.__curSeries.pointLabelsColor()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.__curSeries.setPointLabelsColor(color)

   @pyqtSlot()   ##序列数据点标签字体
   def on_btnSeriesLabFont_clicked(self):
      font=self.__curSeries.pointLabelsFont()    #QFont
      font,ok=QFontDialog.getFont(font)
      if ok:
         self.__curSeries.setPointLabelsFont(font)

   @pyqtSlot()   ##序列数据点标签的显示格式
   def on_radioSeriesLabFormat0_clicked(self):
      self.__curSeries.setPointLabelsFormat("@yPoint")
         
   @pyqtSlot()   ##序列数据点标签的显示格式
   def on_radioSeriesLabFormat1_clicked(self):
      self.__curSeries.setPointLabelsFormat("(@xPoint,@yPoint)")

##=====ToolBox 第3组：==="坐标轴设置" 分组里的功能================

##=======3.1 选择操作的坐标轴对象=======
   @pyqtSlot()   ##选择坐标轴X
   def on_radioAxisX_clicked(self):
      if (self.ui.radioAxisX.isChecked()):
         self.__curAxis=self.ui.chartView.chart().axisX()   #QValueAxis
      else:
         self.__curAxis=self.ui.chartView.chart().axisY()

      ##获取坐标轴的各种属性，显示到界面上
      self.ui.groupBox_Axis.setChecked(self.__curAxis.isVisible()) #坐标轴可见性

      self.ui.chkBoxAxisReverse.setChecked(self.__curAxis.isReverse())
         
      self.ui.spinAxisMin.setValue(self.__curAxis.min())
      self.ui.spinAxisMax.setValue(self.__curAxis.max())

      self.ui.editAxisTitle.setText(self.__curAxis.titleText())   #轴标题
      self.ui.groupBox_AxisTitle.setChecked(self.__curAxis.isTitleVisible())  #轴标题可见

      self.ui.editAxisLabelFormat.setText(self.__curAxis.labelFormat())       #标签格式
      self.ui.groupBox_AxisLabel.setChecked(self.__curAxis.labelsVisible())   #标签可见

      self.ui.groupBox_GridLine.setChecked(self.__curAxis.isGridLineVisible()) #网格线

      self.ui.groupBox_Ticks.setChecked(self.__curAxis.isLineVisible())    #主刻度线

      self.ui.spinTickCount.setValue(self.__curAxis.tickCount())           #主刻度个数

      self.ui.spinMinorTickCount.setValue(self.__curAxis.minorTickCount()) #次刻度个数
      self.ui.groupBox_MinorGrid.setChecked(self.__curAxis.isMinorGridLineVisible())   #次网格线可见


   @pyqtSlot()   ##选择坐标轴Y
   def on_radioAxisY_clicked(self):
      self.on_radioAxisX_clicked()

##======3.2 坐标轴可见性和范围=======
   @pyqtSlot(bool)   ##坐标轴可见性
   def on_groupBox_Axis_clicked(self,checked):
      self.__curAxis.setVisible(checked)

   @pyqtSlot(bool)   ##坐标反向
   def on_chkBoxAxisReverse_clicked(self,checked):
      self.__curAxis.setReverse(checked)
      
   @pyqtSlot()   ##设置坐标范围
   def on_btnSetAxisRange_clicked(self):
      minV=self.ui.spinAxisMin.value()
      maxV=self.ui.spinAxisMax.value()
      self.__curAxis.setRange(minV,maxV)
      
##======3.3 轴标题=======
   @pyqtSlot(bool)   ##坐标轴标题可见性
   def on_groupBox_AxisTitle_clicked(self,checked):
      self.__curAxis.setTitleVisible(checked)

   @pyqtSlot()   ##设置轴标题
   def on_btnAxisSetTitle_clicked(self):
      self.__curAxis.setTitleText(self.ui.editAxisTitle.text())

   @pyqtSlot()   ##设置轴标题的颜色
   def on_btnAxisTitleColor_clicked(self):
      color=self.__curAxis.titleBrush().color()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.__curAxis.setTitleBrush(QBrush(color))

   @pyqtSlot()   ##设置轴标题的字体
   def on_btnAxisTitleFont_clicked(self):
      iniFont=self.__curAxis.titleFont() #QFont
      font,ok=QFontDialog.getFont(iniFont)
      if ok:
         self.__curAxis.setTitleFont(font)

##======3.4 轴刻度标签=======
   @pyqtSlot(bool)   ##可见性
   def on_groupBox_AxisLabel_clicked(self,checked):
      self.__curAxis.setLabelsVisible(checked)
         
   @pyqtSlot()   ##设置标签格式
   def on_btnAxisLabelFormat_clicked(self):
      strFormat=self.ui.editAxisLabelFormat.text()
      self.__curAxis.setLabelFormat(strFormat)
      
   @pyqtSlot()   ##设置标签文字颜色
   def on_btnAxisLabelColor_clicked(self):
      color=self.__curAxis.labelsColor()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.__curAxis.setLabelsColor(color)

   @pyqtSlot()   ##设置标签字体
   def on_btnAxisLabelFont_clicked(self):
      iniFont=self.__curAxis.labelsFont() #QFont
      font,ok=QFontDialog.getFont(iniFont)
      if ok:
         self.__curAxis.setLabelsFont(font)

##======3.5 轴线和主刻度=========
   @pyqtSlot(bool)   ##可见性
   def on_groupBox_Ticks_clicked(self,checked):
      self.__curAxis.setLineVisible(checked)

   @pyqtSlot(int)   ##主刻度个数
   def on_spinTickCount_valueChanged(self,arg1):
      self.__curAxis.setTickCount(arg1)

   @pyqtSlot()   ##设置线条Pen
   def on_btnAxisLinePen_clicked(self):
      iniPen=self.__curAxis.linePen()
      pen,ok=QmyDialogPen.staticGetPen(iniPen)
      if ok:
         self.__curAxis.setLinePen(pen)
         
   @pyqtSlot()   ##设置线条颜色
   def on_btnAxisLinePenColor_clicked(self):
      color=self.__curAxis.linePenColor()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.__curAxis.setLinePenColor(color)
         
##======3.6 主网格线=========
   @pyqtSlot(bool)   ##可见性
   def on_groupBox_GridLine_clicked(self,checked):
      self.__curAxis.setGridLineVisible(checked)
         

   @pyqtSlot()   ##设置线条Pen
   def on_btnGridLinePen_clicked(self):
      iniPen=self.__curAxis.gridLinePen()
      pen,ok=QmyDialogPen.staticGetPen(iniPen)
      if ok:
         self.__curAxis.setGridLinePen(pen)

   @pyqtSlot()   ##设置线条颜色
   def on_btnGridLineColor_clicked(self):
      color=self.__curAxis.gridLineColor()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.__curAxis.setGridLineColor(color)

##======3.7 次网格线=======
   @pyqtSlot(bool)   ##可见性
   def on_groupBox_MinorGrid_clicked(self,checked):
      self.__curAxis.setMinorGridLineVisible(checked)

   @pyqtSlot(int)   ##次刻度个数
   def on_spinMinorTickCount_valueChanged(self,arg1):
      self.__curAxis.setMinorTickCount(arg1)
         
   @pyqtSlot()   ##设置线条Pen
   def on_btnMinorPen_clicked(self):
      iniPen=self.__curAxis.minorGridLinePen()
      pen,ok=QmyDialogPen.staticGetPen(iniPen)    #使用类函数调用方法
      if ok:
         self.__curAxis.setMinorGridLinePen(pen)
         
   @pyqtSlot()   ##设置线条颜色
   def on_btnMinorColor_clicked(self):
      color=self.__curAxis.minorGridLineColor()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.__curAxis.setMinorGridLineColor(color)
         
        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
