# -*- coding: utf-8 -*-

import sys, random

from PyQt5.QtWidgets import  QApplication, QMainWindow

from PyQt5.QtCore import  Qt,pyqtSlot

from PyQt5.QtGui import QStandardItemModel,QStandardItem, QPainter, QPen

from PyQt5.QtChart import *


from ui_MainWindow import Ui_MainWindow


class QmyMainWindow(QMainWindow): 
   COL_NAME    =0   #姓名的列编号
   COL_MATH    =1   #数学的列编号
   COL_CHINESE =2   #语文的列编号
   COL_ENGLISH =3   #英语的列编号
   COL_AVERAGE =4   #平均分的列编号

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo12_5，饼图和各种柱状图")

      self.ui.tableView.setAlternatingRowColors(True)
      self.ui.treeWidget.setAlternatingRowColors(True)
      self.setStyleSheet("QTreeWidget, QTableView{"
                        "alternate-background-color:rgb(170, 241, 190)}")

      self.__studCount=10  #学生人数
      self.ui.spinCount.setValue(self.__studCount)

      self.dataModel = QStandardItemModel(self)    #数据模型
      self.ui.tableView.setModel(self.dataModel)   #设置数据模型
      self.dataModel.itemChanged.connect(self.do_calcuAverage) #自动计算平均分
      
      self.__generateData()   #初始化数据
      self.__surveyData()     #数据统计
      
      self.__iniBarChart()       #柱状图初始化
      self.__iniStackedBar()     #堆积图初始化
      self.__iniPercentBar()     #百分比图初始化
      self.__iniPieChart()       #饼图初始化


##  ==============自定义功能函数========================
   def __generateData(self):  ##随机生成分数数据
      self.dataModel.clear()
      headerList=["姓名","数学","语文","英语","平均分"]
      self.dataModel.setHorizontalHeaderLabels(headerList)  #设置表头文字
      for i in range(self.__studCount):
         itemList=[]
         studName="学生%2d"%(i+1)
         item=QStandardItem(studName)  #创建item
         item.setTextAlignment(Qt.AlignHCenter)
         itemList.append(item)         #添加到列表

         avgScore=0  
         for j in range(self.COL_MATH, 1+self.COL_ENGLISH): #数学，语文，英语
            score=50.0+random.randint(-20,50)
            item=QStandardItem("%.0f"%score)    #创建 item
            item.setTextAlignment(Qt.AlignHCenter)
            itemList.append(item)      #添加到列表
            avgScore =avgScore+score

         item=QStandardItem("%.1f"%(avgScore/3.0))    #创建平均分item
         item.setTextAlignment(Qt.AlignHCenter)
         item.setFlags(item.flags() & (not Qt.ItemIsEditable)) #平均分不允许编辑
         itemList.append(item)   #添加到列表
         self.dataModel.appendRow(itemList)  #添加到数据模型
      
   def __surveyData(self):  ##统计各分数段人数
      for i in range(self.COL_MATH, 1+self.COL_ENGLISH): #统计 三列
         cnt50,cnt60,cnt70,cnt80,cnt90=0,0,0,0,0
         for j in range(self.dataModel.rowCount()):      #行数等于学生人数
            val=float(self.dataModel.item(j,i).text())   #分数
            if val<60:
               cnt50 =cnt50+1
            elif (val>=60 and val<70):
               cnt60 = cnt60+1
            elif (val>=70 and val<80):
               cnt70 =cnt70+1
            elif (val>=80 and val<90):
               cnt80 =cnt80+1
            else:
               cnt90 =cnt90+1
               
         item=self.ui.treeWidget.topLevelItem(0)   #第1行,<60
         item.setText(i,str(cnt50))                #第i列
         item.setTextAlignment(i,Qt.AlignHCenter)

         item=self.ui.treeWidget.topLevelItem(1)   #第2行,[60,70)
         item.setText(i,str(cnt60))  # 第i列
         item.setTextAlignment(i,Qt.AlignHCenter)

         item=self.ui.treeWidget.topLevelItem(2)   #第3行,[70,80)
         item.setText(i,str(cnt70))  # 第i列
         item.setTextAlignment(i,Qt.AlignHCenter)

         item=self.ui.treeWidget.topLevelItem(3)   #第4行,[80,90)
         item.setText(i,str(cnt80))  # 第i列
         item.setTextAlignment(i,Qt.AlignHCenter)

         item=self.ui.treeWidget.topLevelItem(4)   #第5行,[90,100]
         item.setText(i,str(cnt90))  # 第i列
         item.setTextAlignment(i,Qt.AlignHCenter)
         
      
   def __iniBarChart(self):   ##初始化柱状图
      chart = QChart()  
      chart.setTitle("Barchart 演示")
##      chart.setAnimationOptions(QChart.SeriesAnimations)
      self.ui.chartViewBar.setChart(chart)      #为ChartView设置chart
      self.ui.chartViewBar.setRenderHint(QPainter.Antialiasing)
      self.ui.chartViewBar.setCursor(Qt.CrossCursor)  #设置鼠标指针为十字星
      
   def __iniStackedBar(self):    ##初始化堆叠柱状图
      chart = QChart() 
      chart.setTitle("StackedBar 演示")
##      chart.setAnimationOptions(QChart.SeriesAnimations)
      self.ui.chartViewStackedBar.setChart(chart)    #为ChartView设置chart
      self.ui.chartViewStackedBar.setRenderHint(QPainter.Antialiasing)
      self.ui.chartViewStackedBar.setCursor(Qt.CrossCursor)  #设置鼠标指针为十字星
      

   def __iniPercentBar(self):    ##百分比柱状图初始化
      chart = QChart() 
      chart.setTitle("PercentBar 演示")
##      chart.setAnimationOptions(QChart.SeriesAnimations)
      self.ui.chartViewPercentBar.setChart(chart)    #为ChartView设置chart
      self.ui.chartViewPercentBar.setRenderHint(QPainter.Antialiasing)
      self.ui.chartViewPercentBar.setCursor(Qt.CrossCursor)  #设置鼠标指针为十字星

   def __iniPieChart(self):  ##饼图初始化
      chart = QChart() 
      chart.setTitle("Piechart 演示")
      chart.setAnimationOptions(QChart.SeriesAnimations)
   ##      chart.setAcceptHoverEvents(True) # 接受Hover事件
      self.ui.chartViewPie.setChart(chart)    #为ChartView设置chart
      self.ui.chartViewPie.setRenderHint(QPainter.Antialiasing)
      self.ui.chartViewPie.setCursor(Qt.CrossCursor)  #设置鼠标指针为十字星

   def __getCurrentChart(self):  ##获取当前QChart对象
      page=self.ui.tabWidget.currentIndex() 
      if page ==0:
         chart=self.ui.chartViewBar.chart()
      elif page ==1:
         chart=self.ui.chartViewStackedBar.chart()
      elif page ==2:
         chart=self.ui.chartViewPercentBar.chart()
      else:
         chart=self.ui.chartViewPie.chart()
      return chart
      
      
##  ==============event事件处理函数==========================

        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
## 工具栏按钮的功能
   @pyqtSlot()   ##重新生成数据
   def on_toolBtn_GenData_clicked(self):
      self.__studCount=self.ui.spinCount.value()  #学生人数
      self.__generateData()
      self.__surveyData()

   @pyqtSlot()   ##重新统计
   def on_toolBtn_Counting_clicked(self):
      self.__surveyData()  

   @pyqtSlot(int)   ##设置图表主题
   def on_comboTheme_currentIndexChanged(self,index):
      chart=self.__getCurrentChart()
      chart.setTheme(QChart.ChartTheme(index))

   @pyqtSlot(int)   ##图表动画
   def on_comboAnimation_currentIndexChanged(self,index):
      chart=self.__getCurrentChart()
      chart.setAnimationOptions(QChart.AnimationOption(index))


## ======page 1,  柱状图===================
   @pyqtSlot()   ##绘制柱状图
   def on_btnBuildBarChart_clicked(self):
      self.draw_barChart()

   @pyqtSlot()   ##绘制水平柱状图
   def on_btnBuildBarChartH_clicked(self):
      self.draw_barChart(False)
      
   def draw_barChart(self,isVertical=True):  ##绘制柱状图，或水平柱状图
      chart =self.ui.chartViewBar.chart()
      chart.removeAllSeries()          #删除所有序列
      chart.removeAxis(chart.axisX())  #删除坐标轴
      chart.removeAxis(chart.axisY())  #删除坐标轴
      if isVertical: 
         chart.setTitle("Barchart 演示")
         chart.legend().setAlignment(Qt.AlignBottom) 
      else:
         chart.setTitle("Horizontal Barchart 演示")
         chart.legend().setAlignment(Qt.AlignRight) 
         

      setMath = QBarSet("数学")  #QBarSet
      setChinese = QBarSet("语文")
      setEnglish= QBarSet("英语")

      seriesLine = QLineSeries() #QLineSeries序列用于显示平均分
      seriesLine.setName("平均分")
      pen=QPen(Qt.red)
      pen.setWidth(2)
      seriesLine.setPen(pen)

      seriesLine.setPointLabelsVisible(True)    #数据点标签可见
      if isVertical: 
         seriesLine.setPointLabelsFormat("@yPoint")   #显示y数值标签
      else:
         seriesLine.setPointLabelsFormat("@xPoint")   #显示x数值标签

      font=seriesLine.pointLabelsFont()
      font.setPointSize(10)
      font.setBold(True)
      seriesLine.setPointLabelsFont(font)  

      stud_Count=self.dataModel.rowCount()
      nameList=[]        #学生姓名列表,用于QBarCategoryAxis类坐标轴
      for i in range(stud_Count):      #从数据模型获取数据
         item=self.dataModel.item(i,self.COL_NAME)
         nameList.append(item.text())  #姓名,用作坐标轴标签

         item=self.dataModel.item(i,self.COL_MATH)
         setMath.append(float(item.text()))  #数学

         item=self.dataModel.item(i,self.COL_CHINESE)
         setChinese.append(float(item.text()))  #语文

         item=self.dataModel.item(i,self.COL_ENGLISH)
         setEnglish.append( float(item.text())) #英语
         
         item=self.dataModel.item(i,self.COL_AVERAGE)
         if isVertical:
            seriesLine.append(i,float(item.text())) #平均分,用于柱状图
         else:
            seriesLine.append(float(item.text()),i) #平均分，用于水平柱状图
            
      #创建一个序列 QBarSeries, 并添加三个数据集
      if isVertical:
         seriesBar = QBarSeries()   #柱状图
      else:
         seriesBar = QHorizontalBarSeries() #水平柱状图
         
      seriesBar.append(setMath)     #添加数据集
      seriesBar.append(setChinese)
      seriesBar.append(setEnglish)
      seriesBar.setLabelsVisible(True)    #数据点标签可见
      seriesBar.setLabelsFormat("@value") #显示数值标签
      seriesBar.setLabelsPosition(QAbstractBarSeries.LabelsCenter)   #数据标签显示位置
      seriesBar.hovered.connect(self.do_barSeries_Hovered)  #hovered信号
      seriesBar.clicked.connect(self.do_barSeries_Clicked)  #clicked信号

      chart.addSeries(seriesBar)    #添加柱状图序列
      chart.addSeries(seriesLine)   #添加折线图序列

      ##学生姓名坐标轴
      axisStud = QBarCategoryAxis()
      axisStud.append(nameList)  #添加横坐标文字列表
      axisStud.setRange(nameList[0], nameList[stud_Count-1]) #这只坐标轴范围
      
      #数值型坐标轴
      axisValue = QValueAxis()
      axisValue.setRange(0, 100)
      axisValue.setTitleText("分数")
      axisValue.setTickCount(6)  
      axisValue.applyNiceNumbers() 
      #    axisValue.setLabelFormat("%.0f") #标签格式
      #    axisY.setGridLineVisible(false)
      #    axisY.setMinorTickCount(4)
      if isVertical:  
         chart.setAxisX(axisStud, seriesBar)    #seriesBar
         chart.setAxisY(axisValue, seriesBar) 
         chart.setAxisX(axisStud, seriesLine)   #seriesLine
         chart.setAxisY(axisValue, seriesLine)
      else:
         chart.setAxisX(axisValue, seriesBar)   #seriesBar
         chart.setAxisY(axisStud, seriesBar) 
         chart.setAxisY(axisStud, seriesLine)   #seriesLine
         chart.setAxisX(axisValue, seriesLine)
         
      for marker in chart.legend().markers():  #QLegendMarker类型列表
         marker.clicked.connect(self.do_LegendMarkerClicked)


##=========page 2. StackedBar=========
   @pyqtSlot()   ## 绘制StackedBar
   def on_btnBuildStackedBar_clicked(self):
      self.draw_stackedBar()

   @pyqtSlot()   ## 绘制水平StackedBar
   def on_btnBuildStackedBarH_clicked(self):
      self.draw_stackedBar(False)

   def draw_stackedBar(self,isVertical=True):   #堆叠柱状图
      chart =self.ui.chartViewStackedBar.chart()
      chart.removeAllSeries()       #删除所有序列
      chart.removeAxis(chart.axisX())     #删除坐标轴
      chart.removeAxis(chart.axisY())
      if isVertical:    #堆叠柱状图
         chart.setTitle("StackedBar 演示")
         chart.legend().setAlignment(Qt.AlignBottom) 
      else:             #水平堆叠柱状图
         chart.setTitle("Horizontal StackedBar 演示")
         chart.legend().setAlignment(Qt.AlignRight) 

      ##创建三门课程的数据集
      setMath  =  QBarSet("数学")
      setChinese= QBarSet("语文")
      setEnglish= QBarSet("英语")

      stud_Count=self.dataModel.rowCount()
      nameList=[]             #学生姓名列表
      for i in range(stud_Count):   
         item=self.dataModel.item(i,self.COL_NAME)    #姓名
         nameList.append(item.text())

         item=self.dataModel.item(i,self.COL_MATH)    #数学
         setMath.append(float(item.text())) 

         item=self.dataModel.item(i,self.COL_CHINESE)  #语文
         setChinese.append(float(item.text()))

         item=self.dataModel.item(i,self.COL_ENGLISH)  #英语
         setEnglish.append(float(item.text())) 

      ##创建序列
      if isVertical:
         seriesBar = QStackedBarSeries()
      else:
         seriesBar = QHorizontalStackedBarSeries()
         
      seriesBar.append(setMath)
      seriesBar.append(setChinese)
      seriesBar.append(setEnglish)
      seriesBar.setLabelsVisible(True)     #显示每段的标签
      seriesBar.setLabelsFormat("@value")
      seriesBar.setLabelsPosition(QAbstractBarSeries.LabelsCenter)
      #  LabelsCenter,LabelsInsideEnd,LabelsInsideBase,LabelsOutsideEnd
      seriesBar.hovered.connect(self.do_barSeries_Hovered) #hovered信号
      seriesBar.clicked.connect(self.do_barSeries_Clicked) #clicked信号
      chart.addSeries(seriesBar)  

      axisStud =QBarCategoryAxis()  #类别坐标轴
      axisStud.append(nameList)
      axisStud.setRange(nameList[0], nameList[stud_Count-1])

      axisValue =QValueAxis()    #数值坐标轴
      axisValue.setRange(0, 300)
      axisValue.setTitleText("总分")
      axisValue.setTickCount(6)
      axisValue.applyNiceNumbers()

      if isVertical:
         chart.setAxisX(axisStud, seriesBar)
         chart.setAxisY(axisValue, seriesBar)
      else:
         chart.setAxisY(axisStud, seriesBar)
         chart.setAxisX(axisValue, seriesBar)
         
      for marker in chart.legend().markers():  #QLegendMarker类型列表
         marker.clicked.connect(self.do_LegendMarkerClicked)
      

##===========page 3. 百分比柱状图=============
   @pyqtSlot()   ##3.1  绘制 PercentBar
   def on_btnPercentBar_clicked(self):
      self.draw_percentBar()

   @pyqtSlot()   ##3.2  绘制 水平PercentBar
   def on_btnPercentBarH_clicked(self):
      self.draw_percentBar(False)

   def draw_percentBar(self,isVertical=True):
      chart =self.ui.chartViewPercentBar.chart()
      chart.removeAllSeries()
      chart.removeAxis(chart.axisX())
      chart.removeAxis(chart.axisY())
      chart.legend().setAlignment(Qt.AlignRight)   #AlignBottom,AlignTop
      if isVertical:
         chart.setTitle("PercentBar 演示")
      else:
         chart.setTitle(" Horizontal PercentBar 演示")

##创建数据集
      scoreBarSets=[]   #QBarSet对象列表
      sectionCount=5    #5个分数段，分数段是数据集
      for i in range(sectionCount): 
         item=self.ui.treeWidget.topLevelItem(i)
         barSet=QBarSet(item.text(0))  #一个分数段
         scoreBarSets.append(barSet)   #QBarSet对象列表

      categories=["数学","语文","英语"]
      courseCount=3   #3门课程
      for i in range(sectionCount):   #5个分数段，
         item=self.ui.treeWidget.topLevelItem(i)   #treeWidget第i行
         barSet=scoreBarSets[i]   #某个分数段的 QBarSet
         for j in range(courseCount):   #课程是category
            barSet.append(float(item.text(j+1)))
##创建序列
      if isVertical:
         seriesBar = QPercentBarSeries() #序列
      else:
         seriesBar = QHorizontalPercentBarSeries() #序列
      seriesBar.append(scoreBarSets)      #添加一个QBarSet对象列表
      seriesBar.setLabelsVisible(True)    #显示百分比
      seriesBar.hovered.connect(self.do_barSeries_Hovered) #hovered信号
      seriesBar.clicked.connect(self.do_barSeries_Clicked) #clicked信号
      chart.addSeries(seriesBar)
##创建坐标轴
      axisSection =  QBarCategoryAxis()   #分类坐标
      axisSection.append(categories)
      axisSection.setTitleText("分数段")
      axisSection.setRange(categories[0], categories[courseCount-1])

      axisValue =  QValueAxis()  #数值坐标
      axisValue.setRange(0, 100)
      axisValue.setTitleText("累积百分比")
      axisValue.setTickCount(6)
      axisValue.setLabelFormat("%.0f%")   #标签格式
      axisValue.applyNiceNumbers()
      
      if isVertical:
         chart.setAxisX(axisSection, seriesBar)
         chart.setAxisY(axisValue,   seriesBar)
      else:
         chart.setAxisY(axisSection, seriesBar)
         chart.setAxisX(axisValue,   seriesBar)

      for marker in chart.legend().markers():  #QLegendMarker类型列表
         marker.clicked.connect(self.do_LegendMarkerClicked)


##============page 4. 饼图 =====================
   @pyqtSlot(int)  ##选择课程
   def on_comboCourse_currentIndexChanged(self,index):
      self.draw_pieChart()
      
   @pyqtSlot()    ## 绘制饼图
   def on_btnDrawPieChart_clicked(self):
      self.draw_pieChart()

   @pyqtSlot(float)    ##设置 holeSize
   def on_spinHoleSize_valueChanged(self,arg1):
      seriesPie=self.ui.chartViewPie.chart().series()[0]
      seriesPie.setHoleSize(arg1)

   @pyqtSlot(float)   ##设置pieSize
   def on_spinPieSize_valueChanged(self,arg1):
      seriesPie=self.ui.chartViewPie.chart().series()[0]
      seriesPie.setPieSize(arg1)
      
   @pyqtSlot(bool)   ##显示图例checkbox
   def on_chkBox_PieLegend_clicked(self,checked):
      self.ui.chartViewPie.chart().legend().setVisible(checked)

   def draw_pieChart(self):  ##绘制饼图
      chart =self.ui.chartViewPie.chart() #获取chart对象
      chart.legend().setAlignment(Qt.AlignRight) #AlignRight,AlignBottom 
      chart.removeAllSeries()       #删除所有序列

      colNo=1+self.ui.comboCourse.currentIndex() #课程在treeWidget中的列号

      seriesPie =  QPieSeries()   #饼图序列
      seriesPie.setHoleSize(self.ui.spinHoleSize.value()) #饼图中间空心的大小
      sec_count=5    #分数段个数
      seriesPie.setLabelsVisible(True)    #只影响当前的slices，必须添加完slice之后再设置
      for i in range(sec_count):    #添加分块数据,5个分数段
         item=self.ui.treeWidget.topLevelItem(i)
         sliceLabel=item.text(0)+"(%s人)"%item.text(colNo)
         sliceValue=int(item.text(colNo))
         seriesPie.append(sliceLabel,sliceValue)   #添加一个饼图分块数据,(标签，数值)

      seriesPie.setLabelsVisible(True) #只影响当前的slices，必须添加完slice之后再设置
      seriesPie.hovered.connect(self.do_pieHovered)  #鼠标落在某个分块上时，此分块弹出
      chart.addSeries(seriesPie) 
      chart.setTitle("Piechart---"+self.ui.comboCourse.currentText())

           
##  =============自定义槽函数===============================        
   def do_calcuAverage(self,item):  ##计算平均分
      if (item.column()<self.COL_MATH or item.column()>self.COL_ENGLISH):
        return     #如果被修改的item不是数学、语文、英语,就退出
      
      rowNo=item.row() #获取数据的行编号
      avg=0.0
      for i in range(self.COL_MATH, 1+self.COL_ENGLISH):
         item=self.dataModel.item(rowNo,i)
         avg= avg+float(item.text())
      avg=avg/3.0          #计算平均分
      item=self.dataModel.item(rowNo,self.COL_AVERAGE)   #获取平均分数据的item
      item.setText("%.1f"%avg)   #更新平均分数据


   def do_pieHovered(self,pieSlice,state):   ##鼠标在饼图上移入移出
      pieSlice.setExploded(state)  #弹出或缩回，具有动态效果
      if state:  #显示带百分数的标签
         self.__oldLabel=pieSlice.label()  #保存原来的Label
         pieSlice.setLabel(self.__oldLabel+": %.1f%%"   
                        %(pieSlice.percentage()*100) )
      else:    #显示原来的标签
         pieSlice.setLabel(self.__oldLabel)
      

   def do_barSeries_Hovered(self,status,  index, barset): ##关联hovered信号
      hint="hovered barSet="+barset.label()
      if status:
         hint=hint+", index=%d, value=%.2f"%(index, barset.at(index))
      else:
         hint=""
      self.ui.statusBar.showMessage(hint)

   def do_barSeries_Clicked(self, index, barset):  ##关联clicked信号
      hint="clicked barSet="+barset.label()
      hint=hint+", count=%d, sum=%.2f"%(barset.count(), barset.sum())
      self.ui.statusBar.showMessage(hint)

   def do_LegendMarkerClicked(self):   ##图例单击
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

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
