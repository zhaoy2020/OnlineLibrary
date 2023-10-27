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
      pass


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

      
   def __surveyData(self):    ##统计各分数段人数
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
      pass

      
   def __iniStackedBar(self):    ##初始化堆叠柱状图
      pass


   def __iniPercentBar(self):    ##百分比柱状图初始化
      pass


   def __iniPieChart(self):   ##饼图初始化
      pass
   

   def __getCurrentChart(self):  ##获取当前QChart对象
      pass

      
##  ==============event事件处理函数==========================

        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
## 工具栏按钮的功能
   @pyqtSlot()   ##重新生成数据
   def on_toolBtn_GenData_clicked(self):
      pass

   @pyqtSlot()   ##重新统计
   def on_toolBtn_Counting_clicked(self):
      pass
      
      
   @pyqtSlot(int)   ##设置图表主题
   def on_comboTheme_currentIndexChanged(self,index):
      pass


   @pyqtSlot(int)   ##图表动画
   def on_comboAnimation_currentIndexChanged(self,index):
      pass


## ======page 1,  柱状图===================
   @pyqtSlot()   ##绘制柱状图
   def on_btnBuildBarChart_clicked(self):
      self.draw_barChart()

   @pyqtSlot()   ##绘制水平柱状图
   def on_btnBuildBarChartH_clicked(self):
      self.draw_barChart(False)
      
   def draw_barChart(self,isVertical=True):  ##绘制柱状图，或水平柱状图
      pass


##=========page 2. StackedBar=========
   @pyqtSlot()   ## 绘制StackedBar
   def on_btnBuildStackedBar_clicked(self):
      self.draw_stackedBar()

   @pyqtSlot()   ## 绘制水平StackedBar
   def on_btnBuildStackedBarH_clicked(self):
      self.draw_stackedBar(False)

   def draw_stackedBar(self,isVertical=True):   #堆叠柱状图
      pass
      

##===========page 3. 百分比柱状图=============
   @pyqtSlot()   ##3.1  绘制 PercentBar
   def on_btnPercentBar_clicked(self):
      self.draw_percentBar()

   @pyqtSlot()   ##3.2  绘制 水平PercentBar
   def on_btnPercentBarH_clicked(self):
      self.draw_percentBar(False)

   def draw_percentBar(self,isVertical=True):
      pass


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
      pass

           
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
      pass


   def do_barSeries_Hovered(self,status,  index, barset): ##关联hovered信号
      pass


   def do_barSeries_Clicked(self, index, barset):  ##关联clicked信号
      pass


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
