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
      pass


##  ==============自定义功能函数========================
   def __createChart(self):
      pass
      

   def __prepareData(self):  ##为序列设置数据点
      pass


   def __updateFromChart(self):
      pass
     
      
##  ==============event处理函数==========================


##  ==========由connectSlotsByName()自动连接的槽函数============        
      
##========工具栏上的几个按钮的Actions==============
   @pyqtSlot()  ## "刷新绘图" 工具栏按钮
   def on_actDraw_triggered(self):
      pass


##=====ToolBox 第1组：==="图表设置" 分组里的功能================
##=======1.1 标题========
   @pyqtSlot()   ##设置标题文字
   def on_btnTitleSetText_clicked(self):
      pass
        
   @pyqtSlot()   ##设置标题文字颜色
   def on_btnTitleColor_clicked(self):
      pass

   @pyqtSlot()   ##设置标题字体
   def on_btnTitleFont_clicked(self):
      pass

##=======1.2 图例==========
   @pyqtSlot(bool)   ##图例是否可见
   def on_groupBox_Legend_clicked(self,checked):
      pass

   @pyqtSlot()   ##图例的位置, 上
   def on_radioButton_clicked(self):
      pass
         
   @pyqtSlot()   ##图例的位置，下
   def on_radioButton_2_clicked(self):
      pass
      
   @pyqtSlot()   ##图例的位置，左
   def on_radioButton_3_clicked(self):
      pass

   @pyqtSlot()   ##图例的位置，右
   def on_radioButton_4_clicked(self):
      pass

   @pyqtSlot()   ##图例的文字颜色
   def on_btnLegendlabelColor_clicked(self):
      pass

   @pyqtSlot()   ##图例的字体
   def on_btnLegendFont_clicked(self):
      pass

##=======1.3 边距========
   @pyqtSlot()   ##设置图表的4个边距
   def on_btnSetMargin_clicked(self):
      pass

##=======1.4 动画效果========        
   @pyqtSlot(int)   ##动画效果
   def on_comboAnimation_currentIndexChanged(self,index):
      pass

   @pyqtSlot(int)   ##图表的主题
   def on_comboTheme_currentIndexChanged(self,index):
      pass


##=====ToolBox 第2组：==="曲线设置" 分组里的功能================
##=======2.1 选择操作序列========
   @pyqtSlot()   ##获取当前数据序列,sin
   def on_radioSeries0_clicked(self):
      pass


   @pyqtSlot()   ##获取当前数据序列,cos
   def on_radioSeries1_clicked(self):
      pass

      
##======2.2 序列曲线 设置======== 
   @pyqtSlot(bool)   ##序列是否可见
   def on_groupBox_Series_clicked(self,checked):
      pass

   @pyqtSlot()   ##设置序列名称
   def on_btnSeriesName_clicked(self):
      pass


   @pyqtSlot()   ##序列的曲线颜色
   def on_btnSeriesColor_clicked(self):
      pass


   @pyqtSlot()   ##序列曲线的Pen设置
   def on_btnSeriesPen_clicked(self):
      pass


   @pyqtSlot(bool)   ##序列的数据点是否可见,数据点形状是固定的
   def on_chkBoxPointVisible_clicked(self,checked):
      pass

   @pyqtSlot(bool)   ##使用openGL加速后，不能设置线型，不能显示数据点
   def on_chkkBoxUseOpenGL_clicked(self,checked):
      pass

   @pyqtSlot(int)    ##序列的透明度
   def on_sliderOpacity_sliderMoved(self,position):
      pass

##=======2.3 数据点标签 ========     
   @pyqtSlot(bool)   ##数据点标签 groupBox
   def on_groupBox_PointLabel_clicked(self,checked):
      pass

   @pyqtSlot()   ##序列数据点标签颜色
   def on_btnSeriesLabColor_clicked(self):
      pass


   @pyqtSlot()   ##序列数据点标签字体
   def on_btnSeriesLabFont_clicked(self):
      pass

   @pyqtSlot()   ##序列数据点标签的显示格式
   def on_radioSeriesLabFormat0_clicked(self):
      pass
         
   @pyqtSlot()   ##序列数据点标签的显示格式
   def on_radioSeriesLabFormat1_clicked(self):
      pass

##=====ToolBox 第3组：==="坐标轴设置" 分组里的功能================

##=======3.1 选择操作的坐标轴对象=======
   @pyqtSlot()   ##选择坐标轴X
   def on_radioAxisX_clicked(self):
      pass


   @pyqtSlot()   ##选择坐标轴Y
   def on_radioAxisY_clicked(self):
      pass

##======3.2 坐标轴可见性和范围=======
   @pyqtSlot(bool)   ##坐标轴可见性
   def on_groupBox_Axis_clicked(self,checked):
      pass

   @pyqtSlot(bool)   ##坐标反向
   def on_chkBoxAxisReverse_clicked(self,checked):
      pass
      
   @pyqtSlot()    ##设置坐标范围
   def on_btnSetAxisRange_clicked(self):
      pass
      
##======3.3 轴标题=======
   @pyqtSlot(bool)   ##坐标轴标题可见性
   def on_groupBox_AxisTitle_clicked(self,checked):
      pass

   @pyqtSlot()   ##设置轴标题
   def on_btnAxisSetTitle_clicked(self):
      pass

   @pyqtSlot()   ##设置轴标题的颜色
   def on_btnAxisTitleColor_clicked(self):
      pass

   @pyqtSlot()   ##设置轴标题的字体
   def on_btnAxisTitleFont_clicked(self):
      pass

##======3.4 轴刻度标签=======
   @pyqtSlot(bool)   ##可见性
   def on_groupBox_AxisLabel_clicked(self,checked):
      pass
         
   @pyqtSlot()   ##设置标签格式
   def on_btnAxisLabelFormat_clicked(self):
      pass
      
   @pyqtSlot()   ##设置标签文字颜色
   def on_btnAxisLabelColor_clicked(self):
      pass

   @pyqtSlot()   ##设置标签字体
   def on_btnAxisLabelFont_clicked(self):
      pass

##======3.5 轴线和主刻度=========
   @pyqtSlot(bool)   ##可见性
   def on_groupBox_Ticks_clicked(self,checked):
      pass

   @pyqtSlot(int)   ##主刻度个数
   def on_spinTickCount_valueChanged(self,arg1):
      pass

   @pyqtSlot()   ##设置线条Pen
   def on_btnAxisLinePen_clicked(self):
      pass
         
   @pyqtSlot()   ##设置线条颜色
   def on_btnAxisLinePenColor_clicked(self):
      pass
         
##======3.6 主网格线=========
   @pyqtSlot(bool)   ##可见性
   def on_groupBox_GridLine_clicked(self,checked):
      pass
         

   @pyqtSlot()   ##设置线条Pen
   def on_btnGridLinePen_clicked(self):
      pass

   @pyqtSlot()   ##设置线条颜色
   def on_btnGridLineColor_clicked(self):
      pass

##======3.7 次网格线=======
   @pyqtSlot(bool)   ##可见性
   def on_groupBox_MinorGrid_clicked(self,checked):
      pass

   @pyqtSlot(int)    ##次刻度个数
   def on_spinMinorTickCount_valueChanged(self,arg1):
      pass
         
   @pyqtSlot()    ##设置线条Pen
   def on_btnMinorPen_clicked(self):
      pass
         
   @pyqtSlot()    ##设置线条颜色
   def on_btnMinorColor_clicked(self):
      pass
         
        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
