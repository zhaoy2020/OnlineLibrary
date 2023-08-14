# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow

from PyQt5.QtCore import  pyqtSlot,Qt

import numpy as np

import matplotlib as mpl

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo14_4, 几种常见二维图表")
      self.setCentralWidget(self.ui.tabWidget)
      pass

##  ==============自定义功能函数========================
   def __drawHist(self,pointCount=2000, binsCount=40):  #绘制统计直方图
      pass


   def __drawFill(self):  ##绘制填充图
      pass


   def __drawPie(self):  ##绘制饼图
      pass


   def __drawStem(self):  ##绘制火柴杆图
      pass

      
##  ==============event处理函数==========================


##  ==========由connectSlotsByName()自动连接的槽函数============
      

##===========page 1 直方图============
   @pyqtSlot(bool)   ##显示工具栏
   def on_gBoxHist_toolbar_clicked(self,checked):
      pass

      
   @pyqtSlot(bool)   ## 显示坐标提示
   def on_chkBoxHist_ShowHint_clicked(self,checked):
      pass

      
   @pyqtSlot()    ## 紧凑布局
   def on_btnHist_tightLayout_clicked(self):
      pass


   @pyqtSlot()    ## 重画图表
   def on_btnHist_redraw_clicked(self):
      pass


   @pyqtSlot(bool)   ##显示图例
   def on_chkBoxHist_Legend_clicked(self,checked):
      pass


##=======2.填充图=========
   @pyqtSlot()   ##曲线与0之间填充
   def on_radioFill_Both_clicked(self):
      pass

      
   @pyqtSlot()   ##填充y>=0的部分
   def on_radioFill_Up_clicked(self):
      pass


   @pyqtSlot()   ##填充y<=0的部分
   def on_radioFill_Down_clicked(self):
      pass

      
   @pyqtSlot()   ##紧凑布局
   def on_btnFill_tightLayout_clicked(self):
      pass


   @pyqtSlot(bool)   ##显示网格线
   def on_chkBoxFill_gridLine_clicked(self,checked):
      pass


##=======3.饼图=========
   @pyqtSlot()   ## 重画饼图
   def on_btnPie_redraw_clicked(self):
      pass


   @pyqtSlot()   ## 紧凑布局
   def on_btnPie_tightLayout_clicked(self):
      pass

      
   @pyqtSlot(bool)   ## 显示图例
   def on_chkBoxPie_Legend_clicked(self,checked):
      pass

      
##=====4.火柴杆图==============
   @pyqtSlot()   ## 重画曲线
   def on_btnStem_redraw_clicked(self):
      pass

      
   @pyqtSlot()   ##紧凑布局
   def on_btnStem_tightLayout_clicked(self):
      pass


   @pyqtSlot(bool)   ## 显示连续信号
   def on_chkBoxStem_Analog_clicked(self,checked):
      pass

      
   @pyqtSlot(bool)   ## 显示采样保持信号
   def on_chkBoxStem_Holder_clicked(self,checked):
      pass


   @pyqtSlot(bool)   ## 显示图例
   def on_chkBoxStem_Legend_clicked(self,checked):
      pass


##=====5.极坐标图========
   def __drawPolarSpiral(self):  ##绘制螺旋线
      pass

     
   @pyqtSlot()    ## 重画曲线
   def on_btnPolar_redraw_clicked(self):
      pass


   @pyqtSlot(bool)   ## 逆时针方向
   def on_chkBoxPolar_direction_clicked(self,checked):
      pass


   @pyqtSlot(bool)   ## 显示网格
   def on_chkBoxPolar_gridOn_clicked(self,checked):
      pass


   @pyqtSlot(int)    ## 角度偏移量
   def on_spinPolar_offset_valueChanged(self,arg1):
      pass

      
   @pyqtSlot()   ## 紧凑布局
   def on_btnPolar_tightLayout_clicked(self):
      pass


   @pyqtSlot()   ## 旋转
   def on_btnPolar_rotate_clicked(self):
      pass


##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
