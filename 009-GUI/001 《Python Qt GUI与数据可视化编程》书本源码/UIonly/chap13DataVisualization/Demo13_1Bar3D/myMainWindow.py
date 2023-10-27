# -*- coding: utf-8 -*-

import sys,random

from PyQt5.QtWidgets import  (QApplication, QMainWindow, QWidget,
                  QSplitter, QColorDialog,QInputDialog)

from PyQt5.QtCore import  pyqtSlot,Qt


from PyQt5.QtGui import QVector3D


from PyQt5.QtDataVisualization import *


from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo13_1, Q3DBars和QBar3DSeries绘制三维柱状图")
      pass
      

##  ==============自定义功能函数========================

   def __iniGraph3D(self):    ##创建3D图表
      pass
   

##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        

##===工具栏actions==
   @pyqtSlot()   ## "序列基本颜色"
   def on_actSeries_BaseColor_triggered(self):
      pass


   @pyqtSlot()   ## "修改数值"
   def on_actBar_ChangeValue_triggered(self):
      pass

      
   @pyqtSlot()   ## "添加行"
   def on_actData_Add_triggered(self):
      pass


   @pyqtSlot()   ## "插入行"
   def on_actData_Insert_triggered(self):
      pass


   @pyqtSlot()   ## "删除行"
   def on_actData_Delete_triggered(self):
      pass
         
   
##===三维旋转===      
   @pyqtSlot(int)   ## "预设视角"
   def on_comboCamera_currentIndexChanged(self,index):
      pass

      
   @pyqtSlot(int)   ## "水平旋转"
   def on_sliderH_valueChanged(self,value):
      pass


   @pyqtSlot(int)   ## "垂直旋转"
   def on_sliderV_valueChanged(self,value):
      pass


   @pyqtSlot(int)   ## "缩放"
   def on_sliderZoom_valueChanged(self,value):
      pass


   @pyqtSlot()   ## 复位到FrontHigh视角
   def on_btnResetCamera_clicked(self):
      pass

      
   @pyqtSlot()   ## "左移"
   def on_btnMoveLeft_clicked(self):
      pass


   @pyqtSlot()   ## "右移"
   def on_btnMoveRight_clicked(self):
      pass


   @pyqtSlot()   ## "上移"
   def on_btnMoveUp_clicked(self):
      pass


   @pyqtSlot()   ## "下移"
   def on_btnMoveDown_clicked(self):
      pass
      

##====图表总体      
   @pyqtSlot(int)   ## "主题"
   def on_cBoxTheme_currentIndexChanged(self,index):
      pass

      
   @pyqtSlot(int)   ## "字体大小"
   def on_spinFontSize_valueChanged(self,arg1):
      pass


   @pyqtSlot(int)   ## "选择模式"
   def on_cBoxSelectionMode_currentIndexChanged(self,index):
      pass

      
   @pyqtSlot(bool)   ## "显示背景"
   def on_chkBoxBackground_clicked(self,checked):
      pass

         
   @pyqtSlot(bool)   ## "显示背景的网格"
   def on_chkBoxGrid_clicked(self,checked):
      pass


   @pyqtSlot(bool)   ## "数值坐标轴反向"
   def on_chkBoxReverse_clicked(self,checked):
      pass


   @pyqtSlot(bool)   ## "显示倒影"
   def on_chkBoxReflection_clicked(self,checked):
      pass

      
   @pyqtSlot(bool)   ## "轴标题可见"
   def on_chkBoxAxisTitle_clicked(self,checked):
      pass


   @pyqtSlot(bool)   ## "轴标签背景可见"
   def on_chkBoxAxisBackground_clicked(self,checked):
      pass

##===序列设置
   @pyqtSlot(int)   ## "棒柱样式"
   def on_cBoxBarStyle_currentIndexChanged(self,index):
      pass

      
   @pyqtSlot(bool)   ## "光滑效果"
   def on_chkBoxSmooth_clicked(self,checked):
      pass
   

   @pyqtSlot(bool)   ## "选中棒柱的标签可见"
   def on_chkBoxItemLabel_clicked(self,checked):
      pass

        
##  =============自定义槽函数===============================        
   def do_barSelected(self,position):   ##选择一个棒柱时触发
      pass

      

##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
