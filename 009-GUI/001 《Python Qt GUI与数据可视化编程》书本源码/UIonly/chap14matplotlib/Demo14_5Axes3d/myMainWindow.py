# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QLabel

from PyQt5.QtCore import  pyqtSlot,Qt

import numpy as np

import matplotlib as mpl

from mpl_toolkits.mplot3d import axes3d

from ui_MainWindow import Ui_MainWindow


class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo14_5, 三维数据绘图")
      pass
   
      
##  ==============自定义功能函数========================
   def __iniUI(self):
      pass
      
      
   def __generateData(self): #生成数据
      divCount=self.ui.spinDivCount.value()  #划分网格个数
      x=np.linspace(-5, 5, divCount, endpoint=True)
      y=np.linspace(-5, 5, divCount, endpoint=True)
      
      x,y= np.meshgrid(x, y) # 二维网格化数组
      p11=3*((1-x)**2)
      p12=np.exp(-x**2-(y+1)**2)
      p1=p11*p12     #按元素相乘

      p21=x/5-x**3-y**5
      p22=np.exp(-x**2-y**2)
      p2=-10*p21*p22

      p31=np.exp(-(x+1)**2-y**2)
      p3=-p31/3

      self._Z=p1+p2+p3  # Z数据
      self._X=x         # X数据
      self._Y=y         # Y数据

   def __iniFigure(self):  ##初始化图表，创建子图
      pass


##  ==============event处理函数==========================


##  ==========由connectSlotsByName()自动连接的槽函数============


##=========图表操作

   @pyqtSlot()      ## 十字光标
   def on_actSetCursor_triggered(self):  
      pass

   @pyqtSlot()      ##重新生成数据并绘图
   def on_btnRefreshData_clicked(self):  
      pass


##===== 3D绘图设置      
   @pyqtSlot(bool)      ## Z轴反向
   def on_chkBox3D_invertZ_clicked(self,checked):  
      pass


   @pyqtSlot(bool)      ## 显示网格
   def on_chkBox3D_gridOn_clicked(self,checked):  
      pass


   @pyqtSlot(bool)      ## 显示坐标轴
   def on_chkBox3D_axisOn_clicked(self,checked):  
      pass


   @pyqtSlot(int)      ## 3D绘图类型
   def on_combo3D_type_currentIndexChanged(self,index):  #
      pass
      

##==========2D绘图
   @pyqtSlot(int)      ## 2D绘图类型
   def on_combo2D_type_currentIndexChanged(self,index):  #
      pass


##  =================自定义槽函数==========
   @pyqtSlot(str)  ##在ComboBox中选择了colormap
   def do_comboColormap_Changed(self,arg1):
      pass


   def do_canvas_mouseMove(self,event): ##鼠标移动
      pass

      
   def do_canvas_pick(self,event):  ##拾取对象
      pass
    
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
