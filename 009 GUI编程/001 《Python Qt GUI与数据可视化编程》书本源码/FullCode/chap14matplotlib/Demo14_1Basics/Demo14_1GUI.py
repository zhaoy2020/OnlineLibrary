# -*- coding: utf-8 -*-
## 程序文件： Demo14_1GUI.py
## 使用matplotlib 面向对象方法在GUI中绘图

import sys

import numpy as np

import matplotlib as mpl

##import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import (FigureCanvas, 
                 NavigationToolbar2QT as NavigationToolbar)

## FigureCanvas 的父类是QWidget，是Figure的容器类
## NavigationToolbar 是图表上的工具栏

##from matplotlib.figure import Figure   #创建plot窗口的类
## Figure是图表类

from PyQt5.QtWidgets import  QApplication, QMainWindow

from PyQt5.QtCore import  Qt

##from PyQt5.QtWidgets import  QVBoxLayout

##from PyQt5.QtGui import

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数

      self.setWindowTitle("Demo14_1, GUI中的matplotlib绘图")

## rcParams[]参数设置，以正确显示汉字
      mpl.rcParams['font.sans-serif']=['KaiTi','SimHei']    #汉字字体
##                                       'Times New Roman']   
      mpl.rcParams['font.size']=12   #字体大小
##  华文宋体：STSong 华文仿宋：STFangsong 华文黑体：STHeiti
##      黑体：SimHei 仿宋：FangSong 楷体：KaiTi 
      mpl.rcParams['axes.unicode_minus'] =False    #正常显示符号

      self.__iniFigure()    # 创建绘图系统，初始化窗口
      self.__drawFigure()   # 绘图

##==========自定义函数=================
   def __iniFigure(self):  ##创建绘图系统，初始化窗口
      self.__fig=mpl.figure.Figure(figsize=(8, 5))    #单位英寸
      self.__fig.suptitle("plot in GUI application")  #总的图标题
      figCanvas = FigureCanvas(self.__fig)  #创建FigureCanvas对象，必须传递一个Figure对象

## 下面这两行语句是行不通的      
##      figCanvas = FigureCanvas()  #创建FigureCanvas对象，必须传递一个Figure对象
##      self.__fig.set_canvas(figCanvas)
      
      naviToolbar=NavigationToolbar(figCanvas, self)  #创建工具栏
      naviToolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)  #ToolButtonTextUnderIcon,ToolButtonTextBesideIcon

      self.addToolBar(naviToolbar)  #添加工具栏到主窗口
      self.setCentralWidget(figCanvas)   
      
   def __drawFigure(self):  ## 绘图
      t = np.linspace(0, 10, 40)
      y1=np.sin(t)
      y2=np.cos(2*t)

      ax1=self.__fig.add_subplot(1,2,1)   #添加子图，ax1是 matplotlib.axes.Axes 类对象
      ax1.plot(t,y1,'r-o',label="sin", linewidth=1, markersize=5)    #绘制一条曲线
      ax1.plot(t,y2,'b:',label="cos",linewidth=2)           #绘制一条曲线
      ax1.set_xlabel('X 轴')               # X轴标题
      ax1.set_ylabel('Y 轴',fontsize=14)   # Y轴标题
      ax1.set_xlim([0,10])      # X轴范围 
      ax1.set_ylim([-1.5,1.5])  # Y轴范围 
      ax1.set_title("曲线")     # 子图标题
      ax1.legend()              # 自动创建图例

      ax2=self.__fig.add_subplot(1,2,2)   #添加子图，ax2是 matplotlib.axes.Axes 类对象
      week=["Mon","Tue","Wed","Thur","Fri","Sat","Sun"] 
      sales=np.random.randint(200,400,7)
      ax2.bar(week,sales)           # 绘制柱状图
      ax2.set_xlabel('week days')   # X轴标题
      ax2.set_ylabel('参观人数')    # Y轴标题
      ax2.set_title("柱状图")       # 子图标题

##      self.__fig.set_tight_layout(True) #紧凑布局


##  ==============自定义功能函数========================


##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
        
        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   form=QmyMainWindow()             #创建窗体
   form.show()
   sys.exit(app.exec_())
