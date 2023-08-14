# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow

from PyQt5.QtCore import  pyqtSlot,Qt

##from PyQt5.QtWidgets import   QLabel

##from PyQt5.QtGui import QColor

import numpy as np

import matplotlib as mpl


##from matplotlib.ticker import NullFormatter

##import matplotlib.pyplot as plt

##from  matplotlib.backends.backend_qt5agg import (
##           FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

##from matplotlib.figure import Figure  # 创建plot窗口的类


from ui_MainWindow import Ui_MainWindow


class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo14_4, 几种常见二维图表")
      self.setCentralWidget(self.ui.tabWidget)

##  黑体：SimHei 宋体：SimSun 新宋体：NSimSun 仿宋：FangSong  楷体：KaiTi 
      mpl.rcParams['font.sans-serif']=['SimHei'] 
      mpl.rcParams['font.size']=9         #显示汉字
      mpl.rcParams['axes.unicode_minus'] =False    #减号unicode编码

      self.__drawHist()  #直方图
      self.__drawFill()  #曲线填充
      self.__drawPie()   #饼图
      self.__drawStem()  #火柴杆图
      self.__drawPolarSpiral()  #螺旋线


##  ==============自定义功能函数========================
     
   def __drawHist(self,pointCount=2000, binsCount=40):  #绘制统计直方图
      x=range(pointCount)
      y=np.random.randn(pointCount)       #标准正态分布随机数

      self.ui.widgetHist.figure.clear()   #清除图表

      ax1=self.ui.widgetHist.figure.add_subplot(2,1,1,label="points")
      ax1.scatter(x,y,marker=".")
      ax1.set_title("标准正态分布随机数")
      

      ax2=self.ui.widgetHist.figure.add_subplot(2,1,2,label="histogram")
      M, bins, patches=ax2.hist(y,bins=binsCount,density=True,label="直方图")
      ax2.set_title("统计直方图")
      ax2.set_xlabel('数值')   
      ax2.set_ylabel('概率密度')  

      dens = np.exp(-0.5*(bins**2))/np.sqrt(2*np.pi)  #理论概率密度曲线
      ax2.plot(bins,dens,"--r",label="概率密度曲线")
      leg=ax2.legend()
      leg.set_visible(self.ui.chkBoxHist_Legend.isChecked())

   def __drawFill(self):  ##绘制填充图
      xmax=5
      x = np.linspace(0.0, xmax, 200)
      y = np.cos( 2*np.pi * x) * np.exp(-x)
      self.ui.widgetFill.figure.clear()
      ax1=self.ui.widgetFill.figure.add_subplot(1,1,1)
      ax1.plot(x,y,'k-')   #绘制曲线

      if self.ui.radioFill_Both.isChecked():
         ax1.fill_between(x,0,y,facecolor='g')
      elif self.ui.radioFill_Up.isChecked():
         ax1.fill_between(x,0,y,where=y>=0,facecolor='g')
      elif self.ui.radioFill_Down.isChecked():
         ax1.fill_between(x,0,y,where=y<=0,facecolor='g')

      ax1.set_xlim(0,xmax)  
      ax1.set_ylim(-1,1)  
      ax1.set_title("曲线之间填充")
      ax1.set_xlabel('时间(sec)')  
      ax1.set_ylabel('响应幅度')  

      checked=self.ui.chkBoxFill_gridLine.isChecked()
      ax1.grid(b=checked,which='major',axis='both')
      

   def __drawPie(self):  ##绘制饼图
      days=("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
      sliceCount=7      #饼图分块个数
      sales=np.random.randint(50,400,sliceCount)   #50-200, 7个数，销量数据
      self.ui.widgetPie.figure.clear()
      
      exploded =np.zeros(sliceCount)   #具有弹出效果的块
      index=self.ui.comboPie_explode.currentIndex()
      if index<sliceCount:
         exploded[index]=0.1

      holeSize=self.ui.spinPie_HoleSize.value()    #空心圆大小
      ax1=self.ui.widgetPie.figure.add_subplot(1,1,1)
      wedges, texts, autotexts=ax1.pie(sales,labels=days,
                  explode=exploded,wedgeprops=dict(width=1-holeSize),#textprops=dict(fontsize=12),
                  autopct='%.1f%%',shadow=True)

   ##      print (texts)         #标签文字Text列表
   ##      print (autotexts)     #自动生成的文字Text列表
      ax1.set_title("一周每日销量占比")
      ax1.axis('equal')    #等宽坐标轴

##      ax1.legend() # 只有labels标签

      handles, labels = ax1.get_legend_handles_labels()  #获得图例数据
      for i in range(sliceCount):
         lab="%s--%d"%(labels[i],sales[i])
         labels[i]=lab     #改变图例标签
      leg=ax1.legend(handles, labels,loc="upper right")
      leg.set_draggable(True)
      leg.set_visible(self.ui.chkBoxPie_Legend.isChecked())


   def __drawStem(self):  ##绘制火柴杆图
      t=np.linspace(0, 4*np.pi, 200)   #高密度采样来模拟连续信号
      wn=1
      y=np.sin(wn*t)    #模拟正弦信号

      pointCount=self.ui.spinStem_PointCount.value()  #离散采样点数
      t2=np.linspace(0, 4*np.pi, pointCount)
      y2=np.sin(wn*t2)  #离散采样点
      
      self.ui.widgetStem.figure.clear()
      ax1=self.ui.widgetStem.figure.add_subplot(1,1,1)
      isVis=self.ui.chkBoxStem_Analog.isChecked()
      ax1.plot(t,y,"b:",label="连续信号",visible=isVis)   #绘制连续信号
      ax1.plot(t2,y2,"k-",drawstyle='steps-post',label="采样保持信号",
               visible=self.ui.chkBoxStem_Holder.isChecked())  #{'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}
   ##      markerline, stemlines, baseline = ax1.stem(t2, y2, '--',label="采样点")      
      ax1.stem(t2, y2, '--',label="采样点")    #火柴杆图

##      ax1.set_xlim(0,4*np.pi)

      ax1.set_title("信号采样与保持示意图")
      ax1.set_xlabel('时间(sec)')   
      ax1.set_ylabel('信号幅度')  

      leg=ax1.legend()
      leg.set_draggable(True)
      leg.set_visible(self.ui.chkBoxStem_Legend.isChecked())

      
##  ==============event处理函数==========================


##  ==========由connectSlotsByName()自动连接的槽函数============

      

   ##===========page 1 直方图============
   @pyqtSlot(bool)   ##显示工具栏
   def on_gBoxHist_toolbar_clicked(self,checked):
      self.ui.widgetHist.setToolbarVisible(checked)
      
   @pyqtSlot(bool)   ## 显示坐标提示
   def on_chkBoxHist_ShowHint_clicked(self,checked):
      self.ui.widgetHist.setDataHintVisible(checked)

      
   @pyqtSlot()    ## 紧凑布局
   def on_btnHist_tightLayout_clicked(self):
      self.ui.widgetHist.figure.tight_layout()  # 对所有子图 进行一次tight_layout
      self.ui.widgetHist.redraw()   #刷新


   @pyqtSlot()   ## 重画图表
   def on_btnHist_redraw_clicked(self):
      pointCount=self.ui.spinHist_PointCount.value()
      binsCount=self.ui.spinHist_binsCount.value()
      self.__drawHist(pointCount, binsCount)
      self.ui.widgetHist.redraw()   

   @pyqtSlot(bool)   ##显示图例
   def on_chkBoxHist_Legend_clicked(self,checked):
      axesList=self.ui.widgetHist.figure.axes   #子图列表
      leg=axesList[1].get_legend()
      leg.set_visible(checked)
      self.ui.widgetHist.redraw()
      

   ##=======2.填充图=========
   @pyqtSlot()   ##曲线与0之间填充
   def on_radioFill_Both_clicked(self):
      self.__drawFill()
      self.ui.widgetFill.redraw()
      
   @pyqtSlot()   ##填充y>=0的部分
   def on_radioFill_Up_clicked(self):
      self.__drawFill()
      self.ui.widgetFill.redraw()

   @pyqtSlot()   ##填充y<=0的部分
   def on_radioFill_Down_clicked(self):
      self.__drawFill()
      self.ui.widgetFill.redraw()
      
   @pyqtSlot()   ##紧凑布局
   def on_btnFill_tightLayout_clicked(self):
      self.ui.widgetFill.figure.tight_layout()
      self.ui.widgetFill.redraw()

   @pyqtSlot(bool)   ##显示网格线
   def on_chkBoxFill_gridLine_clicked(self,checked):
      ax=self.ui.widgetFill.figure.axes[0]
      ax.grid(b=checked,which='major',axis='both')
   ##      which : {'major', 'minor', 'both'}
   ##      axis : {'both', 'x', 'y'}
      self.ui.widgetFill.redraw()
      

   ##=======3.饼图=========
   @pyqtSlot()   ## 重画饼图
   def on_btnPie_redraw_clicked(self):
      self.__drawPie()
      self.ui.widgetPie.redraw()

   @pyqtSlot()   ## 紧凑布局
   def on_btnPie_tightLayout_clicked(self):
      self.ui.widgetPie.figure.tight_layout()
      self.ui.widgetPie.redraw()
      
   @pyqtSlot(bool)   ## 显示图例
   def on_chkBoxPie_Legend_clicked(self,checked):
      axesList=self.ui.widgetPie.figure.axes    #子图列表
      leg=axesList[0].get_legend()
      leg.set_visible(checked)
      self.ui.widgetPie.redraw()

      
   ##=====4.火柴杆图==============
   @pyqtSlot()   ## 重画曲线
   def on_btnStem_redraw_clicked(self):
      self.__drawStem()
      self.ui.widgetStem.redraw()
      
   @pyqtSlot()   ##紧凑布局
   def on_btnStem_tightLayout_clicked(self):
      self.ui.widgetStem.figure.tight_layout()
      self.ui.widgetStem.redraw()

   @pyqtSlot(bool)   ## 显示连续信号
   def on_chkBoxStem_Analog_clicked(self,checked):
      axesList=self.ui.widgetStem.figure.axes   #子图列表
      line=axesList[0].lines[0]  #连续信号曲线
      line.set_visible(checked)
      self.ui.widgetStem.redraw()
      
   @pyqtSlot(bool)   ## 显示采样保持信号
   def on_chkBoxStem_Holder_clicked(self,checked):
      axesList=self.ui.widgetStem.figure.axes      #子图列表
      line=axesList[0].lines[1]     #采样保持信号曲线
      line.set_visible(checked)
      self.ui.widgetStem.redraw()

   @pyqtSlot(bool)   ## 显示图例
   def on_chkBoxStem_Legend_clicked(self,checked):
      axesList=self.ui.widgetStem.figure.axes   #子图列表
      leg=axesList[0].get_legend()
      leg.set_visible(checked)
      self.ui.widgetStem.redraw()   


##=====5.极坐标图========
      
   def __drawPolarSpiral(self):  ##绘制螺旋线
      rho = np.arange(0, 2.5, 0.02) #极径，0--2.5,间隔0.02
      theta = 2 * np.pi * rho       #角度，单位：弧度
      self.ui.widgetPolar.figure.clear()
      
   ##ax1是matplotlib.projections.polar.PolarAxes类型的子图
      ax1=self.ui.widgetPolar.figure.add_subplot(1,1,1,polar=True)
      ax1.plot(theta, rho,"r",linewidth=3)
      ax1.set_rmax(3)               #极径最大值
      ax1.set_rticks([0, 1,  2])    #极径刻度坐标
      ax1.set_rlabel_position(90)   # 极径刻度坐标，90°是正北
      ax1.grid(self.ui.chkBoxPolar_gridOn.isChecked())   #是否显示网格
     
   @pyqtSlot()   ## 重画曲线
   def on_btnPolar_redraw_clicked(self):
      self.__drawPolarSpiral()
      self.ui.chkBoxPolar_direction.setCheckState(Qt.Checked)  #恢复为逆时针方向
      self.ui.spinPolar_offset.setValue(0)   #偏移量重置为0
      self.ui.widgetPolar.redraw()
      

   @pyqtSlot(bool)   ## 逆时针方向
   def on_chkBoxPolar_direction_clicked(self,checked):
      ax1=self.ui.widgetPolar.figure.axes[0]    #获取子图
      if self.ui.chkBoxPolar_direction.isChecked():
         ax1.set_theta_direction(1)    # 顺时针方向, -1;  逆时针方向 1:
      else:
         ax1.set_theta_direction(-1)   
      self.ui.widgetPolar.redraw()

   @pyqtSlot(bool)   ## 显示网格
   def on_chkBoxPolar_gridOn_clicked(self,checked):
      ax1=self.ui.widgetPolar.figure.axes[0] #获取子图
      ax1.grid(self.ui.chkBoxPolar_gridOn.isChecked())
      self.ui.widgetPolar.redraw()

   @pyqtSlot(int)   ## 角度偏移量
   def on_spinPolar_offset_valueChanged(self,arg1):
      ax1=self.ui.widgetPolar.figure.axes[0]       #获取子图
      offsetDeg=self.ui.spinPolar_offset.value() 
      ax1.set_theta_offset(np.pi*offsetDeg/180.0)  #单位：弧度
      self.ui.widgetPolar.redraw()
      
   @pyqtSlot()   ## 紧凑布局
   def on_btnPolar_tightLayout_clicked(self):
      self.ui.widgetPolar.figure.tight_layout()
      self.ui.widgetPolar.redraw()

   @pyqtSlot()   ## 旋转
   def on_btnPolar_rotate_clicked(self):
      deg=self.ui.spinPolar_rotation.value()    #旋转角度
      radian=np.pi*deg/180.0     #单位：弧度
      
      ax1=self.ui.widgetPolar.figure.axes[0]    #获取子图
      line=ax1.get_lines()[0]       #获取曲线
      xdata=radian+line.get_xdata() #角度数据，单位：弧度
      line.set_xdata(xdata)
      
      self.ui.widgetPolar.redraw()


##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
