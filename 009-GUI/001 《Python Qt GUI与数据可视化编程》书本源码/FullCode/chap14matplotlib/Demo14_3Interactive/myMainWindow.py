# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QLabel

from PyQt5.QtCore import  pyqtSlot,Qt

from PyQt5.QtChart import QChart,QLineSeries,QValueAxis

import numpy as np

import matplotlib as mpl

from  matplotlib.backends.backend_qt5agg import (FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar)

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo14_3, 交互操作")
      self.__labMove=QLabel("Mouse Move:")
      self.__labMove.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.__labMove)
      
      self.__labPick=QLabel("Mouse Pick:")
      self.__labPick.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.__labPick)
      
      mpl.rcParams['font.sans-serif']=['SimHei']   #显示汉字为 楷体， 汉字不支持 粗体，斜体等设置
      mpl.rcParams['font.size']=11  
##  Windows自带的一些字体
##  黑体：SimHei 宋体：SimSun 新宋体：NSimSun 仿宋：FangSong  楷体：KaiTi 
      mpl.rcParams['axes.unicode_minus'] =False    #减号unicode编码

      self.__fig=None         #Figue对象
      self.__createFigure()   #创建Figure和FigureCanvas对象，初始化界面
      self.__drawFig1X2()  

##  ==============自定义功能函数========================
   def __createFigure(self):  ##创建绘图系统
      self.__fig=mpl.figure.Figure(figsize=(8, 5)) #单位英寸
      figCanvas = FigureCanvas(self.__fig)         #创建FigureCanvas对象，必须传递一个Figure对象
      self.__naviBar=NavigationToolbar(figCanvas, self)  #创建NavigationToolbar工具栏

      actList=self.__naviBar.actions()  #关联的Action列表
      for act in actList:     #获得每个Action的标题和tooltip，可注释掉
         print ("text=%s,\ttoolTip=%s"%(act.text(),act.toolTip()))
      self.__changeActionLanguage() #改工具栏的语言为汉语
   ##工具栏改造
      actList[6].setVisible(False)  #隐藏Subplots 按钮
      actList[7].setVisible(False)  #隐藏Customize按钮
      act8=actList[8] #分隔条
      self.__naviBar.insertAction(act8,self.ui.actTightLayout)    #"紧凑布局"按钮
      self.__naviBar.insertAction(act8,self.ui.actSetCursor)      #"十字光标"按钮
      
      count=len(actList)       #Action的个数
      lastAction=actList[count-1]   #最后一个Action
      self.__naviBar.insertAction(lastAction,self.ui.actScatterAgain)  #"重绘散点"按钮

      lastAction.setVisible(False) #隐藏其原有的坐标提示
      self.__naviBar.addSeparator()
      self.__naviBar.addAction(self.ui.actQuit)    #"退出"按钮
      self.__naviBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon) #显示方式
      
      self.addToolBar(self.__naviBar)  #添加作为主窗口工具栏
      self.setCentralWidget(figCanvas)

      figCanvas.setCursor(Qt.CrossCursor)
   ## 必须保留变量cid，否则可能被垃圾回收
      self._cid1=figCanvas.mpl_connect("motion_notify_event",self.do_canvas_mouseMove)
      self._cid2=figCanvas.mpl_connect("axes_enter_event",self.do_axes_mouseEnter)
      self._cid3=figCanvas.mpl_connect("axes_leave_event",self.do_axes_mouseLeave)
      self._cid4=figCanvas.mpl_connect("pick_event",self.do_series_pick)
      self._cid5=figCanvas.mpl_connect("scroll_event",self.do_scrollZoom)

   def __changeActionLanguage(self):
      actList=self.__naviBar.actions()  #关联的Action列表
      actList[0].setText("复位")         #Home
      actList[0].setToolTip("复位到原始视图") #Reset original view
      
      actList[1].setText("回退")         #Back
      actList[1].setToolTip("回到前一视图") #Back to previous view
      
      actList[2].setText("前进")         #Forward
      actList[2].setToolTip("前进到下一视图")#Forward to next view

      actList[4].setText("平动")         #Pan
      actList[4].setToolTip("左键平移坐标轴，右键缩放坐标轴")   #Pan axes with left mouse, zoom with right
      
      actList[5].setText("缩放")         #Zoom
      actList[5].setToolTip("框选矩形框缩放")   #Zoom to rectangle

      actList[6].setText("子图")         #Subplots
      actList[6].setToolTip("设置子图")  #Configure subplots
      
      actList[7].setText("定制")         #Customize
      actList[7].setToolTip("定制图表参数") #Edit axis, curve and image parameters

      actList[9].setText("保存")         #Save
      actList[9].setToolTip("保存图表")  #Save the figure

   def __drawScatters(self,N=15):
      x=range(N)  #序列0,1，....N-1
   ##      x=np.random.rand(N)
      y=np.random.rand(N) 
      colors=np.random.rand(N)  #0~1之间随机数
      self.__markerSize=(40*(0.2+np.random.rand(N)))**2   #0 to 15 point radius
      self.__axScatter.scatter(x,y,s=self.__markerSize,c=colors,
                             marker='*',alpha=0.5,label="scatter series",picker=True)  #允许被拾取pick
      #s=The marker size in points**2
      #c=color, sequence, or sequence of color, optional, default: 'b'
      ## marker : `~matplotlib.markers.MarkerStyle`, optional, default: 'o'
      self.__axScatter.set_title("散点图")
      self.__axScatter.set_xlabel('序号')   # X轴标题
      
   def __drawFig1X2(self):  #初始化绘图
      gs=self.__fig.add_gridspec(1,2)        #1行，2列
   ##      ax1=self.__fig.add_subplot(1,1,1) #添加一个Axes对象，并返回此对象,不支持constrained_layout
      ax1=self.__fig.add_subplot(gs[0,0],label="Line2D plot") 
      
      t = np.linspace(0, 10, 40)
      y1=np.sin(t)
      y2=np.cos(2*t)
      ax1.plot(t,y1,'r-o',label="sin series", linewidth=1, markersize=5,picker=True) #绘制一条曲线
      ax1.plot(t,y2,'b:',label="cos series",linewidth=2)         #绘制一条曲线
      ax1.set_xlabel('X 轴')  
      ax1.set_ylabel('Y 轴') 
      ax1.set_xlim([0,10])   
      ax1.set_ylim([-1.5,1.5])  
      ax1.set_title("曲线")
      ax1.legend()    #自动创建Axes的图例

      self.__axScatter=self.__fig.add_subplot(gs[0,1],label="scatter plot") #创建子图
      self.__drawScatters(N=15) #绘制散点图

##  ==============event处理函数==========================


##  ==========由connectSlotsByName()自动连接的槽函数============        
   @pyqtSlot()   ## 紧凑布局
   def on_actTightLayout_triggered(self):
      self.__fig.tight_layout()     # 对所有子图 进行一次tight_layout
      self.__fig.canvas.draw()


   @pyqtSlot()   ## 设置鼠标光标
   def on_actSetCursor_triggered(self):
      self.__fig.canvas.setCursor(Qt.CrossCursor)

      
   @pyqtSlot()   ## 重新绘制散点图
   def on_actScatterAgain_triggered(self):
      self.__axScatter.clear()   #清除子图
      self.__drawScatters(N=15)
      self.__fig.canvas.draw()   #刷新
      

##  =================自定义槽函数==========
#event类型 matplotlib.backend_bases.MouseEvent
   def do_canvas_mouseMove(self,event):
      if event.inaxes==None:
         return
      info="%s: xdata=%.2f,ydata=%.2f  "%(
           event.inaxes.get_label(),event.xdata,event.ydata)
      self.__labMove.setText(info)

## event类型：matplotlib.backend_bases.LocationEvent      
   def do_axes_mouseEnter(self,event):
      event.inaxes.patch.set_facecolor('g') #设置背景颜色
      event.inaxes.patch.set_alpha(0.2)     #透明度
      event.canvas.draw()
      
   def do_axes_mouseLeave(self,event):
      event.inaxes.patch.set_facecolor('w')  #设置背景颜色
      event.canvas.draw()

##event 类型： matplotlib.backend_bases.PickEvent
   def do_series_pick(self,event):
      series=event.artist   # 产生事件的对象
      index=event.ind[0]  #索引号,是array([int32])类型,可能有多个对象被pick，只取第1个
      #是否有ind属性与具体的对象有关

      if isinstance(series,mpl.collections.PathCollection):  #scatter()生成的序列
         markerSize=self.__markerSize[index]
         info="%s: index=%d, marker size=%d "%(
            event.mouseevent.inaxes.get_label(),index,markerSize)
      elif isinstance(series,mpl.lines.Line2D):  #plot()生成的序列
   ##         xdata=series.get_xdata() #两种方法都可以
   ##         x=xdata[index]
   ##         ydata=series.get_ydata()
   ##         y=ydata[index]
         x=event.mouseevent.xdata   #标量数据点
         y=event.mouseevent.ydata   #标量数据点
         info="%s: index=%d, data_xy=(%.2f, %.2f) "%(
            series.get_label(),index,x,y)

      self.__labPick.setText(info)
     
#event类型 matplotlib.backend_bases.MouseEvent
   def do_scrollZoom(self,event): #通过鼠标滚轮缩放
      ax=event.inaxes   # 产生事件的axes对象
      if ax==None:
         return
      
      self.__naviBar.push_current() #Push the current view limits and position onto the stack，这样才可以还原
      xmin,xmax=ax.get_xbound() #获取范围
      xlen=xmax-xmin
      
      ymin,ymax=ax.get_ybound() #获取范围
      ylen=ymax-ymin

      xchg=event.step*xlen/20  #step [scalar],positive = ’up’, negative ='down'
      xmin=xmin+xchg
      xmax=xmax-xchg
      
      ychg=event.step*ylen/20
      ymin=ymin+ychg
      ymax=ymax-ychg

      ax.set_xbound(xmin,xmax)
      ax.set_ybound(ymin,ymax)

      event.canvas.draw()

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
