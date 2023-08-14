# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,
               QSplitter, QColorDialog, QLabel, QComboBox)

from PyQt5.QtCore import  pyqtSlot,Qt

from PyQt5.QtGui import QColor

import numpy as np

import matplotlib as mpl

import matplotlib.style as mplStyle  #一个模块

from  matplotlib.backends.backend_qt5agg import (FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar)


from ui_MainWindow import Ui_MainWindow


class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo14_2, 绘图主要对象的操作")
      mplStyle.use("classic")    #使用样式，必须在绘图之前调用,修改字体后才可显示汉字

      mpl.rcParams['font.sans-serif']=['KaiTi','SimHei']   #显示汉字为 楷体， 汉字不支持 粗体，斜体等设置
      mpl.rcParams['font.size']=12  
##  Windows自带的一些字体
##  黑体：SimHei 宋体：SimSun 新宋体：NSimSun 仿宋：FangSong  楷体：KaiTi 
      mpl.rcParams['axes.unicode_minus'] =False    #减号unicode编码

      self.__fig=None      #Figue对象
      self.__curAxes=None  #当前操作的Axes，为了方便单独用变量
      self.__curLine=None  #当前操作的曲线
      
      self.__createFigure()  #创建Figure和FigureCanvas对象，初始化界面
      self.__drawFig2X1()    #绘图

      axesList=self.__fig.axes   #子图列表
      for one in axesList:       #添加到工具栏上的下拉列表框里
         self.__comboAxes.addItem(one.get_label())

      legendLocs=['best','upper right','upper left', 'lower left',
                  'lower right', 'right', 'center left','center right',
                  'lower center', 'upper center', 'center']  #图例位置
      self.ui.combo_LegendLoc.addItems(legendLocs)    #添加选项
      
      styleList=mplStyle.available     #可用样式列表，字符串列表
      self.ui.comboFig_Style.addItems(styleList)

##  ==============自定义功能函数========================
   def __createFigure(self):
   ##      self.__fig=mpl.figure.Figure(figsize=(8, 5),constrained_layout=True, tight_layout=None)  #单位英寸
   ##      self.__fig=mpl.figure.Figure(figsize=(8, 5))  #单位英寸
      self.__fig=mpl.figure.Figure() 
      figCanvas = FigureCanvas(self.__fig)  #创建FigureCanvas对象，必须传递一个Figure对象
      self.__fig.suptitle("suptitle:matplotlib in Qt GUI",fontsize=16, fontweight='bold')  # 总的图标题

      naviToolbar=NavigationToolbar(figCanvas, self)  #创建NavigationToolbar工具栏
      actList=naviToolbar.actions()  #关联的Action列表
      count=len(actList)       #Action的个数
      lastAction=actList[count-1]   #最后一个Action

      labCurAxes=QLabel("当前子图")
      naviToolbar.insertWidget(lastAction,labCurAxes)
      self.__comboAxes=QComboBox(self)    #子图列表，用于选择子图
      self.__comboAxes.setToolTip("选择当前子图")
      self.__comboAxes.currentIndexChanged.connect(self.do_currentAxesChaned)
      naviToolbar.insertWidget(lastAction,self.__comboAxes)

      naviToolbar.insertAction(lastAction,self.ui.actQuit)  #在最后一个Action之前插入一个Action
   ##      naviToolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)  #ToolButtonTextUnderIcon
      self.addToolBar(naviToolbar)  #添加作为主窗口工具栏

      splitter = QSplitter(self)
      splitter.setOrientation(Qt.Horizontal)
      splitter.addWidget(self.ui.toolBox)    #左侧控制面板
      splitter.addWidget(figCanvas)          #右侧FigureCanvas对象
      self.setCentralWidget(splitter)


   def __getMag(self,w,zta=0.2,wn=1.0):  ##计算幅频曲线的数据
      w2=w*w
      a1=1-w2/(wn*wn)
      b1=a1*a1

      b2=4*zta*zta/(wn*wn)*w2
      
      b=np.sqrt(b1+b2)
      mag=-20*np.log10(b)   #单位dB
      return mag
   
   def __drawFig2X1(self):  ##初始化绘图
##      gs=self.__fig.add_gridspec(2,1)  #2行，1列
##
##      ax1=self.__fig.add_subplot(gs[0,0],label="sin-cos plot")  #子图1
      ax1=self.__fig.add_subplot(2,1,1,label="sin-cos plot")  #子图1
      t = np.linspace(0, 10, 40)
      y1=np.sin(t)
      y2=np.cos(2*t)
      ax1.plot(t,y1,'r-o',label="sin", linewidth=2, markersize=5)    #绘制一条曲线
      ax1.plot(t,y2,'b--',label="cos",linewidth=2)    #绘制一条曲线
      ax1.set_xlabel('X 轴')      # X轴标题
      ax1.set_ylabel('Y 轴')      # Y轴标题
      ax1.set_xlim([0,10])        # X轴坐标范围
      ax1.set_ylim([-1.5,1.5])    # Y轴坐标范围
      ax1.set_title("三角函数曲线")
      ax1.legend()         #自动创建图例
      self.__curAxes=ax1   #当前操作的Axes对象
      

##      ax2=self.__fig.add_subplot(gs[1,0],label="magnitude plot") #子图2
      ax2=self.__fig.add_subplot(2,1,2,label="magnitude plot")    #子图2
      w = np.logspace(-1, 1, 100)   # 10^(-1,1)之间，100个点
      mag=self.__getMag(w,zta=0.1,wn=1)   #阻尼比=0.1
      ax2.semilogx(w,mag,'g-',label=r"$\zeta=0.2$", linewidth=2)  #绘制一条曲线, 

      mag=self.__getMag(w,zta=0.4,wn=1)   #阻尼比=0.4
      ax2.semilogx(w,mag,'r:',label=r"$\zeta=0.4$", linewidth=2)  #绘制一条曲线

      mag=self.__getMag(w,zta=0.8,wn=1)   #阻尼比=0.8
      ax2.semilogx(w,mag,'b--',label=r"$\zeta=0.8$", linewidth=2) #绘制一条曲线

      ax2.set_xlabel('角频率(rad/sec)')   # X轴标题
      ax2.set_ylabel('幅度(dB)')          # Y轴标题
      ax2.set_title("二阶系统幅频曲线")
      ax2.legend()    #自动创建Axes的图例


##  ==============event处理函数==========================


##  ==========由connectSlotsByName()自动连接的槽函数============        
      
##=====ToolBox 第1组：==="Figure操作" 分组里的功能================
##=======1.1 suptitle  图表的标题
   def  __setFig_suptitle(self,refreshDraw=True): #设置suptitle
      textStr=self.ui.editFig_Title.text()
      objText=self.__fig.suptitle(textStr)
   ##      objText=self.__fig.suptitle(textStr,fontstyle='italic',fontweight='bold')
      objText.set_fontsize(self.ui.spinFig_Fontsize.value())  #设置字体大小
      
      #并非所有的字体都支持粗体和斜体，例如某些汉字字体就不支持
      if self.ui.chkBoxFig_Bold.isChecked():  
         objText.set_fontweight('bold')
      else:
         objText.set_fontweight('normal')

      if self.ui.chkBoxFig_Italic.isChecked():
         objText.set_fontstyle('italic')
      else:
         objText.set_fontstyle('normal')

      if refreshDraw:
         self.__fig.canvas.draw()##刷新
      return objText


   @pyqtSlot(bool)   ##"1.1 suptitle标题"groupBox
   def on_groupBox_suptitle_clicked(self,checked):
      if checked:
         self.__setFig_suptitle()
      else:
         self.__fig.suptitle("")    #相当于不显示
         self.__fig.canvas.draw()   #刷新

   @pyqtSlot()   ##"设置标题"按钮
   def on_btnFig_Title_clicked(self):
      self.__setFig_suptitle()
      
        
   @pyqtSlot(int)   ##字体大小
   def on_spinFig_Fontsize_valueChanged(self,arg1):
      self.__setFig_suptitle()

   @pyqtSlot(bool)   ##粗体
   def on_chkBoxFig_Bold_clicked(self,checked):
      self.__setFig_suptitle()

   @pyqtSlot(bool)   ##斜体
   def on_chkBoxFig_Italic_clicked(self,checked):
      self.__setFig_suptitle()

   @pyqtSlot()   ##文字颜色
   def on_btnFig_TitleColor_clicked(self):
      color=QColorDialog.getColor()    #QColor
      if color.isValid():
         r,g,b,a=color.getRgbF()       # 将QColor转换为r,g,b,a
         objText=self.__setFig_suptitle(False)
         objText.set_color((r,g,b,a))  #文字颜色
         self.__fig.canvas.draw()      #刷新

   @pyqtSlot()   ##文字背景颜色
   def on_btnFig_TitleBackColor_clicked(self):
      color=QColorDialog.getColor()    #QColor
      if color.isValid():
         r,g,b,a=color.getRgbF()       #getRgbF(self) -> Tuple[float, float, float, float]
         objText=self.__setFig_suptitle(False)
         objText.set_backgroundcolor((r,g,b,a))  #文字颜色
         self.__fig.canvas.draw()##刷新

   ##=======1.2 背景与边框
   @pyqtSlot(bool)   ##set_frameon, 显示背景和边框
   def on_chkBoxFig_FrameOn_clicked(self,checked):
      self.__fig.set_frameon(checked)
      self.ui.btnFig_FaceColor.setEnabled(checked)
      self.__fig.canvas.draw()      
      

   @pyqtSlot()   ##set_facecolor 设置背景颜色
   def on_btnFig_FaceColor_clicked(self):
      color=QColorDialog.getColor()    
      if color.isValid():
         r,g,b,a=color.getRgbF()       # getRgbF(self) -> Tuple[float, float, float, float]
         self.__fig.set_facecolor((r,g,b))  #Set the face color of the Figure rectangle.      
         self.__fig.canvas.draw()      

   @pyqtSlot(str)   ##设置样式
   def on_comboFig_Style_currentIndexChanged(self,arg1):
      mplStyle.use(arg1)   #会改变字体设置，汉字无法显示，需要重新设置字体
      mpl.rcParams['font.sans-serif']=['KaiTi','SimHei']    #显示汉字为 楷体， 汉字不支持 粗体，斜体等设置
   ##      mpl.rcParams['font.sans-serif']=['SimHei']       #显示汉字为 楷体， 汉字不支持 粗体，斜体等设置
   ##  Windows自带的一些字体
   ##  黑体：SimHei 宋体：SimSun 新宋体：NSimSun 仿宋：FangSong  楷体：KaiTi 
      mpl.rcParams['axes.unicode_minus'] =False    #减号unicode编码
      mpl.rcParams['font.size']=12  
      self.__fig.clear()     #需要清除后重新绘制 
      self.__drawFig2X1()    
      self.__fig.canvas.draw()      #刷新
      

##=====1.3 边距，子图间隔
   @pyqtSlot()   ## tight_layout,  试验功能
   def on_btnFigure_tightLayout_clicked(self):
      self.__fig.tight_layout()     #对所有子图 进行一次tight_layout
      self.__fig.canvas.draw()      #刷新

   @pyqtSlot(float)   ##left margin
   def on_spinFig_marginLeft_valueChanged(self,value):
      self.__fig.subplots_adjust(left=value)
      self.__fig.canvas.draw()
      
   @pyqtSlot(float)   ##right margin
   def on_spinFig_marginRight_valueChanged(self,value):
      self.__fig.subplots_adjust(right=value)
      self.__fig.canvas.draw()

   @pyqtSlot(float)   ##bottom margin
   def on_spinFig_marginBottom_valueChanged(self,value):
      self.__fig.subplots_adjust(bottom=value)
      self.__fig.canvas.draw()

   @pyqtSlot(float)   ##top margin
   def on_spinFig_marginTop_valueChanged(self,value):
      self.__fig.subplots_adjust(top=value)
      self.__fig.canvas.draw()

   @pyqtSlot(float)   ## wspace
   def on_spinFig_wspace_valueChanged(self,value):
      self.__fig.subplots_adjust(wspace=value)
      self.__fig.canvas.draw()

   @pyqtSlot(float)   ## hspace
   def on_spinFig_hspace_valueChanged(self,value):
      self.__fig.subplots_adjust(hspace=value)
      self.__fig.canvas.draw()


##=====ToolBox 第2组："Axes子图操作" 分组里的功能================

   @pyqtSlot(bool)   ##子图是否可见
   def on_chkBoxAxes_Visible_clicked(self,checked):
      self.__curAxes.set_visible(checked)
      self.__fig.canvas.draw()

      self.ui.groupBox_AxesTitle.setEnabled(checked)  #子图标题
      self.ui.groupBox_AxesBack.setEnabled(checked)
      self.ui.groupBox_AexLegend.setEnabled(checked)

      self.ui.page_Series.setEnabled(checked)   #能否设置曲线

##=======2.1 子图标题
   def  __setAxesTitle(self):
      textStr=self.ui.editAxes_Title.text()
      objText=self.__curAxes.set_title(textStr)    #设置标题,获取Text对象
      objText.set_fontsize(self.ui.spinAxes_Fontsize.value())  #设置字体大小
      
      ##并非所有的字体都支持粗体和斜体，例如某些汉字字体
      if self.ui.chkBoxAxes_Bold.isChecked():  
         objText.set_fontweight('bold')
      else:
         objText.set_fontweight('normal')

      if self.ui.chkBoxAxes_Italic.isChecked():
         objText.set_fontstyle('italic')
      else:
         objText.set_fontstyle('normal')

      self.__fig.canvas.draw()
      return objText 


   @pyqtSlot(bool)   ##"子图标题"GroupBox--CheckBox
   def on_groupBox_AxesTitle_clicked(self,checked):
      objTitle=self.__setAxesTitle()  #设置标题
      objTitle.set_visible(checked)
      self.__fig.canvas.draw()

   @pyqtSlot()   ##"设置标题"按钮
   def on_btnAxes_Title_clicked(self):
      self.__setAxesTitle()  #设置标题
      
        
   @pyqtSlot(int)   ##字体大小
   def on_spinAxes_Fontsize_valueChanged(self,arg1):
      self.__setAxesTitle()

   @pyqtSlot(bool)   ##粗体
   def on_chkBoxAxes_Bold_clicked(self,checked):
      self.__setAxesTitle()

   @pyqtSlot(bool)   ##斜体
   def on_chkBoxAxes_Italic_clicked(self,checked):
      self.__setAxesTitle()

   @pyqtSlot()   ##文字颜色
   def on_btnAxes_TitleColor_clicked(self):
      color=QColorDialog.getColor()    
      if color.isValid():
         r,g,b,a=color.getRgbF()      
         objText=self.__setAxesTitle()
         objText.set_color((r,g,b,a))  #文字颜色
         self.__fig.canvas.draw()

   @pyqtSlot()   ##文字背景颜色
   def on_btnAxes_TitleBackColor_clicked(self):
      color=QColorDialog.getColor()    
      if color.isValid():
         r,g,b,a=color.getRgbF()       # getRgbF(self) -> Tuple[float, float, float, float]
         objText=self.__setAxesTitle()
         objText.set_backgroundcolor((r,g,b,a))  #文字颜色
         self.__fig.canvas.draw()   
         

##=======2.2 子图外观
   @pyqtSlot(bool)   ##set_frame_on, 是否显示背景颜色
   def on_chkBoxAxes_FrameOn_clicked(self,checked):
      self.__curAxes.set_frame_on(checked)  #隐藏背景时，边框也隐藏了
      self.ui.btnAxes_FaceColor.setEnabled(checked)
      self.__fig.canvas.draw()
      

   @pyqtSlot()   ##set_facecolor 设置背景颜色
   def on_btnAxes_FaceColor_clicked(self):
      color=QColorDialog.getColor() 
      if color.isValid():
         r,g,b,a=color.getRgbF()   
         self.__curAxes.set_facecolor((r,g,b)) 
         self.__fig.canvas.draw()

   @pyqtSlot(bool)   ##grid()，设置X网格可见性
   def on_chkBoxAxes_GridX_clicked(self,checked):
      self.__curAxes.grid(b=checked,which='both',axis='x')
   ##      which : {'major', 'minor', 'both'}
   ##      axis : {'both', 'x', 'y'}
      self.__fig.canvas.draw()
         
   @pyqtSlot(bool)   ##grid(), 设置Y网格可见性
   def on_chkBoxAxes_GridY_clicked(self,checked):
      self.__curAxes.grid(b=checked,which='both',axis='y')
      self.__fig.canvas.draw()
            
   @pyqtSlot(bool)   ##set_axis_on和 set_axis_off 显示/隐藏坐标轴
   def on_chkBoxAxes_AxisOn_clicked(self,checked):
      if checked:
         self.__curAxes.set_axis_on()
      else:
         self.__curAxes.set_axis_off()
      self.__fig.canvas.draw()
      
   @pyqtSlot(bool)   ## minorticks_on 和 minorticks_off 显示/隐藏次刻度和网格
   def on_chkBoxAxes_MinorTicksOn_clicked(self,checked):
      if checked:
         self.__curAxes.minorticks_on()
      else:
         self.__curAxes.minorticks_off()
      self.__fig.canvas.draw()   #刷新


      
##======2.3 图例
   @pyqtSlot(bool)   ##图例可见
   def on_groupBox_AexLegend_clicked(self,checked):
      legend=self.__curAxes.get_legend()  #获得Legend对象
      legend.set_visible(checked) 
      self.__fig.canvas.draw()  

   @pyqtSlot(int)   ##图例位置
   def on_combo_LegendLoc_currentIndexChanged(self,index):
      legend=self.__curAxes.legend(loc=index)   #需要重新生成图例对象
      legend.set_draggable(self.ui.chkBoxLegend_Dragable.isChecked()) 
      self.__fig.canvas.draw()  
      
      
   @pyqtSlot(bool)   ##图例可拖动
   def on_chkBoxLegend_Dragable_clicked(self,checked):
      legend=self.__curAxes.get_legend()  #获得Legend对象
      legend.set_draggable(checked) 
      self.__fig.canvas.draw()      #刷新
      
   @pyqtSlot()   ##重新生成图例
   def on_btnLegend_regenerate_clicked(self):
      index=self.ui.combo_LegendLoc.currentIndex()
      legend=self.__curAxes.legend(loc=index)
      legend.set_draggable(self.ui.chkBoxLegend_Dragable.isChecked()) 
      self.__fig.canvas.draw()   



##=====ToolBox 第3组："子图曲线设置" 分组里的功能================

##======3.1 选择操作的曲线
   @pyqtSlot(int)   ##选择当前操作曲线
   def on_comboAxes_Lines_currentIndexChanged(self,index):
      lines=self.__curAxes.get_lines()    #子图中的Line2D对象列表，也就是曲线
      self.__curLine=lines[index]         #当前操作的曲线

      lineVisible=self.__curLine.get_visible()
      self.ui.groupBox_LineSeries.setChecked(lineVisible)
      
      marker=self.__curLine.get_marker()       #数据点标记符号
      isMarked= (marker=="" or marker=="None") #是否有标记点
      self.ui.groupBox_Marker.setChecked(not isMarked)
      self.ui.groupBox_Marker.setEnabled(lineVisible)
         
      lw=self.__curLine.get_linewidth()   #线宽
      self.ui.spinSeries_LineWidth.setValue(lw)

      ms=self.__curLine.get_markersize()  #标记点大小
      self.ui.spinMarker_Size.setValue(ms)

      mew=self.__curLine.get_markeredgewidth()   #标记点边线宽度
      self.ui.spinMarker_EdgeWidth.setValue(mew)

   ##======3.2 曲线外观
   @pyqtSlot(bool)   ##曲线可见
   def on_groupBox_LineSeries_clicked(self,checked):
      self.__curLine.set_visible(checked)
      self.ui.groupBox_Marker.setEnabled(checked)
      self.__fig.canvas.draw()

   @pyqtSlot(str)   ##set_linestyle
   def on_comboSeries_LineStyle_currentIndexChanged(self,arg1):
      self.__curLine.set_linestyle(arg1)
      self.__fig.canvas.draw()      

   @pyqtSlot(int)   ##线宽
   def on_spinSeries_LineWidth_valueChanged(self,arg1):
      self.__curLine.set_linewidth(arg1)
      self.__fig.canvas.draw()

   @pyqtSlot(str)   ##set_drawstyle()
   def on_comboSeries_DrawStyle_currentIndexChanged(self,arg1):
      self.__curLine.set_drawstyle(arg1)
      self.__fig.canvas.draw()

   @pyqtSlot()   ##设置曲线颜色
   def on_btnSeries_LineColor_clicked(self):
      color=QColorDialog.getColor() 
      if color.isValid():
         r,g,b,a=color.getRgbF()  
         self.__curLine.set_color((r,g,b)) 
         self.__fig.canvas.draw()

   ##======3.3 标记点
   @pyqtSlot(bool)   ##标记点可见
   def on_groupBox_Marker_clicked(self,checked):
      if checked:
         arg1=self.ui.comboMarker_Shape.currentText()
         shape=arg1[0]  #左边第1个字符
      else:
         shape=""       #无标记点
      self.__curLine.set_marker(shape)
      self.__fig.canvas.draw()

   @pyqtSlot(str)   ##set_marker 标记点形状
   def on_comboMarker_Shape_currentIndexChanged(self,arg1):
      shape=arg1[0]  #左边第1个字符
      self.__curLine.set_marker(shape) 
      self.__fig.canvas.draw()
         
   @pyqtSlot(int)   ##set_markersize 标记点大小
   def on_spinMarker_Size_valueChanged(self,arg1):
      self.__curLine.set_markersize(arg1) 
      self.__fig.canvas.draw()

   @pyqtSlot()   ##标记点颜色
   def on_btnMarker_Color_clicked(self):
      color=QColorDialog.getColor() 
      if color.isValid():
         r,g,b,a=color.getRgbF()  
         self.__curLine.set_markerfacecolor((r,g,b)) 
         self.__fig.canvas.draw()

   @pyqtSlot(int)   ##set_markeredgewidth 边线线宽
   def on_spinMarker_EdgeWidth_valueChanged(self,arg1):
      self.__curLine.set_markeredgewidth(arg1) 
      self.__fig.canvas.draw()

   @pyqtSlot()   ##set_markeredgecolor边线颜色
   def on_btnMarker_EdgeColor_clicked(self):
      color=QColorDialog.getColor() 
      if color.isValid():
         r,g,b,a=color.getRgbF()  
         self.__curLine.set_markeredgecolor((r,g,b)) 
         self.__fig.canvas.draw()


##=====ToolBox 第4组：==="X坐标轴设置" 分组里的功能================
   @pyqtSlot(bool)   ##axisX 坐标轴可见型，包括label,tick,ticklabels
   def on_groupBox_AxisX_clicked(self,checked):
      self.__curAxes.xaxis.set_visible(checked)
   ##      self.__curAxes.get_xaxis().set_visible(checked) #等效
      self.__fig.canvas.draw()


##======4.1 数据范围======

   @pyqtSlot()   ## set_xbound 设置范围,与set_xlim，它不管是否反向
   def on_btnAxisX_setBound_clicked(self):
      self.__curAxes.set_xbound(self.ui.spinAxisX_Min.value(),
                                self.ui.spinAxisX_Max.value())
      self.__fig.canvas.draw()

   @pyqtSlot()   ## invert_xaxis 反向toggle
   def on_chkBoxAxisX_Invert_clicked(self):
      self.__curAxes.invert_xaxis()
      self.__fig.canvas.draw()

   @pyqtSlot(str)   ## 设置坐标尺度
   def on_comboAxisX_Scale_currentIndexChanged(self,arg1):
      self.__curAxes.set_xscale(arg1)
      self.__fig.canvas.draw()
      
      
##==========4.2 X轴标题
   def  __setAxisX_Label(self,refreshDraw=True):
      textStr=self.ui.editAxisX_Label.text()
      objText=self.__curAxes.set_xlabel(textStr)
   ##      objText=self.__curAxes.set_xlabel(textStr,fontstyle='italic',fontweight='bold')
      objText.set_fontsize(self.ui.spinAxisX_LabelFontsize.value())  #设置字体大小
      
      ## 并非所有的字体都支持粗体和斜体，例如某些汉字字体
      if self.ui.chkBoxAxisX_LabelBold.isChecked():  
         objText.set_fontweight('bold')
      else:
         objText.set_fontweight('normal')

      if self.ui.chkBoxAxisX_LabelItalic.isChecked():
         objText.set_fontstyle('italic')
      else:
         objText.set_fontstyle('normal')

      if refreshDraw:
         self.__fig.canvas.draw()
      return objText


   @pyqtSlot(bool)   ##X 轴标题可见性
   def on_groupBox_AxisXLabel_clicked(self,checked):
      objText=self.__setAxisX_Label(False)  #不立刻draw
      objText.set_visible(checked)
      self.__fig.canvas.draw()

   @pyqtSlot()   ##设置X轴Label
   def on_btnAxisX_setLabel_clicked(self):
      self.__setAxisX_Label()
      
   @pyqtSlot(int)   ##字体大小
   def on_spinAxisX_LabelFontsize_valueChanged(self,arg1):
      self.__setAxisX_Label()

   @pyqtSlot(bool)   ##粗体
   def on_chkBoxAxisX_LabelBold_clicked(self,checked):
      self.__setAxisX_Label()

   @pyqtSlot(bool)   ##斜体
   def on_chkBoxAxisX_LabelItalic_clicked(self,checked):
      self.__setAxisX_Label()

   @pyqtSlot()   ##文字颜色
   def on_btnAxisX_LabelColor_clicked(self):
      color=QColorDialog.getColor() #QColor
      if color.isValid():
         r,g,b,a=color.getRgbF()    #getRgbF(self) -> Tuple[float, float, float, float]
         objText=self.__setAxisX_Label(False)
         objText.set_color((r,g,b,a))  #文字颜色
         self.__fig.canvas.draw()

      
##======4.3  X轴主刻度标签
   @pyqtSlot(bool)   ##"4.3主刻度标签"GroupBox，刻度标签可见性
   def on_groupBoxAxisX_TickLabel_clicked(self,checked):
   ##      for label in self.__curAxes.get_xticklabels(): #等效
      for label in self.__curAxes.xaxis.get_ticklabels(): # 包括top和bottom的
         label.set_visible(checked)
      self.__fig.canvas.draw()

   @pyqtSlot()   ##设置标签格式
   def on_btnAxisX_TickLabFormat_clicked(self):
      formatStr=self.ui.editAxisX_TickLabFormat.text()      # 格式字符串，如 "%.2f"
      formatter = mpl.ticker.FormatStrFormatter(formatStr)  #用字符串指定格式
      self.__curAxes.xaxis.set_major_formatter(formatter)   #设置主刻度字符串格式
      self.__fig.canvas.draw()

   @pyqtSlot()   ##文字颜色
   def on_btnAxisX_TickLabColor_clicked(self):
      color=QColorDialog.getColor() 
      if not color.isValid():
         return
      r,g,b,a=color.getRgbF()      
      for label in self.__curAxes.xaxis.get_ticklabels(): #包括bottom和top的
   ##         for label in self.__curAxes.get_xticklabels(): 等效
         label.set_color((r,g,b,a))
      self.__fig.canvas.draw()
         
   @pyqtSlot(int)   ##字体大小
   def on_spinAxisX_TickLabelFontsize_valueChanged(self,arg1):
      for label in self.__curAxes.xaxis.get_ticklabels():   # label is a Text instance
         label.set_fontsize(arg1)
      self.__fig.canvas.draw()

   @pyqtSlot(bool)   ## bottom axis major ticklabel
   def on_chkBoxAxisX_TickLabBottom_clicked(self,checked):
      for tick in self.__curAxes.xaxis.get_major_ticks():
         tick.label1On = checked  #bottom轴
   ##        tick.label2On = True            #右侧坐标轴
   ##        tick.label1.set_color('red')    #左侧坐标轴标签
   ##        tick.label2.set_color('green')  #右侧坐标轴标签
   ##        tick.gridOn=True                #显示网格线
      self.__fig.canvas.draw()

   @pyqtSlot(bool)   ## top axis major ticklabel
   def on_chkBoxAxisX_TickLabTop_clicked(self,checked):
      for tick in self.__curAxes.xaxis.get_major_ticks():
         tick.label2On = checked  #top轴
      self.__fig.canvas.draw()


#==========4.4 ===主刻度线和主网格线
   @pyqtSlot(bool)   ##bottom主刻度线
   def on_chkBoxX_majorTickBottom_clicked(self,checked):
      for tick in self.__curAxes.xaxis.get_major_ticks():   #Tick对象
         tick.tick1On=checked   #是属性,而不是方法
         tick.tick1line.set_markersize(6)       #长度
         tick.tick1line.set_markeredgewidth(3)  #宽度
      self.__fig.canvas.draw()      

   @pyqtSlot(bool)   ##top主刻度线
   def on_chkBoxX_majorTickTop_clicked(self,checked):
      for tick in self.__curAxes.xaxis.get_major_ticks():   #Tick对象
         tick.tick2On=checked       #是属性,而不是方法
         tick.tick2line.set_markersize(6)       #长度
         tick.tick2line.set_markeredgewidth(3)  #宽度
      self.__fig.canvas.draw()
##      for line in self.__curAxes.xaxis.get_majorticklines(): # line2D对象列表,get_majorticklines
##         line.set_visible(checked)
##         line.set_markersize(6)         #长度
##         line.set_markeredgewidth(3)    #宽度
##      self.__fig.canvas.draw()    
      
   @pyqtSlot()   ##主刻度线颜色
   def on_btnLineColorX_majorTick_clicked(self):
      color=QColorDialog.getColor()    #QColor
      if not color.isValid():
         return
      r,g,b,a=color.getRgbF() 
      for line in self.__curAxes.xaxis.get_majorticklines(): # line2D对象列表，包括top和bottom
         line.set_color((r,g,b,a))
      self.__fig.canvas.draw()
      
      
   @pyqtSlot(bool)   ##显示主网格线
   def on_chkBoxX_majorGrid_clicked(self,checked):
      for tick in self.__curAxes.xaxis.get_major_ticks():   #Tick对象
         tick.gridOn=checked     #是属性,而不是方法
         tick.gridline.set_linewidth(2)
         ls=self.ui.comboLineStyle_XmajorGrid.currentText()
         tick.gridline.set_linestyle(ls)     #通过Tick的gridline属性操作次网格线
      self.__fig.canvas.draw()   

   @pyqtSlot()   ##主网格线颜色
   def on_btnLineColorX_majorGrid_clicked(self):
      color=QColorDialog.getColor() #QColor
      if not color.isValid():
         return
      r,g,b,a=color.getRgbF()  

      for tick in self.__curAxes.xaxis.get_major_ticks():   #Tick对象
         tick.gridline.set_color((r,g,b,a))  
   ##      for line in self.__curAxes.xaxis.get_gridlines(): # 只能得到主网格线 line2D对象列表
   ##         line.set_color((r,g,b,a))
      self.__fig.canvas.draw()
      
   @pyqtSlot(str)   ##主网格线样式
   def on_comboLineStyle_XmajorGrid_currentIndexChanged(self,arg1):
      for line in self.__curAxes.xaxis.get_gridlines():  #line2D对象列表
         line.set_linestyle(arg1)
      self.__fig.canvas.draw()


   ##==========4.5 次刻度线和次网格线
   @pyqtSlot(bool)   ##bottom次刻度线
   def on_chkBoxX_minorTickBottom_clicked(self,checked):
      minorLocator = mpl.ticker.AutoMinorLocator()    #必须创建次刻度
      self.__curAxes.xaxis.set_minor_locator(minorLocator)
      for tick in self.__curAxes.xaxis.get_minor_ticks():   
         tick.tick1On=checked    #是属性,而不是方法
         tick.tick1line.set_markersize(3)       #长度
         tick.tick1line.set_markeredgewidth(2)  #宽度
   ##      for line in self.__curAxes.xaxis.get_minorticklines(): # 上下的全部
   ##         line.set_visible(checked)
   ##         line.set_markersize(3)       #长度
   ##         line.set_markeredgewidth(2)  #宽度
      self.__fig.canvas.draw()

   @pyqtSlot(bool)   ##top次刻度线
   def on_chkBoxX_minorTickTop_clicked(self,checked):
      minorLocator = mpl.ticker.AutoMinorLocator()    #必须创建次刻度
      self.__curAxes.xaxis.set_minor_locator(minorLocator)
      for tick in self.__curAxes.xaxis.get_minor_ticks():   
         tick.tick2On=checked   #是属性,而不是方法
         tick.tick2line.set_markersize(3)       #长度
         tick.tick2line.set_markeredgewidth(2)  #宽度
      self.__fig.canvas.draw()


   @pyqtSlot()   ##次刻度线颜色
   def on_btnLineColorX_minorTick_clicked(self):
      color=QColorDialog.getColor() #QColor
      if not color.isValid():
         return
      r,g,b,a=color.getRgbF()       # getRgbF(self) -> Tuple[float, float, float, float]
      for line in self.__curAxes.xaxis.get_minorticklines():   # line2D对象列表
         line.set_color((r,g,b,a))
   ##下面的方法也可以
   ##      for tick in self.__curAxes.xaxis.get_minor_ticks(): #Tick对象
   ##         tick.tick1line.set_color((r,g,b,a))  #bottom
   ##         tick.tick2line.set_color((r,g,b,a))  #top
      self.__fig.canvas.draw()


   @pyqtSlot(bool)   ##显示次网格线
   def on_chkBoxX_minorGrid_clicked(self,checked):
      for tick in self.__curAxes.xaxis.get_minor_ticks():   #Tick对象
         tick.gridOn=checked   #通过Tick.gridOn属性操作是否显示次网格线
         tick.gridline.set_linewidth(1)
         ls=self.ui.comboLineStyle_XminorGrid.currentText()
         tick.gridline.set_linestyle(ls)  
      self.__fig.canvas.draw()

   @pyqtSlot()   ##次网格线颜色
   def on_btnLineColorX_minorGrid_clicked(self):
      color=QColorDialog.getColor()    
      if not color.isValid():
         return
      r,g,b,a=color.getRgbF()     
      for tick in self.__curAxes.xaxis.get_minor_ticks(): #Tick对象
         tick.gridline.set_color((r,g,b,a))  
      self.__fig.canvas.draw()
      
   @pyqtSlot(str)   ##次网格线样式
   def on_comboLineStyle_XminorGrid_currentIndexChanged(self,arg1):
      for tick in self.__curAxes.xaxis.get_minor_ticks(): #Tick对象
         tick.gridline.set_linestyle(arg1) 
      self.__fig.canvas.draw()


##=====ToolBox 第5组：==="Y坐标轴设置" 分组里的功能================
   @pyqtSlot(bool)   ## axisY 坐标轴可见型，包括label,tick,ticklabels
   def on_groupBox_AxisY_clicked(self,checked):
##      self.__curAxes.get_yaxis().set_visible(checked)  #等效
      self.__curAxes.yaxis.set_visible(checked)
      self.__fig.canvas.draw()

##======5.1 数据范围======

   @pyqtSlot()   ## set_xbound 设置范围,与set_xlim，它不管是否反向
   def on_btnAxisY_setBound_clicked(self):
      self.__curAxes.set_ybound(self.ui.spinAxisY_Min.value(),
                                self.ui.spinAxisY_Max.value())
      self.__fig.canvas.draw()

   @pyqtSlot()   ## invert_xaxis 反向toggle
   def on_chkBoxAxisY_Invert_clicked(self):
      self.__curAxes.invert_yaxis()
      self.__fig.canvas.draw()   

   @pyqtSlot(str)   ## 设置坐标尺度
   def on_comboAxisY_Scale_currentIndexChanged(self,arg1):
      self.__curAxes.set_yscale(arg1)
      self.__fig.canvas.draw()   


##======5.2 Y轴标题
   def  __setAxisY_Label(self,refreshDraw=True):
      textStr=self.ui.editAxisY_Label.text()
      objText=self.__curAxes.set_ylabel(textStr)
##      objText=self.__curAxes.set_ylabel(textStr,fontstyle='italic',fontweight='bold')
      objText.set_fontsize(self.ui.spinAxisY_LabelFontsize.value())  #设置字体大小
      
      ##并非所有的字体都支持粗体和斜体，例如某些汉字字体
      if self.ui.chkBoxAxisY_LabelBold.isChecked():  
         objText.set_fontweight('bold')
      else:
         objText.set_fontweight('normal')

      if self.ui.chkBoxAxisY_LabelItalic.isChecked():
         objText.set_fontstyle('italic')
      else:
         objText.set_fontstyle('normal')

      if refreshDraw:
         self.__fig.canvas.draw()   #刷新
      return objText


   @pyqtSlot(bool)   ##Y 轴标题可见性
   def on_groupBox_AxisYLabel_clicked(self,checked):
      objText=self.__setAxisY_Label(False)  #不立刻draw
      objText.set_visible(checked)
      self.__fig.canvas.draw()   

   @pyqtSlot()   ##设置Y轴Label
   def on_btnAxisY_setLabel_clicked(self):
      self.__setAxisY_Label()
      
   @pyqtSlot(int)   ##字体大小
   def on_spinAxisY_LabelFontsize_valueChanged(self,arg1):
      self.__setAxisY_Label()

   @pyqtSlot(bool)   ##粗体
   def on_chkBoxAxisY_LabelBold_clicked(self,checked):
      self.__setAxisY_Label()

   @pyqtSlot(bool)   ##斜体
   def on_chkBoxAxisY_LabelItalic_clicked(self,checked):
      self.__setAxisY_Label()

   @pyqtSlot()   ##文字颜色
   def on_btnAxisY_LabelColor_clicked(self):
      color=QColorDialog.getColor() #QColor
      if color.isValid():
         r,g,b,a=color.getRgbF()    #getRgbF(self) -> Tuple[float, float, float, float]
         objText=self.__setAxisY_Label(False)
         objText.set_color((r,g,b,a))  #文字颜色
         self.__fig.canvas.draw()      #刷新

##======5.3  Y轴主刻度标签
   @pyqtSlot(bool)   ##刻度标签可见性
   def on_groupBoxAxisY_TickLabel_clicked(self,checked):
      for label in self.__curAxes.yaxis.get_ticklabels(): 
         label.set_visible(checked)
      self.__fig.canvas.draw()   

   @pyqtSlot()   ##设置标签格式
   def on_btnAxisY_TickLabFormat_clicked(self):
      formatStr=self.ui.editAxisY_TickLabFormat.text()      # 格式字符串，如 "%.2f"
##import matplotlib.ticker as ticker
      formatter = mpl.ticker.FormatStrFormatter(formatStr)  #用字符串指定格式
      self.__curAxes.yaxis.set_major_formatter(formatter)   #设置主刻度字符串格式
      self.__fig.canvas.draw()


   @pyqtSlot()   ##文字颜色
   def on_btnAxisY_TickLabColor_clicked(self):
      color=QColorDialog.getColor() 
      if color.isValid():
         r,g,b,a=color.getRgbF()    
         for label in self.__curAxes.yaxis.get_ticklabels():   
            label.set_color((r,g,b,a))
         self.__fig.canvas.draw()
         
   @pyqtSlot(int)   #字体大小
   def on_spinAxisY_TickLabelFontsize_valueChanged(self,arg1):
      for label in self.__curAxes.yaxis.get_ticklabels():  
         label.set_fontsize(arg1)
      self.__fig.canvas.draw()

         
   @pyqtSlot(bool)   ##Left axis major ticklabel
   def on_chkBoxAxisY_TickLabLeft_clicked(self,checked):
      for tick in self.__curAxes.yaxis.get_major_ticks():
        tick.label1On = checked  #left轴的ticklabel
      self.__fig.canvas.draw()


   @pyqtSlot(bool)   ##right axis major ticklabel
   def on_chkBoxAxisY_TickLabRight_clicked(self,checked):
      for tick in self.__curAxes.yaxis.get_major_ticks():
        tick.label2On = checked  #right轴的ticklabel
      self.__fig.canvas.draw()

#==========5.4 ===主刻度线和主网格线=====
   @pyqtSlot(bool)   ##显示Left主刻度线
   def on_chkBoxY_majorTickLeft_clicked(self,checked):
      for tick in self.__curAxes.yaxis.get_major_ticks():   #Tick对象
         tick.tick1On=checked   #是属性,而不是方法
         tick.tick1line.set_markersize(6)       #长度
         tick.tick1line.set_markeredgewidth(3)  #宽度
##      for line in self.__curAxes.yaxis.get_majorticklines(): # line2D对象列表,得到上下的刻度，最好不用这种方法
##         line.set_visible(checked)
##         line.set_markersize(6) #长度
##         line.set_markeredgewidth(3)  #宽度
      self.__fig.canvas.draw()
      

   @pyqtSlot(bool)   ##显示Right主刻度线
   def on_chkBoxY_majorTickRight_clicked(self,checked):
      for tick in self.__curAxes.yaxis.get_major_ticks(): #Tick对象
         tick.tick2On=checked   #是属性,而不是方法
         tick.tick2line.set_markersize(6)       #长度
         tick.tick2line.set_markeredgewidth(3)  #宽度
      self.__fig.canvas.draw()

   @pyqtSlot()   ##主刻度线颜色
   def on_btnLineColorY_majorTick_clicked(self):
      color=QColorDialog.getColor() #QColor
      if not color.isValid():
         return
      r,g,b,a=color.getRgbF()       #getRgbF(self) -> Tuple[float, float, float, float]
      for line in self.__curAxes.yaxis.get_majorticklines():   # line2D对象列表
         line.set_color((r,g,b,a))
      self.__fig.canvas.draw()      
      
      
   @pyqtSlot(bool)   ##显示主网格线
   def on_chkBoxY_majorGrid_clicked(self,checked):
##      self.__curAxes.yaxis.grid(checked,which='major',linewidth=2) #如果设置了属性，总是显示
      for tick in self.__curAxes.yaxis.get_major_ticks():   #Tick对象
         tick.gridOn=checked     #是属性,而不是方法
         tick.gridline.set_linewidth(2)
         ls=self.ui.comboLineStyle_YmajorGrid.currentText()
         tick.gridline.set_linestyle(ls)   #通过Tick的gridline属性操作次网格线
      self.__fig.canvas.draw()      

   @pyqtSlot()   ##主网格线颜色
   def on_btnLineColorY_majorGrid_clicked(self):
      color=QColorDialog.getColor() #QColor
      if not color.isValid():
         return
      r,g,b,a=color.getRgbF()
      for tick in self.__curAxes.yaxis.get_major_ticks():   #Tick对象，也是可以的
         tick.gridline.set_color((r,g,b,a))        #是属性,而不是方法
##      for line in self.__curAxes.yaxis.get_gridlines():   # 只能得到主网格线 line2D对象列表
##         line.set_color((r,g,b,a))
      self.__fig.canvas.draw()
      
   @pyqtSlot(str)   ##主网格线样式
   def on_comboLineStyle_YmajorGrid_currentIndexChanged(self,arg1):
      for line in self.__curAxes.yaxis.get_gridlines():     # 只能得到主网格，不能得到次网格
         line.set_linestyle(arg1)
      self.__fig.canvas.draw()      


#==========5.5 ===次刻度线和次网格线=====
   @pyqtSlot(bool)   ##显示Left次刻度线
   def on_chkBoxY_minorTickLeft_clicked(self,checked):
      minorLocator = mpl.ticker.AutoMinorLocator()    #必须创建次刻度
      self.__curAxes.yaxis.set_minor_locator(minorLocator)

      for tick in self.__curAxes.yaxis.get_minor_ticks(): 
         tick.tick1On=checked   #是属性,而不是方法
         tick.tick1line.set_markersize(3)       #长度
         tick.tick1line.set_markeredgewidth(2)  #宽度
##      for line in self.__curAxes.yaxis.get_minorticklines(): # line2D对象列表
##         line.set_visible(checked)
##         line.set_markersize(3) #长度
##         line.set_markeredgewidth(2)  #宽度
      self.__fig.canvas.draw()

   @pyqtSlot(bool)   ##显示Right次刻度线
   def on_chkBoxY_minorTickRight_clicked(self,checked):
      minorLocator = mpl.ticker.AutoMinorLocator()    #必须创建次刻度
      self.__curAxes.yaxis.set_minor_locator(minorLocator)

      for tick in self.__curAxes.yaxis.get_minor_ticks(): 
         tick.tick2On=checked    #是属性,而不是方法
         tick.tick2line.set_markersize(3)       #长度
         tick.tick2line.set_markeredgewidth(2)  #宽度
      self.__fig.canvas.draw()

   @pyqtSlot()   ##次刻度线颜色
   def on_btnLineColorY_minorTick_clicked(self):
      color=QColorDialog.getColor()    
      if not color.isValid():
         return
      r,g,b,a=color.getRgbF()    #getRgbF(self) -> Tuple[float, float, float, float]
      for line in self.__curAxes.yaxis.get_minorticklines(): # line2D对象列表,get_majorticklines
         line.set_color((r,g,b,a))
##      #下面的方法也可以
##      for tick in self.__curAxes.yaxis.get_minor_ticks(): #Tick对象
##         tick.tick1line.set_color((r,g,b,a))   #通过Tick的gridline属性操作次网格线
##         tick.tick2line.set_color((r,g,b,a))   #通过Tick的gridline属性操作次网格线
      self.__fig.canvas.draw()##刷新


   @pyqtSlot(bool)   ##显示次网格线
   def on_chkBoxY_minorGrid_clicked(self,checked):
##      self.__curAxes.yaxis.grid(which='minor',linewidth=1) #设置网格线属性，如果设置了属性，总是显示
      for tick in self.__curAxes.yaxis.get_minor_ticks():   #Tick对象
         tick.gridOn=checked     #通过Tick.gridOn属性操作是否显示次网格线
         tick.gridline.set_linewidth(1)
         ls=self.ui.comboLineStyle_YminorGrid.currentText()
         tick.gridline.set_linestyle(ls)   #通过Tick的gridline属性操作次网格线
      self.__fig.canvas.draw()
      

   @pyqtSlot()   ##次网格线颜色
   def on_btnLineColorY_minorGrid_clicked(self):
      color=QColorDialog.getColor() 
      if not color.isValid():
         return
      r,g,b,a=color.getRgbF()  

      for tick in self.__curAxes.yaxis.get_minor_ticks(): #Tick对象
         tick.gridline.set_color((r,g,b,a))  
      self.__fig.canvas.draw()

      
   @pyqtSlot(str)   ##次网格线样式
   def on_comboLineStyle_YminorGrid_currentIndexChanged(self,arg1):
      for tick in self.__curAxes.yaxis.get_minor_ticks(): #Tick对象
         tick.gridline.set_linestyle(arg1)   
      self.__fig.canvas.draw()   

      

##  =============自定义槽函数===============================        
   @pyqtSlot(int)   
   def do_currentAxesChaned(self,index):  #当前子图切换
      axesList=self.__fig.axes      #子图列表
      self.__curAxes=self.__fig.axes[index]   #当前操作的Axes，为了方便单独用变量

      ## 3.1 刷新子图内的曲线列表
      self.ui.comboAxes_Lines.clear() 
      lines=self.__curAxes.get_lines()   #子图中的Line2D对象列表，也就是曲线
      for oneLine in lines:
         self.ui.comboAxes_Lines.addItem(oneLine.get_label())

      axesLabel=self.__curAxes.get_label()   #子图的Label
      self.ui.chkBoxAxes_Visible.setText("当前子图可见（"+axesLabel+"）")

      ## 第2组，“当前子图可见”
      axesVisible=self.__curAxes.get_visible()  #子图可见
      self.ui.chkBoxAxes_Visible.setChecked(axesVisible)
      self.on_chkBoxAxes_Visible_clicked(axesVisible)    #执行一次

      ## 2.2 子图外观
      isFrameOn=self.__curAxes.get_frame_on()   #是否显示背景
      self.ui.chkBoxAxes_FrameOn.setChecked(isFrameOn)

      ##  2.3 图例
      legend=self.__curAxes.get_legend()     #返回子图的图例
      self.ui.groupBox_AexLegend.setChecked(legend.get_visible())       #图例可见
      self.ui.chkBoxLegend_Dragable.setChecked(legend.get_draggable())  #图例可拖动

      # 第4组  X 轴
      xmin,xmax=self.__curAxes.get_xbound()  #轴数据范围
      self.ui.spinAxisX_Min.setValue(xmin)
      self.ui.spinAxisX_Max.setValue(xmax)

      textStr=self.__curAxes.get_xlabel()    #轴标题
      self.ui.editAxisX_Label.setText(textStr)

      textStr=self.__curAxes.get_xscale()    #scale
      self.ui.comboAxisX_Scale.setCurrentText(textStr)

      # 第5组  Y 轴
      ymin,ymax=self.__curAxes.get_ybound()  #轴数据范围
      self.ui.spinAxisY_Min.setValue(ymin)
      self.ui.spinAxisY_Max.setValue(ymax)

      textStr=self.__curAxes.get_ylabel()    #轴标题
      self.ui.editAxisY_Label.setText(textStr)
      
      textStr=self.__curAxes.get_yscale()    #scale
      self.ui.comboAxisY_Scale.setCurrentText(textStr)

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
