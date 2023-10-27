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
      pass


##  ==============自定义功能函数========================
   def __createFigure(self):
      pass


   def __getMag(self,w,zta=0.2,wn=1.0):  ##计算幅频曲线的数据
      w2=w*w
      a1=1-w2/(wn*wn)
      b1=a1*a1

      b2=4*zta*zta/(wn*wn)*w2
      
      b=np.sqrt(b1+b2)
      mag=-20*np.log10(b)   #单位dB
      return mag

   
   def __drawFig2X1(self):  ##初始化绘图
      pass


##  ==============event处理函数==========================


##  ==========由connectSlotsByName()自动连接的槽函数============        
      
##=====ToolBox 第1组：==="Figure操作" 分组里的功能================
##=======1.1 suptitle  图表的标题
   def  __setFig_suptitle(self,refreshDraw=True): #设置suptitle
      pass


   @pyqtSlot(bool)   ##"1.1 suptitle标题"groupBox
   def on_groupBox_suptitle_clicked(self,checked):
      pass

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
      pass


   @pyqtSlot()   ##文字背景颜色
   def on_btnFig_TitleBackColor_clicked(self):
      pass


##=======1.2 背景与边框
   @pyqtSlot(bool)   ##set_frameon, 显示背景和边框
   def on_chkBoxFig_FrameOn_clicked(self,checked):
      pass
      

   @pyqtSlot()   ##set_facecolor 设置背景颜色
   def on_btnFig_FaceColor_clicked(self):
      pass


   @pyqtSlot(str)   ##设置样式
   def on_comboFig_Style_currentIndexChanged(self,arg1):
      pass


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
      pass


##=======2.1 子图标题
   def  __setAxesTitle(self):
      pass


   @pyqtSlot(bool)   ##"子图标题"GroupBox--CheckBox
   def on_groupBox_AxesTitle_clicked(self,checked):
      pass


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
      pass


   @pyqtSlot()   ##文字背景颜色
   def on_btnAxes_TitleBackColor_clicked(self):
      pass


##=======2.2 子图外观
   @pyqtSlot(bool)   ##set_frame_on, 是否显示背景颜色
   def on_chkBoxAxes_FrameOn_clicked(self,checked):
      pass


   @pyqtSlot()   ##set_facecolor 设置背景颜色
   def on_btnAxes_FaceColor_clicked(self):
      pass


   @pyqtSlot(bool)   ##grid()，设置X网格可见性
   def on_chkBoxAxes_GridX_clicked(self,checked):
      pass

         
   @pyqtSlot(bool)   ##grid(), 设置Y网格可见性
   def on_chkBoxAxes_GridY_clicked(self,checked):
      pass

            
   @pyqtSlot(bool)   ##set_axis_on和 set_axis_off 显示/隐藏坐标轴
   def on_chkBoxAxes_AxisOn_clicked(self,checked):
      pass

      
   @pyqtSlot(bool)   ## minorticks_on 和 minorticks_off 显示/隐藏次刻度和网格
   def on_chkBoxAxes_MinorTicksOn_clicked(self,checked):
      pass

      
##======2.3 图例
   @pyqtSlot(bool)   ##图例可见
   def on_groupBox_AexLegend_clicked(self,checked):
      pass


   @pyqtSlot(int)   ##图例位置
   def on_combo_LegendLoc_currentIndexChanged(self,index):
      pass

      
   @pyqtSlot(bool)   ##图例可拖动
   def on_chkBoxLegend_Dragable_clicked(self,checked):
      pass

      
   @pyqtSlot()   ##重新生成图例
   def on_btnLegend_regenerate_clicked(self):
      pass


##=====ToolBox 第3组："子图曲线设置" 分组里的功能================

##======3.1 选择操作的曲线
   @pyqtSlot(int)   ##选择当前操作曲线
   def on_comboAxes_Lines_currentIndexChanged(self,index):
      pass


##======3.2 曲线外观
   @pyqtSlot(bool)   ##曲线可见
   def on_groupBox_LineSeries_clicked(self,checked):
      pass


   @pyqtSlot(str)   ##set_linestyle
   def on_comboSeries_LineStyle_currentIndexChanged(self,arg1):
      pass


   @pyqtSlot(int)   ##线宽
   def on_spinSeries_LineWidth_valueChanged(self,arg1):
      pass


   @pyqtSlot(str)   ##set_drawstyle()
   def on_comboSeries_DrawStyle_currentIndexChanged(self,arg1):
      pass

   @pyqtSlot()   ##设置曲线颜色
   def on_btnSeries_LineColor_clicked(self):
      pass


##======3.3 标记点
   @pyqtSlot(bool)   ##标记点可见
   def on_groupBox_Marker_clicked(self,checked):
      pass


   @pyqtSlot(str)   ##set_marker 标记点形状
   def on_comboMarker_Shape_currentIndexChanged(self,arg1):
      pass

         
   @pyqtSlot(int)   ##set_markersize 标记点大小
   def on_spinMarker_Size_valueChanged(self,arg1):
      pass


   @pyqtSlot()   ##标记点颜色
   def on_btnMarker_Color_clicked(self):
      pass


   @pyqtSlot(int)   ##set_markeredgewidth 边线线宽
   def on_spinMarker_EdgeWidth_valueChanged(self,arg1):
      pass


   @pyqtSlot()   ##set_markeredgecolor边线颜色
   def on_btnMarker_EdgeColor_clicked(self):
      pass


##=====ToolBox 第4组：==="X坐标轴设置" 分组里的功能================
   @pyqtSlot(bool)   ##axisX 坐标轴可见型，包括label,tick,ticklabels
   def on_groupBox_AxisX_clicked(self,checked):
      pass


##======4.1 数据范围======
   @pyqtSlot()   ## set_xbound 设置范围,与set_xlim，它不管是否反向
   def on_btnAxisX_setBound_clicked(self):
      pass


   @pyqtSlot()   ## invert_xaxis 反向toggle
   def on_chkBoxAxisX_Invert_clicked(self):
      pass
   

   @pyqtSlot(str)   ## 设置坐标尺度
   def on_comboAxisX_Scale_currentIndexChanged(self,arg1):
      pass
      
      
##==========4.2 X轴标题
   def  __setAxisX_Label(self,refreshDraw=True):
      pass


   @pyqtSlot(bool)   ##X 轴标题可见性
   def on_groupBox_AxisXLabel_clicked(self,checked):
      pass


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
      pass


   @pyqtSlot()   ##设置标签格式
   def on_btnAxisX_TickLabFormat_clicked(self):
      pass


   @pyqtSlot()   ##文字颜色
   def on_btnAxisX_TickLabColor_clicked(self):
      pass

         
   @pyqtSlot(int)   ##字体大小
   def on_spinAxisX_TickLabelFontsize_valueChanged(self,arg1):
      pass


   @pyqtSlot(bool)   ## bottom axis major ticklabel
   def on_chkBoxAxisX_TickLabBottom_clicked(self,checked):
      pass


   @pyqtSlot(bool)   ## top axis major ticklabel
   def on_chkBoxAxisX_TickLabTop_clicked(self,checked):
      pass


##==========4.4 ===主刻度线和主网格线
   @pyqtSlot(bool)   ##bottom主刻度线
   def on_chkBoxX_majorTickBottom_clicked(self,checked):
      pass


   @pyqtSlot(bool)   ##top主刻度线
   def on_chkBoxX_majorTickTop_clicked(self,checked):
      pass

      
   @pyqtSlot()   ##主刻度线颜色
   def on_btnLineColorX_majorTick_clicked(self):
      pass

      
   @pyqtSlot(bool)   ##显示主网格线
   def on_chkBoxX_majorGrid_clicked(self,checked):
      pass


   @pyqtSlot()   ##主网格线颜色
   def on_btnLineColorX_majorGrid_clicked(self):
      pass

      
   @pyqtSlot(str)   ##主网格线样式
   def on_comboLineStyle_XmajorGrid_currentIndexChanged(self,arg1):
      pass


##==========4.5 次刻度线和次网格线
   @pyqtSlot(bool)   ##bottom次刻度线
   def on_chkBoxX_minorTickBottom_clicked(self,checked):
      pass


   @pyqtSlot(bool)   ##top次刻度线
   def on_chkBoxX_minorTickTop_clicked(self,checked):
      pass


   @pyqtSlot()   ##次刻度线颜色
   def on_btnLineColorX_minorTick_clicked(self):
      pass


   @pyqtSlot(bool)   ##显示次网格线
   def on_chkBoxX_minorGrid_clicked(self,checked):
      pass


   @pyqtSlot()   ##次网格线颜色
   def on_btnLineColorX_minorGrid_clicked(self):
      pass

      
   @pyqtSlot(str)   ##次网格线样式
   def on_comboLineStyle_XminorGrid_currentIndexChanged(self,arg1):
      pass


##=====ToolBox 第5组：==="Y坐标轴设置" 分组里的功能================
   @pyqtSlot(bool)   ## axisY 坐标轴可见型，包括label,tick,ticklabels
   def on_groupBox_AxisY_clicked(self,checked):
      pass


##======5.1 数据范围======
   @pyqtSlot()   ## set_xbound 设置范围,与set_xlim，它不管是否反向
   def on_btnAxisY_setBound_clicked(self):
      pass


   @pyqtSlot()   ## invert_xaxis 反向toggle
   def on_chkBoxAxisY_Invert_clicked(self):
      pass


   @pyqtSlot(str)   ## 设置坐标尺度
   def on_comboAxisY_Scale_currentIndexChanged(self,arg1):
      pass


##======5.2 Y轴标题
   def  __setAxisY_Label(self,refreshDraw=True):
      pass


   @pyqtSlot(bool)   ##Y 轴标题可见性
   def on_groupBox_AxisYLabel_clicked(self,checked):
      pass


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
      pass


   @pyqtSlot()   ##设置标签格式
   def on_btnAxisY_TickLabFormat_clicked(self):
      pass


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
      pass


   @pyqtSlot(bool)   ##right axis major ticklabel
   def on_chkBoxAxisY_TickLabRight_clicked(self,checked):
      pass


##==========5.4 ===主刻度线和主网格线=====
   @pyqtSlot(bool)   ##显示Left主刻度线
   def on_chkBoxY_majorTickLeft_clicked(self,checked):
      pass


   @pyqtSlot(bool)   ##显示Right主刻度线
   def on_chkBoxY_majorTickRight_clicked(self,checked):
      pass


   @pyqtSlot()   ##主刻度线颜色
   def on_btnLineColorY_majorTick_clicked(self):
      pass

      
   @pyqtSlot(bool)   ##显示主网格线
   def on_chkBoxY_majorGrid_clicked(self,checked):
      pass


   @pyqtSlot()   ##主网格线颜色
   def on_btnLineColorY_majorGrid_clicked(self):
      pass

      
   @pyqtSlot(str)   ##主网格线样式
   def on_comboLineStyle_YmajorGrid_currentIndexChanged(self,arg1):
      pass


##==========5.5 ===次刻度线和次网格线=====
   @pyqtSlot(bool)   ##显示Left次刻度线
   def on_chkBoxY_minorTickLeft_clicked(self,checked):
      pass


   @pyqtSlot(bool)   ##显示Right次刻度线
   def on_chkBoxY_minorTickRight_clicked(self,checked):
      pass


   @pyqtSlot()   ##次刻度线颜色
   def on_btnLineColorY_minorTick_clicked(self):
      pass


   @pyqtSlot(bool)   ##显示次网格线
   def on_chkBoxY_minorGrid_clicked(self,checked):
      pass


   @pyqtSlot()   ##次网格线颜色
   def on_btnLineColorY_minorGrid_clicked(self):
      pass

   @pyqtSlot(str)   ##次网格线样式
   def on_comboLineStyle_YminorGrid_currentIndexChanged(self,arg1):
      pass


##  =============自定义槽函数===============================        
   @pyqtSlot(int)   
   def do_currentAxesChaned(self,index):  #当前子图切换
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
