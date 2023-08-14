# -*- coding: utf-8 -*-

import sys,random

from PyQt5.QtWidgets import  (QApplication, QMainWindow, QWidget,
                  QSplitter, QColorDialog,QInputDialog)

from PyQt5.QtCore import  pyqtSlot,Qt


from PyQt5.QtGui import QVector3D

##from PyQt5.QtSql import 

##from PyQt5.QtMultimedia import

##from PyQt5.QtMultimediaWidgets import

from PyQt5.QtDataVisualization import *


from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo13_1, Q3DBars和QBar3DSeries绘制三维柱状图")
      self.ui.sliderZoom.setRange(10,500)  #缺省缩放范围，100=原始大小
      self.ui.sliderH.setRange(-180,180)   #水平旋转角度范围
      self.ui.sliderV.setRange(-180,180)   #垂直旋转角度范围

      self.__iniGraph3D() ##创建图表
      splitter=QSplitter(Qt.Horizontal)
      splitter.addWidget(self.ui.frameSetup) #左侧控制面板
      splitter.addWidget(self.__container)   #右侧图表

      self.setCentralWidget(splitter)     #设置主窗口中心组件
      

##  ==============自定义功能函数========================

   def __iniGraph3D(self):    ##创建3D图表
      self.graph3D = Q3DBars()
      self.__container = QWidget.createWindowContainer(self.graph3D) #Q3DBars继承自QWindow
      camView=Q3DCamera.CameraPresetFrontHigh   
      self.graph3D.scene().activeCamera().setCameraPreset(camView)   #设置视角
      self.graph3D.activeTheme().setLabelBackgroundEnabled(False)    #不显示标签背景

   ##  创建坐标轴
      axisV=QValue3DAxis()          #数值坐标轴
      axisV.setTitle("销量(万元)")
      axisV.setTitleVisible(True)   #轴标题可见
      axisV.setLabelFormat("%.1f")  #轴标签格式 
      self.graph3D.setValueAxis(axisV)   #设置数值坐标轴

      axisRow=QCategory3DAxis()     #文字类别型坐标轴
      axisRow.setTitle("row axis")
   ##      rowLabs=["Week1" , "Week2", "Week3"]
   ##      axisRow.setLabels(rowLabs) #可以在此设置行标签，也可以在dataProxy里设置
      axisRow.setTitleVisible(True)
   ##      axisRow.setLabelAutoRotation(90) #随camera自动旋转的角度，最大90，缺省0
      self.graph3D.setRowAxis(axisRow)    #设置行坐标轴

      axisCol=QCategory3DAxis()   #文字类别型坐标轴
      axisCol.setTitle("column axis")
      colLabs=["Mon", "Tue" , "Wed", "Thur", "Fri", "Sat","Sun"]
   ##      axisCol.setLabels(colLabs)   #可以在此设置列标签,也可以在dataProxy里设置
      axisCol.setTitleVisible(True)
   ##      axisCol.setLabelAutoRotation(90) #随camera自动旋转的角度，最大90，缺省0
      self.graph3D.setColumnAxis(axisCol)  #设置列坐标轴

   ##  创建序列
      self.series = QBar3DSeries()   #三维柱状图序列
      self.series.setMesh(QAbstract3DSeries.MeshCylinder)   #棒柱样式 MeshBar,MeshPyramid,MeshCylinder
      self.series.setItemLabelFormat("(@rowLabel,@colLabel): %.1f")  #棒柱的标签显示格式
      self.series.setName("3D柱状图序列")  #在setItemLabelFormat里可引用@seriesName
      self.graph3D.addSeries(self.series)  #添加序列到图表

      self.dataProxy=QBarDataProxy()  #创建数据代理
      for j in range(3):   #3行
         rowItems = []     # 一行的 QBarDataItem 列表
         for i in range(7):   #7列
            value=random.uniform(8,16)    #均匀分布
            item=QBarDataItem(value)      #每个棒柱对应一个QBarDataItem对象
            rowItems.append(item)  
         rowStr="Week%d"%(j+1)   #行标签
         self.dataProxy.addRow(rowItems,rowStr) #添加行，以及行标签

      self.dataProxy.setColumnLabels(colLabs)   #设置列标签
      
      self.series.setDataProxy(self.dataProxy)  #设置数据代理
      self.series.selectedBarChanged.connect(self.do_barSelected)

##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        

##===工具栏actions==
   @pyqtSlot()   ## "序列基本颜色"
   def on_actSeries_BaseColor_triggered(self):
      color=self.series.baseColor()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.series.setBaseColor(color)

   @pyqtSlot()   ## "修改数值"
   def on_actBar_ChangeValue_triggered(self):
      position=self.series.selectedBar()
      if position.x()<0 or position.y()<0:
         return   #必须加此判断
      bar=self.dataProxy.itemAt(position)  #QBarDataItem
      value=bar.value()
      newValue,OK = QInputDialog.getDouble(self,
                     "输入数值","更改棒柱数值",value, 0, 50, 1)
      if OK:
         bar.setValue(newValue)
         self.dataProxy.setItem(position,bar)
      
   @pyqtSlot()   ## "添加行"
   def on_actData_Add_triggered(self):
      rowLabel,OK = QInputDialog.getText(self,"输入字符串","请输入行标签")
      rowLabel=rowLabel.strip()
      if (not OK) or  (rowLabel==""):
         return

      rowItems=[]
      for i in range(7):  #一周7天
         value=random.uniform(10,20)   #均匀分布
         item=QBarDataItem(value)      #创建棒柱数据对象
         rowItems.append(item) 
      self.dataProxy.addRow(rowItems,rowLabel)   #添加行数据，以及行标签

   @pyqtSlot()   ## "插入行"
   def on_actData_Insert_triggered(self):
      rowLabel,OK = QInputDialog.getText(self,"输入字符串","请输入行标签")
      rowLabel=rowLabel.strip()
      if (not OK) or  (rowLabel==""):
         return

      position=self.series.selectedBar()  #当前的棒柱
      if position.x()<0:
         rowIndex=0
      else:
         rowIndex=position.x()
      
      rowItems=[]
      for i in range(7):  #一周7天
         value=random.uniform(5,10)   #均匀分布
         item=QBarDataItem(value)     #创建棒柱数据对象
         rowItems.append(item)
      self.dataProxy.insertRow(rowIndex,rowItems,rowLabel)   #插入行


   @pyqtSlot()   ## "删除行"
   def on_actData_Delete_triggered(self):
      position=self.series.selectedBar()
      if position.x()<0 or position.y()<0:
         return   #必须加此判断
      rowIndex=position.x()
      removeCount=1        #删除的行数
      removeLabels=True    #是否删除行标签
      self.dataProxy.removeRows(rowIndex,removeCount,removeLabels)
         
   
##===三维旋转===      
   @pyqtSlot(int)   ## "预设视角"
   def on_comboCamera_currentIndexChanged(self,index):
      cameraPos=Q3DCamera.CameraPreset(index)
      self.graph3D.scene().activeCamera().setCameraPreset(cameraPos)
      
   @pyqtSlot(int)   ## "水平旋转"
   def on_sliderH_valueChanged(self,value):
      xRot=self.ui.sliderH.value()     #水平
      self.graph3D.scene().activeCamera().setXRotation(xRot)

   @pyqtSlot(int)   ## "垂直旋转"
   def on_sliderV_valueChanged(self,value):
      yRot=self.ui.sliderV.value()     #垂直
      self.graph3D.scene().activeCamera().setYRotation(yRot)

   @pyqtSlot(int)   ## "缩放"
   def on_sliderZoom_valueChanged(self,value):
      zoom=self.ui.sliderZoom.value()  #缩放
      self.graph3D.scene().activeCamera().setZoomLevel(zoom)


   @pyqtSlot()   ## 复位到FrontHigh视角
   def on_btnResetCamera_clicked(self):
      cameraPos=Q3DCamera.CameraPresetFrontHigh   
      self.graph3D.scene().activeCamera().setCameraPreset(cameraPos)
      
   @pyqtSlot()   ## "左移"
   def on_btnMoveLeft_clicked(self):
      target3D=self.graph3D.scene().activeCamera().target()  #QVector3D
      x=target3D.x()
      target3D.setX(x+0.1)
      self.graph3D.scene().activeCamera().setTarget(target3D)

   @pyqtSlot()   ## "右移"
   def on_btnMoveRight_clicked(self):
      target3D=self.graph3D.scene().activeCamera().target()  #QVector3D
      x=target3D.x()
      target3D.setX(x-0.1)
      self.graph3D.scene().activeCamera().setTarget(target3D)

   @pyqtSlot()   ## "上移"
   def on_btnMoveUp_clicked(self):
      target3D=self.graph3D.scene().activeCamera().target()  #QVector3D
      z=target3D.z()
      target3D.setZ(z-0.1)
      self.graph3D.scene().activeCamera().setTarget(target3D)

   @pyqtSlot()   ## "下移"
   def on_btnMoveDown_clicked(self):
      target3D=self.graph3D.scene().activeCamera().target()  #QVector3D
      z=target3D.z()
      target3D.setZ(z+0.1)
      self.graph3D.scene().activeCamera().setTarget(target3D)
      

##====图表总体      
   @pyqtSlot(int)   ## "主题"
   def on_cBoxTheme_currentIndexChanged(self,index):
      theme=Q3DTheme.Theme(index)
      self.graph3D.activeTheme().setType(theme)
      
   @pyqtSlot(int)   ## "字体大小"
   def on_spinFontSize_valueChanged(self,arg1):
      font = self.graph3D.activeTheme().font()
      font.setPointSize(arg1)
      self.graph3D.activeTheme().setFont(font)

   @pyqtSlot(int)   ## "选择模式"
   def on_cBoxSelectionMode_currentIndexChanged(self,index):
      if index<=7:  #前面8个直接用枚举类型转换
         mode=QAbstract3DGraph.SelectionFlags(index)
      elif  index==8:  # row slice
         mode=QAbstract3DGraph.SelectionItemAndRow | QAbstract3DGraph.SelectionSlice
      elif  index==9:  # column slice
         mode=QAbstract3DGraph.SelectionItemAndColumn | QAbstract3DGraph.SelectionSlice
      self.graph3D.setSelectionMode(mode)
      
      
   @pyqtSlot(bool)   ## "显示背景"
   def on_chkBoxBackground_clicked(self,checked):
      self.graph3D.activeTheme().setBackgroundEnabled(checked)
         
   @pyqtSlot(bool)   ## "显示背景的网格"
   def on_chkBoxGrid_clicked(self,checked):
      self.graph3D.activeTheme().setGridEnabled(checked)

   @pyqtSlot(bool)   ## "数值坐标轴反向"
   def on_chkBoxReverse_clicked(self,checked):
      self.graph3D.valueAxis().setReversed(checked)

   @pyqtSlot(bool)   ## "显示倒影"
   def on_chkBoxReflection_clicked(self,checked):
      self.graph3D.setReflection(checked)
      
   @pyqtSlot(bool)   ## "轴标题可见"
   def on_chkBoxAxisTitle_clicked(self,checked):
      self.graph3D.valueAxis().setTitleVisible(checked)
      self.graph3D.rowAxis().setTitleVisible(checked)
      self.graph3D.columnAxis().setTitleVisible(checked)

   @pyqtSlot(bool)   ## "轴标签背景可见"
   def on_chkBoxAxisBackground_clicked(self,checked):
      self.graph3D.activeTheme().setLabelBackgroundEnabled(checked)

##===序列设置
   @pyqtSlot(int)   ## "棒柱样式"
   def on_cBoxBarStyle_currentIndexChanged(self,index):
      mesh=QAbstract3DSeries.Mesh(index+1)   # 0=MeshUserDefined
      self.series.setMesh(mesh)
      
   @pyqtSlot(bool)   ## "光滑效果"
   def on_chkBoxSmooth_clicked(self,checked):
      self.series.setMeshSmooth(checked)

   @pyqtSlot(bool)   ## "选中棒柱的标签可见"
   def on_chkBoxItemLabel_clicked(self,checked):
      self.series.setItemLabelVisible(checked)

        
##  =============自定义槽函数===============================        
   def do_barSelected(self,position):   ##选择一个棒柱时触发
      if position.x()<0 or position.y()<0:   #无选中的棒柱
         self.ui.actBar_ChangeValue.setEnabled(False)
         return   #必须加此判断
      
      self.ui.actBar_ChangeValue.setEnabled(True)
      bar=self.series.dataProxy().itemAt(position)  #QBarDataItem
      info="选中的棒柱，Row=%d, Column=%d, Value=%.1f"%(
            position.x(),position.y(),bar.value())
      self.ui.statusBar.showMessage(info)
      

##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
