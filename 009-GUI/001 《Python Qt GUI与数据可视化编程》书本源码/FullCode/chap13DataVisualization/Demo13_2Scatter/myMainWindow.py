# -*- coding: utf-8 -*-

import sys,math,random

from PyQt5.QtWidgets import  (QApplication, QMainWindow, QWidget,
               QSplitter, QColorDialog, QInputDialog, QLineEdit)

from PyQt5.QtCore import  pyqtSlot,Qt

from PyQt5.QtGui import QVector3D

from PyQt5.QtDataVisualization import *

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo13_2, Q3DScatter和QScatter3DSeries 绘制三维散点图")
      self.__iniGraph3D()     #创建图表

      self.ui.sliderZoom.setRange(10,500)  #缺省缩放范围，100=原始大小
      self.ui.sliderH.setRange(-180,180)   #水平旋转角度范围
      self.ui.sliderV.setRange(-180,180)   #垂直旋转角度范围

      splitter=QSplitter(Qt.Horizontal)
      splitter.addWidget(self.ui.frameSetup)    #左侧控制面板
      splitter.addWidget(self.__container)      #右侧图表

      self.setCentralWidget(splitter)     #设置主窗口中心组建
      

##  ==============自定义功能函数========================
   def __iniGraph3D(self):    ##创建3D图表
      self.graph3D = Q3DScatter()
      self.__container = QWidget.createWindowContainer(self.graph3D)    #继承自QWindow，必须如此创建

      self.dataProxy = QScatterDataProxy()     #数据代理  QScatterDataProxy
      self.series = QScatter3DSeries(self.dataProxy)   #创建序列 QScatter3DSeries
      self.series.setItemLabelFormat("(x,z,y)=(@xLabel,@zLabel,@yLabel)")
      self.series.setMeshSmooth(True) 
      self.graph3D.addSeries(self.series)

   ##  设置坐标轴，使用内建的坐标轴
      self.graph3D.axisX().setTitle("axis X")
      self.graph3D.axisX().setTitleVisible(True)

      self.graph3D.axisY().setTitle("axis Y")
      self.graph3D.axisY().setTitleVisible(True)

      self.graph3D.axisZ().setTitle("axis Z")
      self.graph3D.axisZ().setTitleVisible(True)

      self.graph3D.activeTheme().setLabelBackgroundEnabled(False)

      self.series.setMesh(QAbstract3DSeries.MeshSphere)  #散点形状
      self.series.setItemSize(0.15)    # 散点大小 default 0. value 0~1

   ## 墨西哥草帽,-10:0.5:10， N=41
      N=41
      itemCount=N*N
      itemArray=[]  #QScatterDataItem 的列表
      
      x=-10.0
      for i in range(1,N+1):  
         y=-10.0
         for  j in range(1,N+1):  
            z=math.sqrt(x*x+y*y)
            if z!=0:
               z=10*math.sin(z)/z
            else:
               z=10
            vect3D=QVector3D(x,z,y)  #三维坐标点
            item=QScatterDataItem(vect3D)  #空间一个散点对象 QScatterDataItem
            itemArray.append(item)
            y=y+0.5
         x=x+0.5

      self.dataProxy.resetArray(itemArray)  #重置数组
      self.series.selectedItemChanged.connect(self.do_itemSelected)


##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        

##===工具栏actions==
   @pyqtSlot()   ## "序列基本颜色"
   def on_actSeries_BaseColor_triggered(self):
      color=self.series.baseColor()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.series.setBaseColor(color)

   @pyqtSlot()   ## "修改散点坐标"
   def on_actPoint_ChangeValue_triggered(self):
      index=self.series.selectedItem()    #当前散点的索引
      if index<0:
         return   #必须加此判断

      item=self.dataProxy.itemAt(index)  # QScatterDataItem 对象
      coord="%.2f, %.2f, %.2f"%(item.x(),item.z(),item.y())
      newText,OK = QInputDialog.getText(self,"修改散点坐标",
               "按格式输入点的坐标（x,z,y）",QLineEdit.Normal,coord)
      if not OK:
         return

      newText=newText.strip()
      xzy=newText.split(',')  #按逗号分割
      if len(xzy) != 3:
         QMessageBox.critical(self,"错误","输入坐标数据格式错误")
         return

      item.setX(float(xzy[0]))
      item.setZ(float(xzy[1]))
      item.setY(float(xzy[2]))
      self.dataProxy.setItem(index,item)

      
   @pyqtSlot()   ## "随机添加点"
   def on_actData_Add_triggered(self):
      x=random.uniform(-10,10)
      z=random.uniform(-10,10)
      y=random.uniform(5,10)
      coord="%.2f, %.2f, %.2f"%(x,z,y)
      newText,OK = QInputDialog.getText(self,"随机添加点",
                  "按格式输入点的坐标（x,z,y）",QLineEdit.Normal,coord)
      if not OK:
         return
      newText=newText.strip()
      xzy=newText.split(',')  #按逗号分割
      if len(xzy) != 3:
         QMessageBox.critical(self,"错误","输入坐标数据格式错误")
         return

      item=QScatterDataItem()
      item.setX(float(xzy[0]))
      item.setZ(float(xzy[1]))
      item.setY(float(xzy[2]))
      self.dataProxy.addItem(item)


   @pyqtSlot()   ## "删除当前点"
   def on_actData_Delete_triggered(self):
      index=self.series.selectedItem()    #索引
      if index<0:
         return   #必须加此判断

      removeCount=1  #删除点个数
      self.dataProxy.removeItems(index,removeCount)  
         
   
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


   @pyqtSlot()   ## "复位到FrontHigh视角"
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
      currentTheme = self.graph3D.activeTheme()   #Q3DTheme
      currentTheme.setType(Q3DTheme.Theme(index))
      
   @pyqtSlot(int)   ## "字体大小"
   def on_spinFontSize_valueChanged(self,arg1):
      font = self.graph3D.activeTheme().font()
      font.setPointSize(arg1)
      self.graph3D.activeTheme().setFont(font)

   @pyqtSlot(int)   ## "选择模式"对于散点图只有前2种有效
   def on_cBoxSelectionMode_currentIndexChanged(self,index):
      if index<=7:
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
      self.graph3D.axisY().setReversed(checked)

   @pyqtSlot(bool)   ## "显示阴影"
   def on_chkBoxShadow_clicked(self,checked):
      if checked:
         quality=QAbstract3DGraph.ShadowQualityMedium
      else:
         quality=QAbstract3DGraph.ShadowQualityNone
      self.graph3D.setShadowQuality(quality)
      
   @pyqtSlot(bool)   ## "轴标题可见"
   def on_chkBoxAxisTitle_clicked(self,checked):
      self.graph3D.axisX().setTitleVisible(checked)
      self.graph3D.axisY().setTitleVisible(checked)
      self.graph3D.axisZ().setTitleVisible(checked)

   @pyqtSlot(bool)   ## "轴标签背景可见"
   def on_chkBoxAxisBackground_clicked(self,checked):
      self.graph3D.activeTheme().setLabelBackgroundEnabled(checked)

##===序列设置
   @pyqtSlot(int)   ## "散点形状"
   def on_cBoxBarStyle_currentIndexChanged(self,index):
      mesh=QAbstract3DSeries.Mesh(index+1)   # 0=MeshUserDefined
      self.series.setMesh(mesh)
      
   @pyqtSlot(float)   ## "散点大小"
   def on_spinItemSize_valueChanged(self,arg1):
      self.series.setItemSize(arg1)    #default 0. value 0~1
      
   @pyqtSlot(bool)   ## "光滑效果"
   def on_chkBoxSmooth_clicked(self,checked):
      self.series.setMeshSmooth(checked)

   @pyqtSlot(bool)   ## "选中散点标签可见"
   def on_chkBoxItemLabel_clicked(self,checked):
      self.series.setItemLabelVisible(checked)

        
##  =============自定义槽函数===============================        
   def do_itemSelected(self,index):   ##点击选择一个散点时触发
      if index<0:
         self.ui.actPoint_ChangeValue.setEnabled(False)
         self.ui.actData_Delete.setEnabled(False)
         self.ui.statusBar.showMessage("没有选中点")
         return   #必须加此判断
      
      self.ui.actPoint_ChangeValue.setEnabled(True)
      self.ui.actData_Delete.setEnabled(True)
      item=self.dataProxy.itemAt(index)   #QScatterDataItem对象
      info="选中点的坐标，(x,z,y)=(%.2f, %.2f, %.2f)"%(
            item.x(),item.z(),item.y())
      self.ui.statusBar.showMessage(info)
      

##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   form=QmyMainWindow()             #创建窗体
   form.show()
   sys.exit(app.exec_())
