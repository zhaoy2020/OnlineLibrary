# -*- coding: utf-8 -*-

import sys,math

from PyQt5.QtWidgets import  (QApplication, QMainWindow, QWidget,
                  QSplitter, QColorDialog,QInputDialog,QLineEdit)

from PyQt5.QtCore import  pyqtSlot,Qt

from PyQt5.QtGui import QVector3D, QLinearGradient

from PyQt5.QtDataVisualization import *


from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo13_3, Q3DSurface和QSurface3DSeries 绘制三维曲面")
      self.__iniGraph3D() ##创建图表

      self.ui.sliderZoom.setRange(10,500)    #缺省缩放范围，100=原始大小
      self.ui.sliderH.setRange(-180,180)     #水平旋转角度范围
      self.ui.sliderV.setRange(-180,180)     #垂直旋转角度范围

      splitter=QSplitter(Qt.Horizontal)
      splitter.addWidget(self.ui.frameSetup)    #左侧控制面板
      splitter.addWidget(self.__container)      #右侧图表

      self.setCentralWidget(splitter)        #设置主窗口中心组件
      

##  ==============自定义功能函数========================
   def __iniGraph3D(self):    ##创建3D图表
      self.graph3D = Q3DSurface()
      self.__container = QWidget.createWindowContainer(self.graph3D)    #继承自QWindow，必须如此创建
      self.graph3D.activeTheme().setLabelBackgroundEnabled(False)

      self.dataProxy = QSurfaceDataProxy()   #数据代理  QSurfaceDataProxy
      self.series = QSurface3DSeries(self.dataProxy)     #创建序列 QSurface3DSeries
      self.series.setItemLabelFormat("(x,z,y)=(@xLabel,@zLabel,@yLabel)")
      self.series.setMeshSmooth(True)        #单点光滑显示
      self.series.setMesh(QAbstract3DSeries.MeshSphere)  #单点样式
      self.graph3D.addSeries(self.series)

   ##  创建坐标轴
      axisX=QValue3DAxis()    # QValue3DAxis
      axisX.setTitle("Axis X")
      axisX.setTitleVisible(True)
      axisX.setRange(-11,11)
      ##    axisX.setAutoAdjustRange(True)
      self.graph3D.setAxisX(axisX)

      axisY=QValue3DAxis()   
      axisY.setTitle("Axis Y")   #垂直方向的坐标轴
      axisY.setTitleVisible(True)
      ##    axisY.setRange(-10,10)
      axisY.setAutoAdjustRange(True)   #垂直方向自动调整范围
      self.graph3D.setAxisY(axisY)

      axisZ=QValue3DAxis()
      axisZ.setTitle("Axis Z")
      axisZ.setTitleVisible(True)
      axisZ.setRange(-11,11)
      ##    axisZ.setAutoAdjustRange(True)
      self.graph3D.setAxisZ(axisZ)


   ## 墨西哥草帽,-10:0.5:10， N=41
      N=41
      x=-10.0
      for i in range(1,N+1):  
         itemRow = []     # 一行的 QSurfaceDataItem 列表
         y=-10.0
         for  j in range(1,N+1):   
            z=math.sqrt(x*x+y*y)
            if z!=0:
               z=10*math.sin(z)/z
            else:
               z=10
            vect3D=QVector3D(x,z,y)  #三维坐标点
            item=QSurfaceDataItem(vect3D)    #QSurfaceDataItem对象
            itemRow.append(item)    #一行的QSurfaceDataItem对象
            y=y+0.5
         x=x+0.5
         self.dataProxy.addRow(itemRow)  #每次添加一行
      
      self.series.selectedPointChanged.connect(self.do_pointSelected)


##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        

##===工具栏actions==
   @pyqtSlot()   ## "曲面颜色"
   def on_actSurf_Color_triggered(self):
      color=self.series.baseColor()
      color=QColorDialog.getColor(color)
      if color.isValid():
         self.series.setBaseColor(color)
         self.series.setColorStyle(Q3DTheme.ColorStyleUniform)

   @pyqtSlot()   ## "渐变颜色一"
   def on_actSurf_GradColor1_triggered(self):
      gr=QLinearGradient()
      gr.setColorAt(0.0, Qt.black)
      gr.setColorAt(0.33, Qt.blue)
      gr.setColorAt(0.67, Qt.red)
      gr.setColorAt(1.0, Qt.yellow)
      self.series.setBaseGradient(gr)
      self.series.setColorStyle(Q3DTheme.ColorStyleRangeGradient)

   @pyqtSlot()   ## "渐变颜色二"
   def on_actSurf_GradColor2_triggered(self):
      grGtoR=QLinearGradient()
      grGtoR.setColorAt(1.0, Qt.darkGreen)
      grGtoR.setColorAt(0.5, Qt.yellow)
      grGtoR.setColorAt(0.2, Qt.red)
      grGtoR.setColorAt(0.0, Qt.darkRed)
      self.series.setBaseGradient(grGtoR)
      self.series.setColorStyle(Q3DTheme.ColorStyleRangeGradient)

   @pyqtSlot()   ## "修改点坐标"
   def on_actPoint_Modify_triggered(self):
      position=self.series.selectedPoint()   # QPoint
      if position.x()<0 or position.y()<0:   #必须加此判断
         return   

      item=self.dataProxy.itemAt(position)   # QSurfaceDataItem 对象
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
      self.dataProxy.setItem(position,item)

   @pyqtSlot()   ## "删除行"
   def on_actPoint_DeleteRow_triggered(self):
      position=self.series.selectedPoint()  # QPoint
      if position.x()<0 or position.y()<0:  #必须加此判断
         return   
      removeCount=1  #删除行的个数
      self.dataProxy.removeRows(position.x(),removeCount)  
         
   
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

   @pyqtSlot(int)   ## "选择模式"，除了这几种，其他选择模式无效
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
         self.graph3D.setShadowQuality(QAbstract3DGraph.ShadowQualityMedium)
      else:
         self.graph3D.setShadowQuality(QAbstract3DGraph.ShadowQualityNone)
      
      
   @pyqtSlot(bool)   ## "显示轴标题"
   def on_chkBoxAxisTitle_clicked(self,checked):
      self.graph3D.axisX().setTitleVisible(checked)
      self.graph3D.axisY().setTitleVisible(checked)
      self.graph3D.axisZ().setTitleVisible(checked)

   @pyqtSlot(bool)   ## "显示轴标签背景"
   def on_chkBoxAxisBackground_clicked(self,checked):
      self.graph3D.activeTheme().setLabelBackgroundEnabled(checked)

##===曲面序列设置
   @pyqtSlot(int)   ## "曲面样式"
   def on_comboDrawMode_currentIndexChanged(self,index):
      if index==0:
         self.series.setDrawMode(QSurface3DSeries.DrawWireframe)
      elif index==1:
         self.series.setDrawMode(QSurface3DSeries.DrawSurface)
      else:
         self.series.setDrawMode(QSurface3DSeries.DrawSurfaceAndWireframe)
      
   @pyqtSlot(int)   ## "单点形状"
   def on_cBoxBarStyle_currentIndexChanged(self,index):
      mesh=QAbstract3DSeries.Mesh(index+1)   # 0=MeshUserDefined
      self.series.setMesh(mesh)

   @pyqtSlot(bool)   ## "光滑效果"
   def on_chkBoxSmooth_clicked(self,checked):
      self.series.setMeshSmooth(checked)

   @pyqtSlot(bool)   ## "单点标签可见"
   def on_chkBoxItemLabel_clicked(self,checked):
      self.series.setItemLabelVisible(checked)

   @pyqtSlot(bool)   ## "Flat Shading"
   def on_chkBoxFlatShading_clicked(self,checked):
      self.series.setFlatShadingEnabled(checked)
        
##  =============自定义槽函数===============================        
   def do_pointSelected(self,position):   ##选择一个点时触发
      if position.x()<0 or position.y()<0:      #必须加此判断
         self.ui.actPoint_Modify.setEnabled(False)
         self.ui.actPoint_DeleteRow.setEnabled(False)
         self.ui.statusBar.showMessage("没有选中点")
         return  
      
      self.ui.actPoint_Modify.setEnabled(True)
      self.ui.actPoint_DeleteRow.setEnabled(True)
      item=self.dataProxy.itemAt(position)   # QSurfaceDataItem 对象
      info="选中点的坐标，(x,z,y)=(%.2f, %.2f, %.2f)"%(
            item.x(),item.z(),item.y())
      self.ui.statusBar.showMessage(info)
      

##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
