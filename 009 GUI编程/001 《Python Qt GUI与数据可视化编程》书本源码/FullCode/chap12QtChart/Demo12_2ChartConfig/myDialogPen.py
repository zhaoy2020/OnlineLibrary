# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QDialog,QColorDialog

from PyQt5.QtCore import  pyqtSlot,Qt

##from PyQt5.QtWidgets import   

from PyQt5.QtGui import QPen, QPalette,QColor

from ui_QWDialogPen import Ui_QWDialogPen

class QmyDialogPen(QDialog):

   def __init__(self, parent=None):
      super().__init__(parent) 
      self.ui=Ui_QWDialogPen()
      self.ui.setupUi(self)       #构造UI界面

      self.__pen=QPen()
      
##“线型”ComboBox的选择项设置
      self.ui.comboPenStyle.clear()
      self.ui.comboPenStyle.addItem("NoPen",0)
      self.ui.comboPenStyle.addItem("SolidLine",1)
      self.ui.comboPenStyle.addItem("DashLine",2)
      self.ui.comboPenStyle.addItem("DotLine",3)
      self.ui.comboPenStyle.addItem("DashDotLine",4)
      self.ui.comboPenStyle.addItem("DashDotDotLine",5)
      self.ui.comboPenStyle.addItem("CustomDashLine",6)

      self.ui.comboPenStyle.setCurrentIndex(1)
      
##=================自定义接口函数====================
   def setPen(self,pen):   ##设置pen
      self.__pen=pen  

      self.ui.spinWidth.setValue(pen.width()) #线宽
      i=int(pen.style())   #枚举类型转换为整型
      self.ui.comboPenStyle.setCurrentIndex(i)  

      color=pen.color()    #QColor
##      self.ui.btnColor.setAutoFillBackground(True) 
      qss="background-color: rgb(%d, %d, %d)"%(
            color.red(),color.green(),color.blue())
      self.ui.btnColor.setStyleSheet(qss) #使用样式表设置按钮背景色

   def getPen(self):   ##返回pen
      index=self.ui.comboPenStyle.currentIndex()
      self.__pen.setStyle(Qt.PenStyle(index))   #线型
      self.__pen.setWidth(self.ui.spinWidth.value())  #线宽

      color=self.ui.btnColor.palette().color(QPalette.Button)
      self.__pen.setColor(color)    #颜色
      return  self.__pen
   

   @staticmethod    ##类函数，或静态函数
   def staticGetPen(iniPen):
# 不能有参数self,不能与类的成员函数同名，也就是不能命名为getPen()
      Dlg=QmyDialogPen()   #创建一个对话框
      Dlg.setPen(iniPen)   #设置初始化QPen

      pen=iniPen
      ok=False
      
      ret=Dlg.exec()    #模态显示对话框
      if ret==QDialog.Accepted:
         pen=Dlg.getPen()  #获取pen
         ok=True    

      return  pen ,ok  #返回设置的QPen对象
   

##  ==========由connectSlotsByName()自动连接的槽函数============        
   @pyqtSlot()   ##选择颜色
   def on_btnColor_clicked(self):
      color=QColorDialog.getColor()
      if color.isValid():    #用样式表设置QPushButton的背景色
         qss="background-color: rgb(%d, %d, %d);"%(
            color.red(),color.green(),color.blue())
         self.ui.btnColor.setStyleSheet(qss)


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":      
   app = QApplication(sys.argv)  
   iniPen=QPen(Qt.blue)
   pen=QmyDialogPen.staticGetPen(iniPen)  #测试类函数调用
   sys.exit(app.exec_())
