# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QWidget,qApp

from PyQt5.QtCore import  QFile 

##from PyQt5.QtCore import  pyqtSlot,pyqtSignal,Qt,QFile 

##from PyQt5.QtWidgets import  

##from PyQt5.QtGui import

##from PyQt5.QtSql import 

##from PyQt5.QtMultimedia import

##from PyQt5.QtMultimediaWidgets import



from ui_Widget import Ui_Widget


class QmyWidget(QWidget): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

##      self.ui.editRequired.setProperty("required","true")
##      self.setStyleSheet("QLineEdit { background-color: red }")
##      self.ui.editName.setStyleSheet("color: blue;"
##	                     "background-color: lime;"
##	                     "selection-color: yellow;"
##	                     "selection-background-color: red;")
  


##  ==============自定义功能函数========================


##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
        
        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序

##   qApp.setStyleSheet("QLineEdit { background-color: gray }")

   file=QFile("myStyle.qss")
   file.open(QFile.ReadOnly)
   qtBytes=file.readAll()     # QByteArray
   pyBytes=qtBytes.data()     # QByteArray转换为bytes

   styleStr=pyBytes.decode("utf-8")  #bytes转换为str
   app.setStyleSheet(styleStr)
   
   form=QmyWidget()            #创建窗体
   form.show()
   sys.exit(app.exec_())
