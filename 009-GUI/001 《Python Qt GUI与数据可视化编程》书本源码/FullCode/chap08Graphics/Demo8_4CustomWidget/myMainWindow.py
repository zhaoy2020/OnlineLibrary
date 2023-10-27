# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow

##from PyQt5.QtCore import  pyqtSlot,pyqtSignal,Qt

##from PyQt5.QtWidgets import  

##from PyQt5.QtGui import

##from PyQt5.QtSql import 

##from PyQt5.QtMultimedia import

##from PyQt5.QtMultimediaWidgets import


from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setWindowTitle("Demo8_4, 自定义界面组件")
      self.ui.slider.setRange(0,100)
      self.ui.LabInfo.setMaximumHeight(20)
      self.ui.battery.powerLevelChanged.connect(self.do_battery_changed)
      self.ui.slider.setValue(60)
      
##  ==============自定义功能函数========================


##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
   def on_slider_valueChanged(self,value):
      self.ui.battery.setPowerLevel(value)
      
        
##  =============自定义槽函数===============================        
   def do_battery_changed(self,power):
      powStr="当前电量：%d %%"%power
      self.ui.LabInfo.setText(powStr)
      
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
