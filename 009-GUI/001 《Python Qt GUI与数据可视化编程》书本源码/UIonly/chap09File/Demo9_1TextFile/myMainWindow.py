# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow

from PyQt5.QtWidgets import  QFileDialog,QMessageBox

from PyQt5.QtCore import  pyqtSlot,QDir,QFile,QIODevice,QTextStream

##from PyQt5.QtGui import


from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      self.setCentralWidget(self.ui.textEdit)
      

##  ==============自定义功能函数========================
   def __openByIODevice(self,fileName):   ##用QFile打开文件
      pass


   def __saveByIODevice(self,fileName):   ##用QFile保存文件
      pass


   def __openByStream(self,fileName):     ##用QTextStream打开文件
      pass


   def __saveByStream(self,fileName):     ##用 QTextStream 保存文件
      pass
      

##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============
   
   @pyqtSlot()   ##用QFile 打开文件
   def on_actQFile_Open_triggered(self):
      pass


   @pyqtSlot()   ##用QFile 另存文件
   def on_actQFile_Save_triggered(self):
      pass


   @pyqtSlot()   ##用QTextStream 打开文件
   def on_actStream_Open_triggered(self):
      pass


   @pyqtSlot()   ##用QTextStream 另存文件
   def on_actStream_Save_triggered(self):
      pass


   @pyqtSlot()   ##用 python 的file()打开文件
   def on_actPY_Open_triggered(self):
      pass

         
   @pyqtSlot()   ##用 python的file() 保存文件
   def on_actPY_Save_triggered(self):
      pass

        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
