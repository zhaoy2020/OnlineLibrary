import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QLabel

from PyQt5.QtCore import  pyqtSlot,Qt

from PyQt5.QtGui import QImage,QPixmap

##from PyQt5.QtSql import 

from PyQt5.QtMultimedia import (QCameraInfo,QCameraImageCapture,
      QImageEncoderSettings,QMultimedia,QVideoFrame,QSound,QCamera)

##from PyQt5.QtMultimediaWidgets import QCameraViewfinder

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      pass


##  ==============自定义功能函数========================
   def __iniCamera(self):   ##创建 QCamera对象
      pass


   def __iniImageCapture(self):  ##创建 QCameraImageCapture对象
      pass

      
##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
   @pyqtSlot(bool)   ##设置保存方式
   def on_chkBoxSaveToFile_clicked(self,checked):
      pass


   @pyqtSlot()   ##拍照
   def on_actCapture_triggered(self): 
      pass


   @pyqtSlot()   ##打开摄像头
   def on_actStartCamera_triggered(self):
      pass
   

   @pyqtSlot()  ##关闭摄像头
   def on_actStopCamera_triggered(self):
      pass
      
        
##  =============自定义槽函数===============================        
   def do_cameraStateChanged(self,state):    ##摄像头状态变化
      pass

   def do_imageReady(self,ready):   ##是否可以拍照了
      pass

   def do_imageCaptured(self,imageID,preview): ##图片被抓取到内存
      pass

   def do_imageSaved(self,imageID,fileName):    ##图片被保存
      pass
      
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
