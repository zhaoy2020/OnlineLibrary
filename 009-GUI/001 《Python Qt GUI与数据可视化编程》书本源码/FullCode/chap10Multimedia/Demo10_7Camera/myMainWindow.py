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

      self.__LabCameraState=QLabel("摄像头state:")
      self.__LabCameraState.setMinimumWidth(150)
      self.ui.statusBar.addWidget(self.__LabCameraState)

      self.__LabImageID=QLabel("图片文件ID:") 
      self.__LabImageID.setMinimumWidth(100)
      self.ui.statusBar.addWidget(self.__LabImageID)

      self.__LabImageFile=QLabel("")   #保存的图片文件名
##      self.ui.statusBar.addWidget(self.__LabImageFile)
      self.ui.statusBar.addPermanentWidget(self.__LabImageFile)

      self.camera=None   #QCamera对象
      cameras = QCameraInfo.availableCameras()  #list[QCameraInfo]
      if len(cameras)>0:
         self.__iniCamera()       #初始化摄像头
         self.__iniImageCapture() #初始化静态画图
         self.camera.start()

##  ==============自定义功能函数========================
   def __iniCamera(self):   ##创建 QCamera对象
      camInfo=QCameraInfo.defaultCamera()    #获取缺省摄像头,QCameraInfo
      self.ui.comboCamera.addItem(camInfo.description())    #摄像头描述
      self.ui.comboCamera.setCurrentIndex(0)

      self.camera=QCamera(camInfo)  #创建摄像头对象
      self.camera.setViewfinder(self.ui.viewFinder)   #设置取景框预览
   ##          camera.setCaptureMode(QCamera.CaptureViewfinder) #预览
      self.camera.setCaptureMode(QCamera.CaptureStillImage)  #设置为抓图
   ##          camera.setCaptureMode(QCamera.CaptureVideo)

      mode=QCamera.CaptureStillImage
      supported=self.camera.isCaptureModeSupported(mode)
      self.ui.checkStillImage.setChecked(supported) #支持拍照

      supported=self.camera.isCaptureModeSupported(QCamera.CaptureVideo)
      self.ui.checkVideo.setChecked(supported)     #支持视频录制

      supported=self.camera.exposure().isAvailable()
      self.ui.checkExposure.setChecked(supported) #支持曝光补偿
      
      supported=self.camera.focus().isAvailable()
      self.ui.checkFocus.setChecked(supported)    #支持变焦
      
      self.camera.stateChanged.connect(self.do_cameraStateChanged)

   def __iniImageCapture(self):  ##创建 QCameraImageCapture对象
      self.capturer = QCameraImageCapture(self.camera)
      settings=QImageEncoderSettings()    #拍照设置
      settings.setCodec("image/jpeg")     #设置抓图图形编码
      settings.setResolution(640, 480)    #分辨率
      settings.setQuality(QMultimedia.HighQuality)    #图片质量
      self.capturer.setEncodingSettings(settings) 

      self.capturer.setBufferFormat(QVideoFrame.Format_Jpeg)   #缓冲区格式

      if self.ui.chkBoxSaveToFile.isChecked():
         dest=QCameraImageCapture.CaptureToFile    #保存到文件
      else:
         dest=QCameraImageCapture.CaptureToBuffer  #保存到缓冲区
      self.capturer.setCaptureDestination(dest)    #保存目标
      
      self.capturer.readyForCaptureChanged.connect(self.do_imageReady)
      
      self.capturer.imageCaptured.connect(self.do_imageCaptured)
      
      self.capturer.imageSaved.connect(self.do_imageSaved)
      
##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
   @pyqtSlot(bool)   ##设置保存方式
   def on_chkBoxSaveToFile_clicked(self,checked):
      if checked:
         dest=QCameraImageCapture.CaptureToFile    #保存到文件
      else:
         dest=QCameraImageCapture.CaptureToBuffer  #保存到缓冲区
      self.capturer.setCaptureDestination(dest)    #保存目标

   @pyqtSlot()   ##拍照
   def on_actCapture_triggered(self): 
      QSound.play("shutter.wav")    #播放快门音效
      self.camera.searchAndLock()   #快门半按下时锁定摄像头参数
      self.capturer.capture()       #拍照
      self.camera.unlock()          #快门按钮释放时解除锁定

   @pyqtSlot()   ##打开摄像头
   def on_actStartCamera_triggered(self):
      self.camera.start()

   @pyqtSlot()  ##关闭摄像头
   def on_actStopCamera_triggered(self):
      self.camera.stop()
      
   
        
   ##  =============自定义槽函数===============================        
   def do_cameraStateChanged(self,state):    ##摄像头状态变化
      if (state==QCamera.UnloadedState):
         self.__LabCameraState.setText("摄像头state: UnloadedState")
      elif (state==QCamera.LoadedState):
         self.__LabCameraState.setText("摄像头state: LoadedState")
      elif (state==QCamera.ActiveState):
         self.__LabCameraState.setText("摄像头state: ActiveState")
         
      self.ui.actStartCamera.setEnabled(state!=QCamera.ActiveState)
      self.ui.actStopCamera.setEnabled(state==QCamera.ActiveState)

   def do_imageReady(self,ready):   ##是否可以拍照了
      self.ui.actCapture.setEnabled(ready)

   def do_imageCaptured(self,imageID,preview): ##图片被抓取到内存
                                          #preview是 QImage
      H=self.ui.LabImage.height()
      W=self.ui.LabImage.width()
      
      scaledImage = preview.scaled(W,H,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
      self.ui.LabImage.setPixmap(QPixmap.fromImage(scaledImage))
      self.__LabImageID.setText("图片文件ID:%d"%imageID)
      self.__LabImageFile.setText("图片保存为： ")

   def do_imageSaved(self,imageID,fileName):    ##图片被保存
      self.__LabImageID.setText("图片文件ID:%d"%imageID)
      self.__LabImageFile.setText("图片保存为： "+fileName)
      
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
