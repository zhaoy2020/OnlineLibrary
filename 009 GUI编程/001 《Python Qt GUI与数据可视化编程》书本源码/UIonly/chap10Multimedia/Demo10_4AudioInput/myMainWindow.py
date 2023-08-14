import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QMessageBox

from PyQt5.QtCore import  pyqtSlot,pyqtSignal, QIODevice,QFile

##from PyQt5.QtWidgets import  

##from PyQt5.QtGui import

##from PyQt5.QtSql import 

from PyQt5.QtMultimedia import QAudioDeviceInfo, QAudio,QAudioFormat,QAudioInput

from ui_MainWindow import Ui_MainWindow

##from myAudioReader import QmyAudioReader

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      pass


##  ==============自定义功能函数========================
   def __getSampleTypeStr(self,sampleType):     ##采样数据点类型的字符串表示
      pass


   def __getByteOrderStr(self,endian):    ##字节序的字符串表示
      pass
      

##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============
      
   @pyqtSlot(int)   ##选择音频输入设备
   def on_comboDevices_currentIndexChanged(self,index):        
      pass


   @pyqtSlot()   ##使用内建IODevice
   def on_radioSaveMode_Inner_clicked(self):
      pass
      
   @pyqtSlot()   ##使用QFile对象(test.raw)
   def on_radioSaveMode_QFile_clicked(self):
      pass

   @pyqtSlot(int)   ##调节录音音量
   def on_sliderVolumn_valueChanged(self,value):
      pass


   @pyqtSlot()   ##测试音频输入设备是否支持选择的设置
   def on_actDeviceTest_triggered(self):
      pass


   @pyqtSlot()   ##开始音频输入
   def on_actStart_triggered(self):
      pass


   @pyqtSlot()   ##停止音频输入
   def on_actStop_triggered(self):
      pass

        
##  =============自定义槽函数===============================        
##1. 使用QIODevice* start()返回的内置的IODevice, pull mode，利用readyRead()信号读出数据
   def do_IO_readyRead(self):    ##内建IODevice，读取缓冲区数据
      pass


   def do_stateChanged(self,state):    ##设备状态变化
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
