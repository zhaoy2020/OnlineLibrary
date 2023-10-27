import sys,os

from PyQt5.QtWidgets import  QApplication, QMainWindow,QFileDialog,QMessageBox

from PyQt5.QtCore import  pyqtSlot,QUrl

##from PyQt5.QtWidgets import 

##from PyQt5.QtGui import

from PyQt5.QtMultimedia import (QAudioRecorder,QAudioProbe,QMultimedia,
      QMediaRecorder,QAudioEncoderSettings,QAudioFormat)


from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      pass


##  ==============自定义功能函数============
   def __setRecordParams(self):  ##设置音频输入参数
      pass

           
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##"录音输出文件"按钮
   def on_btnGetFile_clicked(self):
      pass


   @pyqtSlot()    ##开始录音
   def on_actRecord_triggered(self): 
      pass


   @pyqtSlot()    ##暂停
   def on_actPause_triggered(self): 
      pass
      
   @pyqtSlot()    ##停止
   def on_actStop_triggered(self): 
      pass
      
        
##  =============自定义槽函数===============================        
   def do_stateChanged(self,state):  ##状态变化
      pass


   def do_durationChanged(self,duration):  ##持续时间长度变化
      pass
   
   def do_processBuffer(self,buffer):     ##解析缓冲区数据
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
