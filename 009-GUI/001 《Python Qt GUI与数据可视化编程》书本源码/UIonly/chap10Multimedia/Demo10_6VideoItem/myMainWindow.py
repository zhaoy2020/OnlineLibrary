import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,QFileDialog,
         QGraphicsScene,QGraphicsItem,QGraphicsTextItem)

from PyQt5.QtCore import  pyqtSlot,QSizeF, QUrl,Qt,QFileInfo,QDir

from PyQt5.QtGui import QIcon,QFont

from PyQt5.QtMultimedia import QMediaContent,QMediaPlayer

from PyQt5.QtMultimediaWidgets import  QGraphicsVideoItem

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      pass


##  ==============自定义功能函数========================


##  ==============event处理函数==========================
   def closeEvent(self,event):  #窗体关闭时
   # 窗口关闭时不能自动停止播放，需手动停止
      if (self.player.state()==QMediaPlayer.PlayingState):
         self.player.stop()
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============        
   @pyqtSlot()    ##打开文件
   def on_btnOpen_clicked(self):
      pass

   @pyqtSlot()    ##播放
   def on_btnPlay_clicked(self):
      pass

   @pyqtSlot()    ##暂停
   def on_btnPause_clicked(self):
      pass

   @pyqtSlot()    ##停止
   def on_btnStop_clicked(self):
      pass

   @pyqtSlot()    ##全屏
   def on_btnFullScreen_clicked(self):
      pass

   @pyqtSlot()    ##静音按钮
   def on_btnSound_clicked(self):
      pass
         
   @pyqtSlot(int) ##音量调节
   def on_sliderVolumn_valueChanged(self,value):
      pass

   @pyqtSlot(int) ##播放进度调节
   def on_sliderPosition_valueChanged(self,value):
      pass

   @pyqtSlot()    ##放大
   def on_btnZoomIn_clicked(self):
      pass

   @pyqtSlot()    ##缩小
   def on_btnZoomOut_clicked(self):
      pass
      
   @pyqtSlot(bool)   ##弹幕
   def on_btnText_clicked(self,checked):
      pass
      
        
##  =============自定义槽函数===============================        
   def do_stateChanged(self,state):
      pass

   def do_durationChanged(self,duration):
      pass

   def do_positionChanged(self,position):
      pass

##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
