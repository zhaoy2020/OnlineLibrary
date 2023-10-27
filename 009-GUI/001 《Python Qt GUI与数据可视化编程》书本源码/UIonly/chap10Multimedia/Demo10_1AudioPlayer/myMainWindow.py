import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QFileDialog,QListWidgetItem

from PyQt5.QtCore import  pyqtSlot,QUrl,QModelIndex,QDir,QFileInfo

from PyQt5.QtGui import QIcon

from PyQt5.QtMultimedia import QMediaPlayer, QMediaPlaylist,QMediaContent


from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      pass

##  ==============自定义功能函数============



##  ===============event 处理函数==========
   def closeEvent(self,event):  ##窗体关闭时
##  窗口关闭时不能自动停止播放，需手动停止
      pass

         
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
#  播放列表管理
   @pyqtSlot()    ##添加文件
   def on_btnAdd_clicked(self):  
      pass

      
   @pyqtSlot()    ##移除一个文件
   def on_btnRemove_clicked(self): 
      pass


   @pyqtSlot()    ##清空播放列表
   def on_btnClear_clicked(self):  
      pass

      
##   @pyqtSlot()    ##双击时切换播放文件
   def on_listWidget_doubleClicked(self,index):  
      pass
      

##  播放控制
   @pyqtSlot()    ##播放
   def on_btnPlay_clicked(self):  
      pass
   

   @pyqtSlot()    ##暂停
   def on_btnPause_clicked(self):  
      pass
   

   @pyqtSlot()    ##停止
   def on_btnStop_clicked(self):  
      pass
   

   @pyqtSlot()    ##上一曲目
   def on_btnPrevious_clicked(self):  
      pass
   

   @pyqtSlot()    ##下一曲目
   def on_btnNext_clicked(self):  
      pass
   

   @pyqtSlot()    ##静音控制
   def on_btnSound_clicked(self):  
      pass
   
      
   @pyqtSlot(int)    ##调节音量
   def on_sliderVolumn_valueChanged(self,value):  
      pass
   

   @pyqtSlot(int)    ##文件进度调控
   def on_sliderPosition_valueChanged(self,value):  
      pass
      

##  =============自定义槽函数===============================        
   def do_stateChanged(self,state):    ##播放器状态变化
      pass
   

   def do_positionChanged(self,position):    ##当前文件播放位置变化，更新进度显示
      pass


   def do_durationChanged(self,duration):    ##文件时长变化
      pass


   def do_currentChanged(self,position):     ##playlist当前曲目变化
      pass
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
    app = QApplication(sys.argv)    #创建GUI应用程序
    form=QmyMainWindow()            #创建窗体
    form.show()
    sys.exit(app.exec_())
