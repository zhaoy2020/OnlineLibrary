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

##      self.setWindowTitle("Demo10_1，音乐播放器")
      self.player = QMediaPlayer(self)
      self.playlist = QMediaPlaylist(self)
      self.player.setPlaylist(self.playlist)
      self.playlist.setPlaybackMode(QMediaPlaylist.Loop)    #循环模式

      self.__duration=""      #文件总时间长度
      self.__curPos=""        #当前播放到位置

      self.player.stateChanged.connect(self.do_stateChanged)
      
      self.player.positionChanged.connect(self.do_positionChanged)
      
      self.player.durationChanged.connect(self.do_durationChanged)
      
      self.playlist.currentIndexChanged.connect(self.do_currentChanged)
      
##  ==============自定义功能函数============



##  ===============event 处理函数==========
   def closeEvent(self,event):  ##窗体关闭时
##  窗口关闭时不能自动停止播放，需手动停止
      if (self.player.state()==QMediaPlayer.PlayingState):
         self.player.stop()

         
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
#  播放列表管理
   @pyqtSlot()    ##添加文件
   def on_btnAdd_clicked(self):  
   ##      curPath=os.getcwd()     #获取系统当前目录
   ##      curPath=QDir.homePath()
      curPath=QDir.currentPath()
      dlgTitle="选择音频文件" 
      filt="音频文件(*.mp3 *.wav *.wma);;所有文件(*.*)" 
      fileList,flt=QFileDialog.getOpenFileNames(self,dlgTitle,curPath,filt)
      count=len(fileList)
      if count<1:
         return

      filename=fileList[0]
      fileInfo=QFileInfo(filename)  #文件信息
      QDir.setCurrent(fileInfo.absolutePath())  #重设当前路径

      for i in range(count):
         filename=fileList[i] 
         fileInfo.setFile(filename)
         song=QMediaContent(QUrl.fromLocalFile(filename))
         self.playlist.addMedia(song)  #添加播放媒体
   ##         basename=os.path.basename(filename)    #文件名和后缀
         basename=fileInfo.baseName()
         self.ui.listWidget.addItem(basename)    #添加到界面文件列表

      if (self.player.state()!=QMediaPlayer.PlayingState):
         self.playlist.setCurrentIndex(0)
         self.player.play()
      
   @pyqtSlot()    ##移除一个文件
   def on_btnRemove_clicked(self): 
      pos=self.ui.listWidget.currentRow()
      item=self.ui.listWidget.takeItem(pos)     #python会自动删除

      if (self.playlist.currentIndex()==pos):   #是当前播放的曲目
         nextPos=0
         if pos>=1:
            nextPos=pos-1

         self.playlist.removeMedia(pos)   #从播放列表里移除
         if self.ui.listWidget.count()>0: #剩余个数
            self.playlist.setCurrentIndex(nextPos)
            self.do_currentChanged(nextPos)  #当前曲目变化
         else:
            self.player.stop()
            self.ui.LabCurMedia.setText("无曲目")
      else:
         self.playlist.removeMedia(pos)
      

   @pyqtSlot()    ##清空播放列表
   def on_btnClear_clicked(self):  
      self.playlist.clear()   #清空播放列表
      self.ui.listWidget.clear()  
      self.player.stop()      #停止播放
      
   ##   @pyqtSlot()    ##双击时切换播放文件
   def on_listWidget_doubleClicked(self,index):  
      rowNo=index.row()  #行号
      self.playlist.setCurrentIndex(rowNo)
      self.player.play()
      

#  播放控制
   @pyqtSlot()    ##播放
   def on_btnPlay_clicked(self):  
      if (self.playlist.currentIndex()<0):
         self.playlist.setCurrentIndex(0)
      self.player.play()

   @pyqtSlot()    ##暂停
   def on_btnPause_clicked(self):  
      self.player.pause()

   @pyqtSlot()    ##停止
   def on_btnStop_clicked(self):  
      self.player.stop()

   @pyqtSlot()    ##上一曲目
   def on_btnPrevious_clicked(self):  
      self.playlist.previous()

   @pyqtSlot()    ##下一曲目
   def on_btnNext_clicked(self):  
      self.playlist.next()

   @pyqtSlot()    ##静音控制
   def on_btnSound_clicked(self):  
      mute=self.player.isMuted()
      self.player.setMuted(not mute)
      if mute:
         self.ui.btnSound.setIcon(QIcon(":/icons/images/volumn.bmp"))
      else:
         self.ui.btnSound.setIcon(QIcon(":/icons/images/mute.bmp"))
      
   @pyqtSlot(int)    ##调节音量
   def on_sliderVolumn_valueChanged(self,value):  
      self.player.setVolume(value)

   @pyqtSlot(int)    ##文件进度调控
   def on_sliderPosition_valueChanged(self,value):  
      self.player.setPosition(value)
      

##  =============自定义槽函数===============================        
   def do_stateChanged(self,state):    ##播放器状态变化
      self.ui.btnPlay.setEnabled(state!=QMediaPlayer.PlayingState)
      self.ui.btnPause.setEnabled(state==QMediaPlayer.PlayingState)
      self.ui.btnStop.setEnabled(state==QMediaPlayer.PlayingState)

   def do_positionChanged(self,position): ##当前文件播放位置变化，更新进度显示
      if (self.ui.sliderPosition.isSliderDown()): #在拖动滑块调整进度
         return
      self.ui.sliderPosition.setSliderPosition(position)
      secs=position/1000   #秒
      mins=secs/60         #分钟
      secs=secs % 60       #余数秒
      self.__curPos="%d:%d"%(mins,secs)
      self.ui.LabRatio.setText(self.__curPos+"/"+self.__duration)

   def do_durationChanged(self,duration):    ##文件时长变化
      self.ui.sliderPosition.setMaximum(duration)

      secs=duration/1000   #秒
      mins=secs/60         #分钟
      secs=secs % 60       #余数秒
      self.__duration="%d:%d"%(mins,secs)
      self.ui.LabRatio.setText(self.__curPos+"/"+self.__duration)

   def do_currentChanged(self,position):  ##playlist当前曲目变化
      self.ui.listWidget.setCurrentRow(position)
      item=self.ui.listWidget.currentItem()  #QListWidgetItem
      if (item != None):
         self.ui.LabCurMedia.setText(item.text())
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
    app = QApplication(sys.argv)    #创建GUI应用程序
    form=QmyMainWindow()            #创建窗体
    form.show()
    sys.exit(app.exec_())
