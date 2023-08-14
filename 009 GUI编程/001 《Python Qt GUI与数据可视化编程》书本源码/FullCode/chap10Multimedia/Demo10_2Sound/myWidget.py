import sys

from PyQt5.QtWidgets import  QApplication, QWidget

from PyQt5.QtCore import  pyqtSlot, QUrl

##from PyQt5.QtWidgets import  

##from PyQt5.QtGui import

from PyQt5.QtMultimedia import QSoundEffect,QSound

from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面


##  =================自定义功能函数=================================

        
##  ==========由connectSlotsByName() 自动连接的槽函数===============        
   @pyqtSlot()    ##播放文件
   def on_btnEffect_File_clicked(self):   
      url=QUrl.fromLocalFile("Ak47.wav")
      player=QSoundEffect(self)
      player.setLoopCount(2)  #播放循环次数
      player.setSource(url)   #设置源文件
      player.play()

   @pyqtSlot()    ##播放资源文件的内容
   def on_btnEffect_Resource_clicked(self):   
      url=QUrl.fromLocalFile(":/Wave/sound/blast.wav")
      player=QSoundEffect(self)  
      player.setLoopCount(2) 
      player.setSource(url)
      player.play()  #无法播放资源文件

   @pyqtSlot()    ##播放文件
   def on_btnSound_File_clicked(self):  
      QSound.play("fire2.wav")
      QSound.play("machinegun.wav")

   @pyqtSlot()    ##播放资源文件
   def on_btnSound_Resource_clicked(self):  
      QSound.play(":/Wave/sound/machinegun.wav")  #不能播放

        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
    app = QApplication(sys.argv)    #创建GUI应用程序
    
    form=QmyWidget()                #创建窗体
    form.show()

    sys.exit(app.exec_())
