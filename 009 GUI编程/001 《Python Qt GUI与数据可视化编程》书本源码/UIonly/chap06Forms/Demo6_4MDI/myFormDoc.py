import sys,codecs,os

from PyQt5.QtWidgets import  QApplication, QWidget,QFontDialog

##from PyQt5.QtWidgets import  

from PyQt5.QtCore import  Qt

##from PyQt5.QtGui import 

from ui_QWFormDoc import Ui_QWFormDoc

class QmyFormDoc(QWidget):

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_QWFormDoc()      #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      pass


   def __del__(self):   ##析构函数
      pass

##  ==============自定义功能函数============
   def loadFromFile(self, aFileName):  ##打开文件
      pass


   def currentFileName(self): ##返回当前文件名
      pass

   def isFileOpened(self):    ##文件是否打开
      pass

   def textCut(self):   ## cut 操作
      pass

   def textCopy(self):  ## copy 操作
      pass
        
   def textPaste(self): ## paste 操作
      pass

   def textSetFont(self):     ##设置字体
      pass

        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
        
    
##  =============自定义槽函数===============================        
        
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyFormDoc()
   form.show()
   sys.exit(app.exec_())
