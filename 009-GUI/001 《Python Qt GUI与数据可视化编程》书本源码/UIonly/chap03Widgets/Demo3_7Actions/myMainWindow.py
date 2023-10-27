import sys

from PyQt5.QtWidgets import (QApplication, QMainWindow,QActionGroup,
                 QLabel, QProgressBar, QSpinBox, QFontComboBox)

from PyQt5.QtCore import  Qt, pyqtSlot

from PyQt5.QtGui import  QTextCharFormat, QFont

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      self.setCentralWidget(self.ui.textEdit)
      pass


##  ============自定义功能函数================================        
   def __buildUI(self):    ##窗体上动态添加组件
      pass

        
##  ===========由connectSlotsByName() 自动连接的槽函数=====================      
   @pyqtSlot(bool)   ##设置粗体 
   def on_actFont_Bold_triggered(self, checked):
      pass

   @pyqtSlot(bool)   ##设置斜体 
   def on_actFont_Italic_triggered(self,checked):    
      pass
        
   @pyqtSlot(bool)   ##设置下划线 
   def on_actFont_UnderLine_triggered(self,checked):  
      pass

   def on_textEdit_copyAvailable(self, avi):    ##文本框内容可copy
      pass
                                         
   def on_textEdit_selectionChanged(self):      ##文本选择内容发生变化
      pass

   def on_textEdit_customContextMenuRequested(self,pos):  ##标准右键菜单
      pass

   @pyqtSlot(bool)   ##设置工具栏按钮样式 
   def on_actSys_ToggleText_triggered(self,checked):  
      pass

   def on_actFile_New_triggered(self):     ##新建文件，不实现具体功能
      pass

   def on_actFile_Open_triggered(self):    ##打开文件，不实现具体功能
      pass
        
   def on_actFile_Save_triggered(self):    ##保存文件，不实现具体功能
      pass
        
##  =============自定义槽函数===============================        
   @pyqtSlot(int)    ##设置字体大小,关联 __spinFontSize
   def do_fontSize_Changed(self, fontSize):      
      pass

   @pyqtSlot(str)    ##选择字体名称，关联__comboFontName
   def do_fontName_Changed(self, fontName):  
      pass
   
##  ===========窗体测试程序=================================        
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
