import sys,os  #codecs

from PyQt5.QtWidgets import  (QApplication, QWidget,QFileDialog,
                         QToolBar, QVBoxLayout,QFontDialog)

from PyQt5.QtCore import  pyqtSlot, pyqtSignal,Qt

from PyQt5.QtGui import QPalette,  QFont

from ui_QWFormDoc import Ui_QWFormDoc

class QmyFormDoc(QWidget):
   docFileChanged=pyqtSignal(str)  ##自定义信号，打开文件时发射

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_QWFormDoc()     #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      pass

        
   def __del__(self):   ##析构函数
      pass
   

   def __buildUI(self):    ##使用UI可视化设计的Actions创建工具栏
      pass
        

##  ==============自定义功能函数============
        
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##打开文件
   def on_actOpen_triggered(self): 
      pass


   @pyqtSlot()    ##设置字体
   def on_actFont_triggered(self):
      pass

        
##  =============自定义槽函数===============================        
        
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyFormDoc()
   form.show()
   sys.exit(app.exec_())
