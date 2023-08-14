import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QFileSystemModel

##from PyQt5.QtCore import  pyqtSlot,QDir

from PyQt5.QtCore import  QDir

##from PyQt5.QtWidgets import  QFileSystemModel

##from PyQt5.QtGui import

from ui_MainWindow import Ui_MainWindow


class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      pass


##  ==============自定义功能函数============
   def  __buildModelView(self):   ##构造Model/View 系统
      pass
   
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   def on_treeView_clicked(self,index):  ##treeView单击
      pass
        
        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":       
    app = QApplication(sys.argv)  
    form=QmyMainWindow()          
    form.show()
    sys.exit(app.exec_())
