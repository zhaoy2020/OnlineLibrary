import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QLabel,QDialog

from PyQt5.QtCore import  pyqtSlot,pyqtSignal, Qt, QItemSelectionModel

from PyQt5.QtGui import QStandardItemModel

from ui_MainWindow import Ui_MainWindow

from myDialogSize import QmyDialogSize

from myDialogHeaders import QmyDialogHeaders

from myDialogLocate import QmyDialogLocate

class QmyMainWindow(QMainWindow):

   cellIndexChanged= pyqtSignal(int,int)

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      self.setCentralWidget(self.ui.tableView)
      pass

        
   def __del__(self):
      pass


##  ==============自定义功能函数============
        
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##设置行数列数对话框
   def on_actTab_SetSize_triggered(self):  
      pass


   @pyqtSlot()    ##设置表头标题
   def on_actTab_SetHeader_triggered(self): 
      pass

        
   @pyqtSlot()    ##"定位单元格"
   def on_actTab_Locate_triggered(self):
      pass

    
##  =============自定义槽函数===============================        
   def do_currentChanged(self,current,previous):
      pass


   def do_setActLocateEnable(self,enable):
      pass
    

   def do_setACellText(self,row, column, text):
      pass


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyMainWindow()
   form.show()
   sys.exit(app.exec_())
