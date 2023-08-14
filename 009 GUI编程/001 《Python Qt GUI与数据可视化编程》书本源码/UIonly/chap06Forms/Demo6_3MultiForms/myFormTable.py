import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,
                     QLabel,QAbstractItemView,QDialog)

from PyQt5.QtCore import  pyqtSlot, Qt, QItemSelectionModel

from PyQt5.QtGui import QStandardItemModel

from ui_QWFormTable import Ui_QWFormTable

from myDialogSize import QmyDialogSize

from myDialogHeaders import QmyDialogHeaders

class QmyFormTable(QMainWindow):

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_QWFormTable()    #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      pass

        
   def __del__(self): ##析构函数
      pass

##  ==============自定义功能函数============

        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##设置表格大小
   def on_actSetSize_triggered(self): 
      pass


   @pyqtSlot()    ##设置表头标题
   def on_actSetHeader_triggered(self):  
      pass

        
##  =============自定义槽函数===============================        
        
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyFormTable()
   form.show()
   sys.exit(app.exec_())
