import sys

from PyQt5.QtWidgets import (QApplication, QMainWindow,QFileDialog,
                     QAbstractItemView, QMessageBox, QDialog)

from PyQt5.QtCore import  pyqtSlot, Qt,QItemSelectionModel,QModelIndex

from PyQt5.QtSql import QSqlDatabase, QSqlQueryModel, QSqlRecord, QSqlQuery  

##from PyQt5.QtGui import QPixmap

from ui_MainWindow import Ui_MainWindow

from myDialogData import QmyDialogData

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      self.setCentralWidget(self.ui.tableView)
      pass


##  ==============自定义功能函数============
   def __getFieldNames(self):  ##获取所有字段名称
      pass

   
   def __openTable(self):  #查询数据
      pass


   def __updateRecord(self,recNo): ##更新一条记录
      pass
      
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##打开数据库
   def on_actOpenDB_triggered(self):
      pass


   @pyqtSlot()    ##插入记录
   def on_actRecInsert_triggered(self):
      pass


   @pyqtSlot()    ##删除记录
   def on_actRecDelete_triggered(self):
      pass

      
   @pyqtSlot()    ##编辑记录
   def on_actRecEdit_triggered(self):
      pass


   ##   @pyqtSlot()  ##双击编辑记录
   def on_tableView_doubleClicked(self,index):
      pass
   

   @pyqtSlot()    ##遍历记录，涨工资
   def on_actScan_triggered(self):
      pass


   @pyqtSlot()    ##SQL语句测试
   def on_actTestSQL_triggered(self):
      pass

      
##  =============自定义槽函数===============================        
   def do_currentRowChanged(self, current, previous):    ##行切换时触发
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
