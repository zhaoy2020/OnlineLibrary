import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,QFileDialog,
                     QAbstractItemView, QMessageBox,QDataWidgetMapper)

from PyQt5.QtCore import  pyqtSlot, Qt,QItemSelectionModel,QModelIndex

from PyQt5.QtSql import QSqlDatabase, QSqlQueryModel, QSqlRecord, QSqlQuery  

from PyQt5.QtGui import QPixmap

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      self.setCentralWidget(self.ui.splitter)
      pass


##  ==============自定义功能函数============
   def __getFieldNames(self):  ##获取所有字段名称
      pass

   
   def __openTable(self):     ##查询数据
      pass


   def __refreshTableView(self): ##刷新tableView显示
      pass

        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##“打开数据库”按钮
   def on_actOpenDB_triggered(self):
      pass


   @pyqtSlot()    ##首记录
   def on_actRecFirst_triggered(self): 
      pass
   

   @pyqtSlot()    ##前一记录
   def on_actRecPrevious_triggered(self): 
      pass
   

   @pyqtSlot()    ##后一条记录
   def on_actRecNext_triggered(self): 
      pass
   

   @pyqtSlot()    ##最后一条记录
   def on_actRecLast_triggered(self): 
      pass

      
##  =============自定义槽函数===============================        
   def do_currentRowChanged(self, current, previous):    ##记录移动时触发
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
