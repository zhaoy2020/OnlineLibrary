import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,QFileDialog,
                         QAbstractItemView,  QMessageBox,QDataWidgetMapper)

from PyQt5.QtCore import  (pyqtSlot, Qt,QItemSelectionModel,
                         QModelIndex,QFile,QIODevice)

from PyQt5.QtSql import QSqlDatabase, QSqlTableModel, QSqlRecord 

from PyQt5.QtGui import QPixmap

from ui_MainWindow import Ui_MainWindow

from myDelegates import QmyComboBoxDelegate  

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

   
   def __openTable(self): ##打开数据表
      pass

        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##选择数据库，打开数据表
   def on_actOpenDB_triggered(self):
      pass


   @pyqtSlot()    ##保存修改
   def on_actSubmit_triggered(self):   
      pass


   @pyqtSlot()    ##取消修改
   def on_actRevert_triggered(self): 
      pass
   

   @pyqtSlot()    ##添加记录
   def on_actRecAppend_triggered(self):
      pass


   @pyqtSlot()    ##插入记录
   def on_actRecInsert_triggered(self):
      pass


   @pyqtSlot()    ##删除记录
   def on_actRecDelete_triggered(self):
      pass


   @pyqtSlot()    ##清除照片
   def on_actPhotoClear_triggered(self):
      pass


   @pyqtSlot()    ##设置照片
   def on_actPhoto_triggered(self):
      pass


   @pyqtSlot()    ##涨工资，遍历数据表所有记录
   def on_actScan_triggered(self):
      pass


   @pyqtSlot(int)    ##排序字段变化
   def on_comboFields_currentIndexChanged(self,index):
      pass


   @pyqtSlot()    ##升序
   def on_radioBtnAscend_clicked(self):
      pass

      
   @pyqtSlot()    ##降序
   def on_radioBtnDescend_clicked(self):
      pass


   @pyqtSlot()    ##过滤，男
   def on_radioBtnMan_clicked(self):
      pass
      
   @pyqtSlot()    ##数据过滤，女
   def on_radioBtnWoman_clicked(self):
      pass


   @pyqtSlot()    ##取消数据过滤
   def on_radioBtnBoth_clicked(self):
      pass

      
##  =============自定义槽函数===============================        
   def do_currentChanged(self,current,previous):   ##更新actPost和actCancel 的状态
      pass
   

   def do_currentRowChanged(self, current, previous): #行切换时的状态控制
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
