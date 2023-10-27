import sys

from PyQt5.QtWidgets import  QApplication, QDialog,QFileDialog

##from PyQt5.QtWidgets import  

from PyQt5.QtCore import  pyqtSlot, Qt, QFile,QIODevice,QDate

from PyQt5.QtSql import   QSqlRecord  

##from PyQt5.QtCore import  pyqtSlot,Qt

##from PyQt5.QtWidgets import  

from PyQt5.QtGui import QPixmap

from ui_QWDialogData import Ui_QWDialogData

class QmyDialogData(QDialog): 
   def __init__(self,parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_QWDialogData()  #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.__record=QSqlRecord() #用于存储一条记录的数据
      
##  ==============自定义功能函数============
   def setInsertRecord(self,recData): ##设置插入记录的数据
      pass
      

   def setUpdateRecord(self,recData): ##设置更新记录的数据
      pass


   def getRecordData(self):   ##返回界面编辑的数据
      pass
   
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##“导入照片”按钮
   def on_btnSetPhoto_clicked(self):
      pass


   @pyqtSlot()    ##“清除照片按钮”
   def on_btnClearPhoto_clicked(self):
      pass
      
        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyDialogData()            #创建窗体
   form.show()
   sys.exit(app.exec_())
