import sys,os

from PyQt5.QtWidgets import  (QApplication, QMainWindow,
                   QLabel, QAbstractItemView, QFileDialog)

from PyQt5.QtCore import  Qt, pyqtSlot,QItemSelectionModel, QModelIndex

from PyQt5.QtGui import QStandardItemModel, QStandardItem

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      self.setCentralWidget(self.ui.splitter)
      pass
      

##  ==============自定义功能函数============
   def __buildStatusBar(self):   ##构建状态栏
      pass
        

   def __iniModelFromStringList(self,allLines):  ##从字符串列表构建模型
      pass
  

   def  __setCellAlignment(self, align=Qt.AlignHCenter):
      pass
        
    
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##“打开文件”
   def on_actOpen_triggered(self):  
      pass


   @pyqtSlot()    ##保存文件
   def on_actSave_triggered(self): 
      pass
        

   @pyqtSlot()    ##在最后添加一行
   def on_actAppend_triggered(self):   
      pass


   @pyqtSlot()    ##插入一行
   def on_actInsert_triggered(self):   
      pass


   @pyqtSlot()    ##删除当前行
   def on_actDelete_triggered(self): 
      pass
   

   @pyqtSlot()    ##左对齐
   def on_actAlignLeft_triggered(self):  
      pass


   @pyqtSlot()    ##中间对齐
   def on_actAlignCenter_triggered(self):  
      pass
   
        
   @pyqtSlot()    ##右对齐
   def on_actAlignRight_triggered(self):   
      pass
   

   @pyqtSlot(bool)   ##字体Bold
   def on_actFontBold_triggered(self,checked):
      pass


   @pyqtSlot()    ##模型数据显示到plainTextEdit里
   def on_actModelData_triggered(self):  
      pass
        
    
##  =============自定义槽函数===============================        
   def do_curChanged(self,current,previous):
      pass

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
    app = QApplication(sys.argv)    #创建GUI应用程序
    form=QmyMainWindow()            #创建窗体
    form.show()
    sys.exit(app.exec_())
