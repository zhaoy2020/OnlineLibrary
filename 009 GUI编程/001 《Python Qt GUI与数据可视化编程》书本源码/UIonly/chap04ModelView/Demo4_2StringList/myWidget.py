import sys

from PyQt5.QtWidgets import  QApplication, QWidget,QAbstractItemView

from PyQt5.QtCore import  pyqtSlot, QStringListModel, Qt, QModelIndex

##from PyQt5.QtWidgets import  QAbstractItemView

##from PyQt5.QtGui import  

from ui_Widget import Ui_Widget


class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      pass


##  =================自定义功能函数=================================

        
##  ==========由connectSlotsByName() 自动连接的槽函数===============        
   @pyqtSlot()    ##重设模型数据内容
   def on_btnList_Reset_clicked(self):
      pass

   @pyqtSlot()    ##添加项
   def on_btnList_Append_clicked(self):   
      pass


   @pyqtSlot()    ##插入项
   def on_btnList_Insert_clicked(self):   
      pass


   @pyqtSlot()    ##删除当前项
   def on_btnList_Delete_clicked(self):   
      pass
   
        
   @pyqtSlot()    ##清空列表
   def on_btnList_Clear_clicked(self): 
      pass
   

   @pyqtSlot()    ##清空文本
   def on_btnText_Clear_clicked(self):
      pass


   @pyqtSlot()    ##显示数据模型的内容
   def on_btnText_Display_clicked(self):  
      pass


   def on_listView_clicked(self,index):
      pass
        
    
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":      
   app = QApplication(sys.argv) 
   form=QmyWidget()           
   form.show()
   sys.exit(app.exec_())
