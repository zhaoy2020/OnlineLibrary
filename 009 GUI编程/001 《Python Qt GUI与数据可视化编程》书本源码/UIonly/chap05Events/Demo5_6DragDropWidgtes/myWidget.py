import sys

from PyQt5.QtWidgets import (QApplication, QWidget,  QAbstractItemView,
                QTreeWidgetItem, QListWidget,QTreeWidget,QTableWidget)
from PyQt5.QtCore import  pyqtSlot, Qt, QEvent

##from PyQt5.QtGui import  QPixmap

from ui_Widget import Ui_Widget


class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()         #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      pass


##  =================自定义功能函数=================================
   def __getDropActionIndex(self,actionType):
      pass
   

   def __getDropActionType(self,index):
      pass


   def __refreshToUI(self): ##属性显示到界面
      pass
        

##  ==================event处理函数=================================
   def eventFilter(self,watched,event):
      pass
      return super().eventFilter(watched,event)
        
##  ==========由connectSlotsByName()自动连接的槽函数===============        
   @pyqtSlot()
   def on_radio_Source_clicked(self):  ##listSource
      pass

   @pyqtSlot()
   def on_radio_List_clicked(self):  ##listWidget
      pass
    
   @pyqtSlot()
   def on_radio_Tree_clicked(self):    ##treeWidget
      pass

   @pyqtSlot()
   def on_radio_Table_clicked(self):   ##tableWidget
      pass

##============用于属性设置的4个组件的槽函数===================
   @pyqtSlot(bool)
   def on_chkBox_AcceptDrops_clicked(self,checked):
      pass

   @pyqtSlot(bool)
   def on_chkBox_DragEnabled_clicked(self,checked):
      pass

   @pyqtSlot(int)
   def on_combo_Mode_currentIndexChanged(self,index):
      pass
        
   @pyqtSlot(int)
   def on_combo_DefaultAction_currentIndexChanged(self,index):
      pass

        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":     
   app = QApplication(sys.argv)
   form=QmyWidget()            
   form.show()
   sys.exit(app.exec_())
