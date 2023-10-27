import sys

from PyQt5.QtWidgets import (QApplication, QWidget,  QAbstractItemView,
                QTreeWidgetItem, QListWidget,QTreeWidget,QTableWidget)
from PyQt5.QtCore import  pyqtSlot, Qt, QEvent

from ui_Widget import Ui_Widget
class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)    
      self.ui=Ui_Widget()         
      self.ui.setupUi(self)       

      self.ui.listSource.installEventFilter(self)     #安装事件过滤器
      self.ui.listWidget.installEventFilter(self)
      self.ui.treeWidget.installEventFilter(self)
      self.ui.tableWidget.installEventFilter(self)

      self.ui.listSource.setAcceptDrops(True)
      self.ui.listSource.setDragDropMode(QAbstractItemView.DragDrop)
      self.ui.listSource.setDragEnabled(True)
      self.ui.listSource.setDefaultDropAction(Qt.CopyAction) 

      self.ui.listWidget.setAcceptDrops(True)
      self.ui.listWidget.setDragDropMode(QAbstractItemView.DragDrop)
      self.ui.listWidget.setDragEnabled(True)
      self.ui.listWidget.setDefaultDropAction(Qt.MoveAction)  

      self.ui.treeWidget.setAcceptDrops(True)
      self.ui.treeWidget.setDragDropMode(QAbstractItemView.DragDrop)
      self.ui.treeWidget.setDragEnabled(True)
      self.ui.treeWidget.setDefaultDropAction(Qt.MoveAction)  

      self.ui.tableWidget.setAcceptDrops(True)
      self.ui.tableWidget.setDragDropMode(QAbstractItemView.DragDrop)
      self.ui.tableWidget.setDragEnabled(True)
      self.ui.tableWidget.setDefaultDropAction(Qt.MoveAction) 

      self.__itemView=None    #用于设置属性的当前组件
      self.on_radio_Source_clicked()   #调用一次槽函数，初始化界面

##  =================自定义功能函数=================================
   def __getDropActionIndex(self,actionType):
      if actionType==Qt.CopyAction:
         return 0
      elif actionType==Qt.MoveAction:
         return 1
      elif actionType==Qt.LinkAction:
         return 2
      elif actionType==Qt.IgnoreAction:
         return 3
      else:
         return 0

   def __getDropActionType(self,index):
      if index==0:
         return Qt.CopyAction
      elif index==1:
         return Qt.MoveAction
      elif index==2:
         return Qt.LinkAction
      elif index==3:
         return Qt.IgnoreAction
      else:
         return Qt.CopyAction


   def __refreshToUI(self): ##属性显示到界面
      self.ui.chkBox_AcceptDrops.setChecked(self.__itemView.acceptDrops())
      self.ui.chkBox_DragEnabled.setChecked(self.__itemView.dragEnabled())

      self.ui.combo_Mode.setCurrentIndex(self.__itemView.dragDropMode())
      index=self.__getDropActionIndex(self.__itemView.defaultDropAction())
      self.ui.combo_DefaultAction.setCurrentIndex(index)
        

##  ==================event处理函数=================================
   def eventFilter(self,watched,event):
      if (event.type()==QEvent.KeyPress) and (event.key()==Qt.Key_Delete):
         if (watched==self.ui.listSource):
            self.ui.listSource.takeItem(self.ui.listSource.currentRow())
         elif (watched==self.ui.listWidget):
            self.ui.listWidget.takeItem(self.ui.listWidget.currentRow())
         elif (watched==self.ui.treeWidget):
            curItem=self.ui.treeWidget.currentItem()
            if (curItem.parent()!=None):
               parItem=curItem.parent()
               parItem.removeChild(curItem)
            else:
               index=self.ui.treeWidget.indexOfTopLevelItem(curItem)
               self.ui.treeWidget.takeTopLevelItem(index)
         elif (watched==self.ui.tableWidget):
            self.ui.tableWidget.takeItem(self.ui.tableWidget.currentRow(),
                                       self.ui.tableWidget.currentColumn())
      return super().eventFilter(watched,event)
        
##  ==========由connectSlotsByName()自动连接的槽函数===============        
   @pyqtSlot()
   def on_radio_Source_clicked(self):  ##listSource
      self.__itemView=self.ui.listSource
      self.__refreshToUI()

   @pyqtSlot()
   def on_radio_List_clicked(self):    ##listWidget
      self.__itemView=self.ui.listWidget
      self.__refreshToUI()
    
   @pyqtSlot()
   def on_radio_Tree_clicked(self):    ##treeWidget
      self.__itemView=self.ui.treeWidget
      self.__refreshToUI()

   @pyqtSlot()
   def on_radio_Table_clicked(self):   ##tableWidget
      self.__itemView=self.ui.tableWidget
      self.__refreshToUI()

##============用于属性设置的4个组件的槽函数===================
   @pyqtSlot(bool)
   def on_chkBox_AcceptDrops_clicked(self,checked):
      self.__itemView.setAcceptDrops(checked)

   @pyqtSlot(bool)
   def on_chkBox_DragEnabled_clicked(self,checked):
      self.__itemView.setDragEnabled(checked)

   @pyqtSlot(int)
   def on_combo_Mode_currentIndexChanged(self,index):
      mode= (QAbstractItemView.DragDropMode)(index)
      self.__itemView.setDragDropMode(mode)
        
   @pyqtSlot(int)
   def on_combo_DefaultAction_currentIndexChanged(self,index):
      actionType=self.__getDropActionType(index)
      self.__itemView.setDefaultDropAction(actionType)

        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":     
   app = QApplication(sys.argv)
   form=QmyWidget()            
   form.show()
   sys.exit(app.exec_())
