import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,
                       QListWidgetItem, QMenu, QToolButton)

from PyQt5.QtGui import  QIcon, QCursor

from PyQt5.QtCore import  pyqtSlot, Qt

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面

      self.setCentralWidget(self.ui.splitter)
      pass

        

##  ================自定义功能函数============================
   def __setActionsForButton(self):  ##为界面上的ToolButton按钮设置Actions
      pass

   def __createSelectionPopMenu(self):  ##创建ToolButton的下拉菜单
      pass

        
##  =========connectSlotsByName() 自动连接的槽函数===========        
    
   @pyqtSlot()  ##初始化列表
   def on_actList_Ini_triggered(self):    
      pass


   @pyqtSlot()    ##插入一项
   def on_actList_Insert_triggered(self):  
      pass

        
   @pyqtSlot()  ##添加一项
   def on_actList_Append_triggered(self):    
      pass
            

   @pyqtSlot()   ##删除当前项
   def on_actList_Delete_triggered(self):  
      pass

        
   @pyqtSlot()   ##清空列表
   def on_actList_Clear_triggered(self):   
      pass

   @pyqtSlot()
   def on_actSel_ALL_triggered(self):     ##全选
      pass

            
   @pyqtSlot()
   def on_actSel_None_triggered(self):     ##全不选
      pass

    
   @pyqtSlot()
   def on_actSel_Invs_triggered(self):     ##反选
      pass

   @pyqtSlot(bool)
   def on_chkBoxList_Editable_clicked(self,checked):   ##改变可编辑状态
      pass


   def on_listWidget_currentItemChanged(self,current,previous):    ##当前项切换变化
      pass


   def on_listWidget_customContextMenuRequested(self,pos):  ##右键快捷菜单
      pass

        
##  ================自定义槽函数===============================        

   
##  ================窗体测试程序 ============================        
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
