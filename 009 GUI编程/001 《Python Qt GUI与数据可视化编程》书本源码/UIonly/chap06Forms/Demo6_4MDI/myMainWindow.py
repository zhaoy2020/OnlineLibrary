import sys,os

from PyQt5.QtWidgets import  QApplication, QMainWindow,QFileDialog,QMdiArea

from PyQt5.QtCore import  pyqtSlot, Qt


from ui_MainWindow import Ui_MainWindow

from myFormDoc import QmyFormDoc

class QmyMainWindow(QMainWindow):

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      self.setCentralWidget(self.ui.mdiArea)    #填充满工作区
      pass
   

##  ==============事件处理函数===================
   def closeEvent(self,event):   ##关闭窗口事件
      pass


##  ==============自定义功能函数============
   def __enableEditActions(self,enabled):
      pass
        
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##“新建文档”
   def on_actDoc_New_triggered(self):
      pass


   @pyqtSlot()    ##“打开文档”
   def on_actDoc_Open_triggered(self): 
      pass
        

   @pyqtSlot()    ##“关闭全部”
   def on_actDoc_CloseALL_triggered(self):
      pass

   @pyqtSlot(bool)   ##“MDI模式”
   def on_actMDI_Mode_triggered(self,checked):
      pass


   @pyqtSlot()    ##“级联展开”
   def on_actMDI_Cascade_triggered(self):
      pass
        
   @pyqtSlot()    ##“平铺展开”
   def on_actMDI_Tile_triggered(self):  
      pass


   @pyqtSlot()    ##Cut 操作
   def on_actEdit_Cut_triggered(self):
      pass
    
   @pyqtSlot()    ##Copy 操作
   def on_actEdit_Copy_triggered(self):
      pass


   @pyqtSlot()    ##Paste 操作
   def on_actEdit_Paste_triggered(self):
      pass

   @pyqtSlot()    ##“字体设置”
   def on_actEdit_Font_triggered(self):
      pass

##    @pyqtSlot(type)   ##子窗口切换
   def on_mdiArea_subWindowActivated(self,arg1):
      pass

        
        
##  =============自定义槽函数===============================

        
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyMainWindow()
   form.show()
   sys.exit(app.exec_())
