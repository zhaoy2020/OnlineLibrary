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
##      self.setWindowState(Qt.WindowMaximized) #窗口最大化显示
      self.ui.mainToolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

##  ==============事件处理函数===================
   def closeEvent(self,event):
      self.ui.mdiArea.closeAllSubWindows()   #关闭所有子窗口
      event.accept()


##  ==============自定义功能函数============
        
   def __enableEditActions(self,enabled):
      self.ui.actEdit_Cut.setEnabled(enabled)
      self.ui.actEdit_Copy.setEnabled(enabled)
      self.ui.actEdit_Paste.setEnabled(enabled)
      self.ui.actEdit_Font.setEnabled(enabled)
        
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##“新建文档”
   def on_actDoc_New_triggered(self):
      formDoc = QmyFormDoc(self)
      self.ui.mdiArea.addSubWindow(formDoc)   #文档窗口添加到MDI
      formDoc.show()    #在单独的窗口中显示
      self.__enableEditActions(True)

   @pyqtSlot()    ##“打开文档”
   def on_actDoc_Open_triggered(self): 
      needNew=False   # 是否需要新建子窗口
      if len(self.ui.mdiArea.subWindowList())>0: #如果有打开的MDI窗口，获取活动窗口
         formDoc=self.ui.mdiArea.activeSubWindow().widget()
         needNew=formDoc.isFileOpened()   #文件已经打开，需要新建窗口
      else:
         needNew=True

      curPath=os.getcwd()  #获取当前路径
      filename,flt=QFileDialog.getOpenFileName(self,"打开一个文件",curPath,
                  "文本文件(*.cpp *.h *.py);;所有文件(*.*)")
      if (filename==""):
         return

      if(needNew):
         formDoc = QmyFormDoc(self)    #必须指定父窗口
         self.ui.mdiArea.addSubWindow(formDoc)  #添加到MDI区域
        
      formDoc.loadFromFile(filename)
      formDoc.show()
      self.__enableEditActions(True)
        

   @pyqtSlot()    ##“关闭全部”
   def on_actDoc_CloseALL_triggered(self):
      self.ui.mdiArea.closeAllSubWindows()

   @pyqtSlot(bool)      ##“MDI模式”
   def on_actMDI_Mode_triggered(self,checked):
      if checked:     #Tab多页显示模式
         self.ui.mdiArea.setViewMode(QMdiArea.TabbedView)    #Tab多页显示模式
         self.ui.mdiArea.setTabsClosable(True)       #页面可关闭
         self.ui.actMDI_Cascade.setEnabled(False)
         self.ui.actMDI_Tile.setEnabled(False)
      else:       #子窗口模式
         self.ui.mdiArea.setViewMode(QMdiArea.SubWindowView)   #子窗口模式
         self.ui.actMDI_Cascade.setEnabled(True)
         self.ui.actMDI_Tile.setEnabled(True)

   @pyqtSlot()    ##“级联展开”
   def on_actMDI_Cascade_triggered(self):
      self.ui.mdiArea.cascadeSubWindows()
        
   @pyqtSlot()    ##“平铺展开”
   def on_actMDI_Tile_triggered(self):  
      self.ui.mdiArea.tileSubWindows()


   @pyqtSlot()    ##Cut 操作
   def on_actEdit_Cut_triggered(self):
      formDoc=self.ui.mdiArea.activeSubWindow().widget()
      formDoc.textCut()
    
   @pyqtSlot()    ##Copy 操作
   def on_actEdit_Copy_triggered(self):
      formDoc=self.ui.mdiArea.activeSubWindow().widget()
      formDoc.textCopy()

   @pyqtSlot()    ##Paste 操作
   def on_actEdit_Paste_triggered(self):
      formDoc=self.ui.mdiArea.activeSubWindow().widget()
      formDoc.textPaste()

   @pyqtSlot()    ##“字体设置”
   def on_actEdit_Font_triggered(self):
      formDoc=self.ui.mdiArea.activeSubWindow().widget()
      formDoc.textSetFont()

##    @pyqtSlot(type)   ##子窗口切换
   def on_mdiArea_subWindowActivated(self,arg1):
      cnt=len(self.ui.mdiArea.subWindowList())
      if (cnt==0):
         self.__enableEditActions(False)
         self.ui.statusBar.clearMessage()
      else:
         formDoc=self.ui.mdiArea.activeSubWindow().widget()
         self.ui.statusBar.showMessage(formDoc.currentFileName()) #显示子窗口的文件名
        
        
##  =============自定义槽函数===============================

        
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyMainWindow()
   form.show()
   sys.exit(app.exec_())
