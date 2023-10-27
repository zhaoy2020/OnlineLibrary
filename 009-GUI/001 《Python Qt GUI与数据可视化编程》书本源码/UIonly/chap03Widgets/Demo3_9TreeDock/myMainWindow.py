import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,
             QTreeWidgetItem, QLabel,QFileDialog,QDockWidget)

from enum import Enum   ##枚举类型

from PyQt5.QtCore import  pyqtSlot, Qt, QDir, QFileInfo

from PyQt5.QtGui import  QIcon,QPixmap

from ui_MainWindow import Ui_MainWindow


class TreeItemType(Enum):    ##节点类型枚举类型
   itTopItem=1001    #顶层节点
   itGroupItem=1002  #组节点 
   itImageItem=1003  #图片文件节点

class TreeColNum(Enum):   ##目录树的列号枚举类型
   colItem=0         #分组/文件名列
   colItemType=1     #节点类型列

class QmyMainWindow(QMainWindow):
   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面

      self.setCentralWidget(self.ui.scrollArea)
      pass



##  =================自定义功能函数================================
   def __iniTree(self):     ##初始化目录树
      pass
   

   def __displayImage(self,item):   ##显示节点item 的图片
      pass
   

   def __changeItemCaption(self,item):    ##递归调用函数，修改节点标题
      pass
        
        
##  ===========connectSlotsByName() 自动连接的槽函数=================        
   @pyqtSlot()    ##添加目录节点
   def on_actTree_AddFolder_triggered(self):  
      pass
        

   @pyqtSlot()    ##添加图片文件节点
   def on_actTree_AddFiles_triggered(self): 
      pass


   @pyqtSlot()    ##删除当前节点
   def on_actTree_DeleteItem_triggered(self): 
      pass

   @pyqtSlot()    ##遍历节点
   def on_actTree_ScanItems_triggered(self): 
      pass


   def on_treeFiles_currentItemChanged(self,current,previous):  ##目录树节点变化
      pass


   @pyqtSlot()  ##适应高度显示图片
   def on_actZoomFitH_triggered(self): 
      pass


   @pyqtSlot()  ##适应宽度显示
   def on_actZoomFitW_triggered(self): 
      pass


   @pyqtSlot()   ##实际大小
   def on_actZoomRealSize_triggered(self): 
      pass

   @pyqtSlot()  ##放大显示
   def on_actZoomIn_triggered(self):  
      pass


   @pyqtSlot() ##缩小显示
   def on_actZoomOut_triggered(self): 
      pass


   @pyqtSlot(bool)  ##设置停靠区浮动性
   def on_actDockFloat_triggered(self,checked):  
      pass


   @pyqtSlot(bool)  ##停靠区浮动性改变
   def on_dockWidget_topLevelChanged(self,topLevel): 
      pass
   
        
   @pyqtSlot(bool)  ##设置停靠区可见性
   def on_actDockVisible_triggered(self,checked): 
      pass
   

   @pyqtSlot(bool) ##停靠区可见性变化
   def on_dockWidget_visibilityChanged(self,visible):  
      pass

##  ============自定义槽函数================================        

   
##  ===========窗体测试程序=================================        
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
