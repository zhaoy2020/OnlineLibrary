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

      self.curPixmap=QPixmap()   #图片
      self.pixRatio=1            #显示比例
      self.itemFlags=(Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
                     | Qt.ItemIsEnabled | Qt.ItemIsAutoTristate)  #节点标志
      self.setCentralWidget(self.ui.scrollArea)
      self.__iniTree()

      #以下的属性设置在UI Designer里已经设置，这里是代码设置方法
      self.ui.dockWidget.setFeatures(QDockWidget.AllDockWidgetFeatures)
      self.ui.dockWidget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
      self.ui.scrollArea.setWidgetResizable(True)     #自动调整内部组件大小
      self.ui.scrollArea.setAlignment(Qt.AlignCenter) 
      self.ui.LabPicture.setAlignment(Qt.AlignCenter)  


##  =================自定义功能函数================================
   def __iniTree(self):     ##初始化目录树
      self.ui.treeFiles.clear()
      icon= QIcon(":/images/icons/15.ico")

      item=QTreeWidgetItem(TreeItemType.itTopItem.value)
      item.setIcon(TreeColNum.colItem.value, icon)
      item.setText(TreeColNum.colItem.value,"图片文件")
      item.setFlags(self.itemFlags)
      item.setCheckState(TreeColNum.colItem.value, Qt.Checked)

      item.setData(TreeColNum.colItem.value, Qt.UserRole,"")
      self.ui.treeFiles.addTopLevelItem(item)

   def __displayImage(self,item):   ##显示节点item 的图片
      filename=item.data(TreeColNum.colItem.value, Qt.UserRole)
      self.ui.statusBar.showMessage(filename)   #状态栏显示文件名

      self.curPixmap.load(filename)    #原始图片
      self.on_actZoomFitH_triggered()  #适合高度显示

      self.ui.actZoomFitH.setEnabled(True)
      self.ui.actZoomFitW.setEnabled(True)
      self.ui.actZoomIn.setEnabled(True)
      self.ui.actZoomOut.setEnabled(True)
      self.ui.actZoomRealSize.setEnabled(True)

   def __changeItemCaption(self,item):    ##递归调用函数，修改节点标题
      title="*"+item.text(TreeColNum.colItem.value)
      item.setText(TreeColNum.colItem.value, title)
      if (item.childCount()>0):
         for i in range(item.childCount()):
            self.__changeItemCaption(item.child(i))
        
        
##  ===========connectSlotsByName() 自动连接的槽函数=================        
   @pyqtSlot()    ##添加目录节点
   def on_actTree_AddFolder_triggered(self):  
      dirStr=QFileDialog.getExistingDirectory()   #选择目录
      if (dirStr == ""):
         return

      parItem=self.ui.treeFiles.currentItem()   #当前节点
      if (parItem == None):
         parItem=self.ui.treeFiles.topLevelItem(0)

      icon= QIcon(":/images/icons/open3.bmp")

      dirObj=QDir(dirStr)     #QDir对象
      nodeText=dirObj.dirName()     #最后一级目录的名称

      item=QTreeWidgetItem(TreeItemType.itGroupItem.value)   #节点类型
      item.setIcon(TreeColNum.colItem.value, icon)
      item.setText(TreeColNum.colItem.value,nodeText)        #第1列
      item.setText(TreeColNum.colItemType.value,"Group")     #第2列
      item.setFlags(self.itemFlags)
      item.setCheckState(TreeColNum.colItem.value, Qt.Checked)

      item.setData(TreeColNum.colItem.value, Qt.UserRole,dirStr)  #关联数据为目录全名
      parItem.addChild(item)
      parItem.setExpanded(True)  #展开节点
        

   @pyqtSlot()    ##添加图片文件节点
   def on_actTree_AddFiles_triggered(self): 
      fileList,flt=QFileDialog.getOpenFileNames(self,
                     "选择一个或多个文件","","Images(*.jpg)")
      #多选文件,返回两个结果，fileList是一个列表类型，存储了所有文件名； flt是设置的文件filter，即"Images(*.jpg)"
      if (len(fileList)<1): #fileList是list[str]
         return

      item=self.ui.treeFiles.currentItem() #当前节点
      if (item.type()==TreeItemType.itImageItem.value): #若当前节点是图片节点，取其父节点作为父节点
         parItem=item.parent()
      else:      #否则取当前节点为父节点
         parItem=item

      icon= QIcon(":/images/icons/31.ico")
      for i in range(len(fileList)):
         fullFileName=fileList[i]        #带路径文件名
         fileinfo=QFileInfo(fullFileName)
         nodeText=fileinfo.fileName()    #不带路径文件名

         item=QTreeWidgetItem(TreeItemType.itImageItem.value)    #节点类型
         item.setIcon(TreeColNum.colItem.value, icon)    #第1列的图标
         item.setText(TreeColNum.colItem.value, nodeText) #第1列的文字
         item.setText(TreeColNum.colItemType.value,"Image")  #第2列的文字
         item.setFlags(self.itemFlags)
         item.setCheckState(TreeColNum.colItem.value, Qt.Checked)

         item.setData(TreeColNum.colItem.value, Qt.UserRole,fullFileName)  #关联数据为文件全名
         parItem.addChild(item)
            
      parItem.setExpanded(True) #展开节点


   @pyqtSlot()    ##删除当前节点
   def on_actTree_DeleteItem_triggered(self): 
      item =self.ui.treeFiles.currentItem()
      parItem=item.parent()
      parItem.removeChild(item)

   @pyqtSlot()    ##遍历节点
   def on_actTree_ScanItems_triggered(self): 
      count=self.ui.treeFiles.topLevelItemCount()
      for i in range(count):
         item=self.ui.treeFiles.topLevelItem(i)
         self.__changeItemCaption(item)

   def on_treeFiles_currentItemChanged(self,current,previous):  ##目录树节点变化
      if (current == None):
         return
      nodeType=current.type() #获取节点类型

      if (nodeType==TreeItemType.itTopItem.value):    #顶层节点
         self.ui.actTree_AddFolder.setEnabled(True)
         self.ui.actTree_AddFiles.setEnabled(True)
         self.ui.actTree_DeleteItem.setEnabled(False) #顶层节点不能删除

      elif (nodeType==TreeItemType.itGroupItem.value):  #组节点        
         self.ui.actTree_AddFolder.setEnabled(True)
         self.ui.actTree_AddFiles.setEnabled(True)
         self.ui.actTree_DeleteItem.setEnabled(True) 
         
      elif (nodeType==TreeItemType.itImageItem.value):    #图片节点       
         self.ui.actTree_AddFolder.setEnabled(False)
         self.ui.actTree_AddFiles.setEnabled(True)
         self.ui.actTree_DeleteItem.setEnabled(True)
         self.__displayImage(current)   #显示图片

   @pyqtSlot()  ##适应高度显示图片
   def on_actZoomFitH_triggered(self): 
      H=self.ui.scrollArea.height() #得到scrollArea的高度
      realH=self.curPixmap.height() #原始图片的实际高度
      self.pixRatio=float(H)/realH  #当前显示比例，必须转换为浮点数
      pix=self.curPixmap.scaledToHeight(H-30) #图片缩放到指定高度
      self.ui.LabPicture.setPixmap(pix)       #设置Label的PixMap

   @pyqtSlot()  ##适应宽度显示
   def on_actZoomFitW_triggered(self): 
      W=self.ui.scrollArea.width()-20
      realW=self.curPixmap.width() 
      self.pixRatio=float(W)/realW
      pix=self.curPixmap.scaledToWidth(W-30) 
      self.ui.LabPicture.setPixmap(pix)       #设置Label的PixMap
       

   @pyqtSlot()   ##实际大小
   def on_actZoomRealSize_triggered(self): 
      self.pixRatio=1  #恢复显示比例为1
      self.ui.LabPicture.setPixmap(self.curPixmap)

   @pyqtSlot()  ##放大显示
   def on_actZoomIn_triggered(self):  
      self.pixRatio=self.pixRatio*1.2
      W=self.pixRatio*self.curPixmap.width()
      H=self.pixRatio*self.curPixmap.height()
      pix=self.curPixmap.scaled(W,H)  #图片缩放到指定高度和宽度，保持长宽比例
      self.ui.LabPicture.setPixmap(pix)

   @pyqtSlot() ##缩小显示
   def on_actZoomOut_triggered(self): 
      self.pixRatio=self.pixRatio*0.8
      W=self.pixRatio*self.curPixmap.width()
      H=self.pixRatio*self.curPixmap.height()
      pix=self.curPixmap.scaled(W,H)    #图片缩放到指定高度和宽度，保持长宽比例
      self.ui.LabPicture.setPixmap(pix)

   @pyqtSlot(bool)  ##设置停靠区浮动性
   def on_actDockFloat_triggered(self,checked):  
      self.ui.dockWidget.setFloating(checked)

   @pyqtSlot(bool)  ##停靠区浮动性改变
   def on_dockWidget_topLevelChanged(self,topLevel): 
      self.ui.actDockFloat.setChecked(topLevel)
        
   @pyqtSlot(bool)  ##设置停靠区可见性
   def on_actDockVisible_triggered(self,checked): 
      self.ui.dockWidget.setVisible(checked)

   @pyqtSlot(bool) ##停靠区可见性变化
   def on_dockWidget_visibilityChanged(self,visible):  
      self.ui.actDockVisible.setChecked(visible)

##  ============自定义槽函数================================        

   
##  ===========窗体测试程序=================================        
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
