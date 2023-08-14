import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QFileSystemModel

##from PyQt5.QtCore import  pyqtSlot,QDir

from PyQt5.QtCore import  QDir

##from PyQt5.QtWidgets import  QFileSystemModel

##from PyQt5.QtGui import

from ui_MainWindow import Ui_MainWindow


class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面

      self.__buildModelView()

##  ==============自定义功能函数============

   def  __buildModelView(self):   ##构造Model/View 系统
      self.model=QFileSystemModel(self)
      self.model.setRootPath(QDir.currentPath())

      self.ui.treeView.setModel(self.model)  #设置数据模型
      self.ui.listView.setModel(self.model)
      self.ui.tableView.setModel(self.model)

      self.ui.treeView.clicked.connect(self.ui.listView.setRootIndex)
      self.ui.treeView.clicked.connect(self.ui.tableView.setRootIndex)
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
    
   def on_treeView_clicked(self,index):  ##treeView单击
      self.ui.chkBox_IsDir.setChecked(self.model.isDir(index)) #是否是目录

      self.ui.LabPath.setText(self.model.filePath(index))      #目录名
      self.ui.LabType.setText(self.model.type(index))          #节点类型
      self.ui.LabFileName.setText(self.model.fileName(index))  #文件名

      fileSize=self.model.size(index)/1024
      if (fileSize<1024):
         self.ui.LabFileSize.setText("%d KB"% fileSize)
      else:
         self.ui.LabFileSize.setText("%.2f MB"% (fileSize/1024.0))
        
        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":       
    app = QApplication(sys.argv)  
    form=QmyMainWindow()          
    form.show()
    sys.exit(app.exec_())
