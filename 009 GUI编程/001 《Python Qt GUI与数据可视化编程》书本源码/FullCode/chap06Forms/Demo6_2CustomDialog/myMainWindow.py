import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QLabel,QDialog

from PyQt5.QtCore import  pyqtSlot,pyqtSignal, Qt, QItemSelectionModel

from PyQt5.QtGui import QStandardItemModel

from ui_MainWindow import Ui_MainWindow

from myDialogSize import QmyDialogSize

from myDialogHeaders import QmyDialogHeaders

from myDialogLocate import QmyDialogLocate

class QmyMainWindow(QMainWindow):

   cellIndexChanged= pyqtSignal(int,int)

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.__dlgSetHeaders=None  
      self.setCentralWidget(self.ui.tableView)

   ##构建状态栏
      self.LabCellPos = QLabel("当前单元格：",self)
      self.LabCellPos.setMinimumWidth(180)
      self.ui.statusBar.addWidget(self.LabCellPos)

      self.LabCellText = QLabel("单元格内容：",self)
      self.LabCellText.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.LabCellText)

   ##构建Item Model/View
      self.itemModel = QStandardItemModel(10,5,self)  #数据模型,10行5列
      self.selectionModel = QItemSelectionModel(self.itemModel) #Item选择模型
      self.selectionModel.currentChanged.connect(self.do_currentChanged)

   ##为tableView设置数据模型
      self.ui.tableView.setModel(self.itemModel)   #设置数据模型
      self.ui.tableView.setSelectionModel(self.selectionModel)    #设置选择模型
##      self.ui.tableView.setSelectionMode(QAbstractItemView.SingleSelection)    #单选
##      self.ui.tableView.setSelectionBehavior(QAbstractItemView.SelectItems)    #单元格选择

##      self.ui.tableView.setAlternatingRowColors(True)
##      self.ui.tableView.verticalHeader().setDefaultSectionSize(25)#缺省行高
        
   def __del__(self):
##      super().__del__(self)
      print("QmyMainWindow 对象被删除了")

##  ==============自定义功能函数============
        
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##设置行数列数对话框
   def on_actTab_SetSize_triggered(self):  
      dlgTableSize=QmyDialogSize() #局部变量，构建时不能传递self
      dlgTableSize.setIniSize(self.itemModel.rowCount(),
                             self.itemModel.columnCount())
      ret=dlgTableSize.exec() #模态方式运行对话框

      if (ret == QDialog.Accepted):
         rows,cols=dlgTableSize.getTableSize()
         self.itemModel.setRowCount(rows)
         self.itemModel.setColumnCount(cols)

   @pyqtSlot()    ##设置表头标题
   def on_actTab_SetHeader_triggered(self): 
      if (self.__dlgSetHeaders == None): #未创建对话框
         self.__dlgSetHeaders=QmyDialogHeaders(self)

      count=len(self.__dlgSetHeaders.headerList())
      if (count != self.itemModel.columnCount()): #列数改变了
         strList=[]
         for i in range(self.itemModel.columnCount()):
             text=str(self.itemModel.headerData(i,Qt.Horizontal,Qt.DisplayRole))
             strList.append(text)   #现有表格标题
         self.__dlgSetHeaders.setHeaderList(strList)

      ret=self.__dlgSetHeaders.exec() #以模态方式运行对话框
      if (ret == QDialog.Accepted):
         strList2=self.__dlgSetHeaders.headerList()
         self.itemModel.setHorizontalHeaderLabels(strList2)
        
   @pyqtSlot()    ##"定位单元格"
   def on_actTab_Locate_triggered(self):
      dlgLocate=QmyDialogLocate(self)
      dlgLocate.setSpinRange(self.itemModel.rowCount(),
                             self.itemModel.columnCount())

      dlgLocate.changeActionEnable.connect(self.do_setActLocateEnable)
      dlgLocate.changeCellText.connect(self.do_setACellText)

      self.cellIndexChanged.connect(dlgLocate.do_setSpinValue)

      dlgLocate.setAttribute(Qt.WA_DeleteOnClose) #对话框关闭时自动删除
      dlgLocate.show()
        
    
##  =============自定义槽函数===============================        
   def do_currentChanged(self,current,previous):
      if (current != None):        #当前模型索引有效
         self.LabCellPos.setText("当前单元格：%d行，%d列"
                               %(current.row(),current.column()))    #显示模型索引的行和列号
         item=self.itemModel.itemFromIndex(current)   #从模型索引获得Item
         self.LabCellText.setText("单元格内容："+item.text())  #显示item的文字内容

         self.cellIndexChanged.emit(current.row(),current.column())

##    @pyqtSlot(bool)
   def do_setActLocateEnable(self,enable):
      self.ui.actTab_Locate.setEnabled(enable)
    
##    @pyqtSlot(int,int,str)
   def do_setACellText(self,row, column, text):
      index=self.itemModel.index(row,column)  #获取模型索引
      self.selectionModel.clearSelection()    #清除现有选择
      self.selectionModel.setCurrentIndex(index,QItemSelectionModel.Select)  #定位到单元格
      self.itemModel.setData(index,text,Qt.DisplayRole)   #设置单元格字符串

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyMainWindow()
   form.show()
   sys.exit(app.exec_())
