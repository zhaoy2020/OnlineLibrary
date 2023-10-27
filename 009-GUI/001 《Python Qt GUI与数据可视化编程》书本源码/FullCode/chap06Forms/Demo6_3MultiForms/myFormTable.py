import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,
                     QLabel,QAbstractItemView,QDialog)

from PyQt5.QtCore import  pyqtSlot, Qt, QItemSelectionModel

from PyQt5.QtGui import QStandardItemModel

from ui_QWFormTable import Ui_QWFormTable

from myDialogSize import QmyDialogSize

from myDialogHeaders import QmyDialogHeaders

class QmyFormTable(QMainWindow):

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_QWFormTable()    #创建UI对象
      self.ui.setupUi(self)       #构造UI界面

      self.__dlgSetHeaders=None  
      self.setAutoFillBackground(True)

      self.setCentralWidget(self.ui.tableView)
      self.ui.tableView.setAlternatingRowColors(True)
##        self.ui.tableView.verticalHeader().setDefaultSectionSize(25)  #缺省行高
##        self.ui.tableView.setSelectionMode(QAbstractItemView.SingleSelection)  #单选
##        self.ui.tableView.setSelectionBehavior(QAbstractItemView.SelectItems)  #单元格选择
        
      #构建Model/View
      self.itemModel = QStandardItemModel(10,5,self)  #数据模型,10行5列
      self.selectionModel = QItemSelectionModel(self.itemModel)   #Item选择模型
      self.ui.tableView.setModel(self.itemModel)      #设置数据模型
      self.ui.tableView.setSelectionModel(self.selectionModel)    #设置选择模型
        
   def __del__(self): ##析构函数
      print("QmyFormTable 对象被删除了")

##  ==============自定义功能函数============
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##设置表格大小
   def on_actSetSize_triggered(self): 
      dlgTableSize=QmyDialogSize()  #局部变量，构建时不能传递self
      dlgTableSize.setIniSize(self.itemModel.rowCount(),
                             self.itemModel.columnCount())
      ret=dlgTableSize.exec()

      if (ret == QDialog.Accepted):
         rows,cols=dlgTableSize.getTableSize()
         self.itemModel.setRowCount(rows)
         self.itemModel.setColumnCount(cols)

   @pyqtSlot()    ##设置表头标题
   def on_actSetHeader_triggered(self):  
      if (self.__dlgSetHeaders == None):
         self.__dlgSetHeaders=QmyDialogHeaders(self)

      count=len(self.__dlgSetHeaders.headerList())
      if (count != self.itemModel.columnCount()):
         strList=[]
         for i in range(self.itemModel.columnCount()):
            text=str(self.itemModel.headerData(i,Qt.Horizontal,Qt.DisplayRole))
            strList.append(text)
         self.__dlgSetHeaders.setHeaderList(strList)

      ret=self.__dlgSetHeaders.exec()
      if (ret == QDialog.Accepted):
         strList2=self.__dlgSetHeaders.headerList()
         self.itemModel.setHorizontalHeaderLabels(strList2)
        
##  =============自定义槽函数===============================        
        
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyFormTable()
   form.show()
   sys.exit(app.exec_())
