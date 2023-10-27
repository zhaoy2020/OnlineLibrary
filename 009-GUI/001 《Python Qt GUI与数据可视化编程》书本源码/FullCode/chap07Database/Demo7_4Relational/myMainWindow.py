import sys

from PyQt5.QtWidgets import (QApplication, QMainWindow,QFileDialog,
                           QAbstractItemView, QMessageBox)

from PyQt5.QtCore import  pyqtSlot, Qt,QItemSelectionModel,QModelIndex

from PyQt5.QtSql import (QSqlDatabase, QSqlRelationalTableModel, 
            QSqlTableModel, QSqlRecord, QSqlRelation,QSqlRelationalDelegate)

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setCentralWidget(self.ui.tableView)

#   tableView显示属性设置
      self.ui.tableView.setSelectionBehavior(QAbstractItemView.SelectItems)
      self.ui.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
      self.ui.tableView.setAlternatingRowColors(True)

      self.ui.tableView.verticalHeader().setDefaultSectionSize(22)
      self.ui.tableView.horizontalHeader().setDefaultSectionSize(100)


##  ==============自定义功能函数============
   def __getFieldNames(self):  ##获取所有字段名称
      emptyRec=self.tabModel.record()     #获取空记录，只有字段名
      self.fldNum={}   #字段名与序号的字典
      for i in range(emptyRec.count()):
         fieldName=emptyRec.fieldName(i)
         self.fldNum.setdefault(fieldName)
         self.fldNum[fieldName]=i
      print (self.fldNum)
   
   def __openTable(self):   ##打开数据表
      self.tabModel=QSqlRelationalTableModel(self,self.DB)     #数据表
      
      self.tabModel.setTable("studInfo")  #设置数据表
      self.tabModel.setEditStrategy(QSqlTableModel.OnManualSubmit)   #数据保存方式，OnManualSubmit , OnRowChange
      self.tabModel.setSort(self.tabModel.fieldIndex("studID"),Qt.AscendingOrder)   #排序
      
      if (self.tabModel.select() == False):  #查询数据失败
         QMessageBox.critical(self, "错误信息",
              "打开数据表错误,错误信息\n"+self.tabModel.lastError().text())
         return

      self.__getFieldNames()  #获取字段名和序号
      
   ##字段显示名
      self.tabModel.setHeaderData(self.fldNum["studID"],    Qt.Horizontal,"学号")
      self.tabModel.setHeaderData(self.fldNum["name"],      Qt.Horizontal,"姓名")
      self.tabModel.setHeaderData(self.fldNum["gender"],    Qt.Horizontal,"性别")
      self.tabModel.setHeaderData(self.fldNum["departID"],  Qt.Horizontal,"学院")
      self.tabModel.setHeaderData(self.fldNum["majorID"],   Qt.Horizontal,"专业")

   ##    设置代码字段的查询关系数据表
      self.tabModel.setRelation(self.fldNum["departID"],
                                QSqlRelation("departments","departID","department"))   #学院
      self.tabModel.setRelation(self.fldNum["majorID"],
                                QSqlRelation("majors","majorID","major"))  #专业


      self.selModel=QItemSelectionModel(self.tabModel)  #关联选择模型

   ##selModel当前项变化时触发currentChanged信号
      self.selModel.currentChanged.connect(self.do_currentChanged)
   ##选择行变化时
   ##      self.selModel.currentRowChanged.connect(self.do_currentRowChanged)

      self.ui.tableView.setModel(self.tabModel)    #设置数据模型
      self.ui.tableView.setSelectionModel(self.selModel)    #设置选择模型

      delgate=QSqlRelationalDelegate(self.ui.tableView)
      self.ui.tableView.setItemDelegate(delgate)  #为关系型字段设置缺省代理组件

      self.tabModel.select()  #必须重新查询数据
   ##更新actions和界面组件的使能状态
      self.ui.actOpenDB.setEnabled(False)

      self.ui.actRecAppend.setEnabled(True)
      self.ui.actRecInsert.setEnabled(True)
      self.ui.actRecDelete.setEnabled(True)
      self.ui.actFields.setEnabled(True)
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()
   def on_actOpenDB_triggered(self):
      dbFilename,flt=QFileDialog.getOpenFileName(self,"选择数据库文件","",
                             "SQL Lite数据库(*.db *.db3)")
      if (dbFilename==''):
         return

      #打开数据库
      self.DB=QSqlDatabase.addDatabase("QSQLITE")    #添加 SQL LITE数据库驱动
      self.DB.setDatabaseName(dbFilename)   #设置数据库名称
   ##    DB.setHostName()
   ##    DB.setUserName()
   ##    DB.setPassword()
      if self.DB.open():      #打开数据库
         self.__openTable()   #打开数据表
      else:         
         QMessageBox.warning(self, "错误", "打开数据库失败")

   @pyqtSlot()    ##保存修改
   def on_actSubmit_triggered(self):   
      res=self.tabModel.submitAll()
      if (res==False):
         QMessageBox.information(self, "消息",
                  "数据保存错误,错误信息\n"+self.tabModel.lastError().text())
      else:
         self.ui.actSubmit.setEnabled(False)
         self.ui.actRevert.setEnabled(False)

   @pyqtSlot()    ##取消修改
   def on_actRevert_triggered(self): 
      self.tabModel.revertAll()
      self.ui.actSubmit.setEnabled(False)
      self.ui.actRevert.setEnabled(False)

   @pyqtSlot()    ##添加记录
   def on_actRecAppend_triggered(self):
      self.tabModel.insertRow(self.tabModel.rowCount(),QModelIndex())   #在末尾添加一个记录
      curIndex=self.tabModel.index(self.tabModel.rowCount()-1,1)        #创建最后一行的ModelIndex
      self.selModel.clearSelection()  #清空选择项
      self.selModel.setCurrentIndex(curIndex,QItemSelectionModel.Select)  #设置刚插入的行为当前选择行

   @pyqtSlot()    ##插入记录
   def on_actRecInsert_triggered(self):
      curIndex=self.ui.tableView.currentIndex()   #QModelIndex
      self.tabModel.insertRow(curIndex.row(),QModelIndex())
      self.selModel.clearSelection()      #清除已有选择
      self.selModel.setCurrentIndex(curIndex,QItemSelectionModel.Select)

   @pyqtSlot()    ##删除记录
   def on_actRecDelete_triggered(self):
      curIndex=self.selModel.currentIndex()     #获取当前选择单元格的模型索引
      self.tabModel.removeRow(curIndex.row())   #删除最后一行

   @pyqtSlot()    ##显示字段列表
   def on_actFields_triggered(self):
      emptyRec=self.tabModel.record()     #获取空记录，只有字段名
      str=''
      for i in range(emptyRec.count()):
         str=str+emptyRec.fieldName(i)+'\n'
      QMessageBox.information(self, "所有字段名", str)
      
      
##  =============自定义槽函数===============================        
   def do_currentChanged(self,current,previous):   ##更新actPost和actCancel 的状态
      self.ui.actSubmit.setEnabled(self.tabModel.isDirty()) #有未保存修改时可用
      self.ui.actRevert.setEnabled(self.tabModel.isDirty())

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
