import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,QFileDialog,
                     QAbstractItemView, QMessageBox,QDataWidgetMapper)

from PyQt5.QtCore import  pyqtSlot, Qt,QItemSelectionModel,QModelIndex

from PyQt5.QtSql import QSqlDatabase, QSqlQueryModel, QSqlRecord, QSqlQuery  

from PyQt5.QtGui import QPixmap

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setCentralWidget(self.ui.splitter)

#   tableView显示属性设置
      self.ui.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
      self.ui.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
      self.ui.tableView.setAlternatingRowColors(True)
      self.ui.tableView.verticalHeader().setDefaultSectionSize(22)
      self.ui.tableView.horizontalHeader().setDefaultSectionSize(60)
##      self.ui.tableView.resizeColumnsToContents()


##  ==============自定义功能函数============
   def __getFieldNames(self):  ##获取所有字段名称
      emptyRec=self.qryModel.record()     #获取空记录，只有字段名
      self.fldNum={}    #字段名与序号的字典
      for i in range(emptyRec.count()):
         fieldName=emptyRec.fieldName(i)
         self.fldNum.setdefault(fieldName)
         self.fldNum[fieldName]=i
      print (self.fldNum)
   
   def __openTable(self):     ##查询数据
      self.qryModel=QSqlQueryModel(self)
      self.qryModel.setQuery('''SELECT empNo, Name, Gender,  Birthday,  Province,
                             Department, Salary FROM employee ORDER BY empNo''')  

      if self.qryModel.lastError().isValid():
         QMessageBox.critical(self, "错误",
             "数据表查询错误,错误信息\n"+self.qryModel.lastError().text())
         return
      
      self.ui.statusBar.showMessage("记录条数：%d"%self.qryModel.rowCount())
      self.__getFieldNames()  #获取字段名和序号
      
   ##设置字段显示名，直接使用序号
      self.qryModel.setHeaderData(0,   Qt.Horizontal,"工号")
      self.qryModel.setHeaderData(1,   Qt.Horizontal,"姓名")
      self.qryModel.setHeaderData(2,   Qt.Horizontal,"性别")
      self.qryModel.setHeaderData(3,   Qt.Horizontal,"出生日期")
      self.qryModel.setHeaderData(4,   Qt.Horizontal,"省份")
      self.qryModel.setHeaderData(5,   Qt.Horizontal,"部门")
      self.qryModel.setHeaderData(6,   Qt.Horizontal,"工资")
      
##      self.qryModel.setHeaderData(self.fldNum["empNo"],      Qt.Horizontal, "工号")
##      self.qryModel.setHeaderData(self.fldNum["Name"],       Qt.Horizontal, "姓名")
##      self.qryModel.setHeaderData(self.fldNum["Gender"],     Qt.Horizontal, "性别")
##      self.qryModel.setHeaderData(self.fldNum["Birthday"],   Qt.Horizontal, "出生日期")
##      self.qryModel.setHeaderData(self.fldNum["Province"],   Qt.Horizontal, "省份")
##      self.qryModel.setHeaderData(self.fldNum["Department"], Qt.Horizontal, "部门")
##      self.qryModel.setHeaderData(self.fldNum["Salary"],     Qt.Horizontal, "工资")

   ##创建界面组件与数据模型的字段之间的数据映射
      self.mapper= QDataWidgetMapper()
      self.mapper.setModel(self.qryModel)    #设置数据模型
##      self.mapper.setSubmitPolicy(QDataWidgetMapper.AutoSubmit)
    
   ##界面组件与qryModel的具体字段之间的联系
##      self.mapper.addMapping(self.ui.dbSpinEmpNo, self.fldNum["empNo"])
##      self.mapper.addMapping(self.ui.dbEditName,  self.fldNum["Name"])
##      self.mapper.addMapping(self.ui.dbComboSex,  self.fldNum["Gender"])
##      self.mapper.addMapping(self.ui.dbEditBirth, self.fldNum["Birthday"])
##      self.mapper.addMapping(self.ui.dbComboProvince,   self.fldNum["Province"] )
##      self.mapper.addMapping(self.ui.dbComboDep,  self.fldNum["Department"] )
##      self.mapper.addMapping(self.ui.dbSpinSalary,self.fldNum["Salary"] )

      self.mapper.addMapping(self.ui.dbSpinEmpNo,     0)
      self.mapper.addMapping(self.ui.dbEditName,      1)
      self.mapper.addMapping(self.ui.dbComboSex,      2)
      self.mapper.addMapping(self.ui.dbEditBirth,     3)
      self.mapper.addMapping(self.ui.dbComboProvince, 4)
      self.mapper.addMapping(self.ui.dbComboDep,      5)
      self.mapper.addMapping(self.ui.dbSpinSalary,    6)
      self.mapper.toFirst() #移动到首记录

      self.selModel=QItemSelectionModel(self.qryModel)   #关联选择模型
      self.selModel.currentRowChanged.connect(self.do_currentRowChanged)   #选择行变化时

      self.ui.tableView.setModel(self.qryModel)    #设置数据模型
      self.ui.tableView.setSelectionModel(self.selModel) #设置选择模型

      self.ui.actOpenDB.setEnabled(False)

   def __refreshTableView(self): ##刷新tableView显示
      index=self.mapper.currentIndex()
      curIndex=self.qryModel.index(index,1)  #QModelIndex
      self.selModel.clearSelection()         #清空选择项
      self.selModel.setCurrentIndex(curIndex,QItemSelectionModel.Select)

        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()  ##“打开数据库”按钮
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

   @pyqtSlot()    ##首记录
   def on_actRecFirst_triggered(self): 
      self.mapper.toFirst()
      self.__refreshTableView()

   @pyqtSlot()    ##前一记录
   def on_actRecPrevious_triggered(self): 
      self.mapper.toPrevious()
      self.__refreshTableView()

   @pyqtSlot()    ##后一条记录
   def on_actRecNext_triggered(self): 
      self.mapper.toNext()
      self.__refreshTableView()

   @pyqtSlot()    ##最后一条记录
   def on_actRecLast_triggered(self): 
      self.mapper.toLast()
      self.__refreshTableView()

      
##  =============自定义槽函数===============================        
   def do_currentRowChanged(self, current, previous):    ##记录移动时触发
      if (current.isValid() ==False):
         self.ui.dbLabPhoto.clear()    #清除图片显示
         return

      self.mapper.setCurrentIndex(current.row())  #更新数据映射的行号

      first=(current.row()==0)   #是否首记录
      last=(current.row()==self.qryModel.rowCount()-1)   #是否尾记录
      self.ui.actRecFirst.setEnabled(not first)          #更新使能状态
      self.ui.actRecPrevious.setEnabled(not first)
      self.ui.actRecNext.setEnabled(not last)
      self.ui.actRecLast.setEnabled(not last)

      curRec=self.qryModel.record(current.row())  #获取当前记录,QSqlRecord类型
      empNo=curRec.value("EmpNo")      #不需要加 toInt()函数

      query=QSqlQuery(self.DB) 
      query.prepare('''SELECT EmpNo, Memo, Photo FROM employee WHERE EmpNo = :ID''')
      query.bindValue(":ID",empNo)
##      if not query.exec_():  #注意，在PyQt5.11.2之前的版本里只能使用exec_()函数
      if not query.exec():    #注意，在PyQt5.11.2添加了遗漏的overload型exec()函数，在PyQt5.11.2里没问题了
         QMessageBox.critical(self, "错误", "执行SQL语句错误\n"+query.lastError().text())
         return
      else:
         query.first()

      picData=query.value("Photo")
      if (picData==None):  #图片字段内容为空
         self.ui.dbLabPhoto.clear()
      else: #显示照片
         pic=QPixmap()
         pic.loadFromData(picData)
         W=self.ui.dbLabPhoto.size().width()
         self.ui.dbLabPhoto.setPixmap(pic.scaledToWidth(W))

      memoData=query.value("Memo")  #显示备注
      self.ui.dbEditMemo.setPlainText(memoData)
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
