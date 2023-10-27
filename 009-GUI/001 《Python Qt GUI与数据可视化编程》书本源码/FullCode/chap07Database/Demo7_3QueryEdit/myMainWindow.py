import sys

from PyQt5.QtWidgets import (QApplication, QMainWindow,QFileDialog,
                     QAbstractItemView, QMessageBox, QDialog)

from PyQt5.QtCore import  pyqtSlot, Qt,QItemSelectionModel,QModelIndex

from PyQt5.QtSql import QSqlDatabase, QSqlQueryModel, QSqlRecord, QSqlQuery  

##from PyQt5.QtGui import QPixmap

from ui_MainWindow import Ui_MainWindow

from myDialogData import QmyDialogData

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setCentralWidget(self.ui.tableView)

#   tableView显示属性设置
##      self.ui.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
##      self.ui.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
      self.ui.tableView.setAlternatingRowColors(True)
      self.ui.tableView.verticalHeader().setDefaultSectionSize(22)
      self.ui.tableView.horizontalHeader().setDefaultSectionSize(60)
##      self.ui.tableView.resizeColumnsToContents()


##  ==============自定义功能函数============
   def __getFieldNames(self):  ##获取所有字段名称
      emptyRec=self.qryModel.record()     #获取空记录，只有字段名
      self.fldNum={}   #字段名与序号的字典
      for i in range(emptyRec.count()):
         fieldName=emptyRec.fieldName(i)
         self.fldNum.setdefault(fieldName)
         self.fldNum[fieldName]=i
      print (self.fldNum)

   
   def __openTable(self):  #查询数据
      self.qryModel=QSqlQueryModel(self)
      self.qryModel.setQuery('''SELECT empNo, Name, Gender,  Birthday,  Province,
                             Department, Salary FROM employee ORDER BY empNo''')  

      if self.qryModel.lastError().isValid():
         QMessageBox.critical(self, "错误",
             "数据表查询错误,错误信息\n"+self.qryModel.lastError().text())
         return
      
      self.__getFieldNames()  #获取字段名和序号
      
   ##字段显示名
      self.qryModel.setHeaderData(0,   Qt.Horizontal,"工号")
      self.qryModel.setHeaderData(1,   Qt.Horizontal,"姓名")
      self.qryModel.setHeaderData(2,   Qt.Horizontal,"性别")
      self.qryModel.setHeaderData(3,   Qt.Horizontal,"出生日期")
      self.qryModel.setHeaderData(4,   Qt.Horizontal,"省份")
      self.qryModel.setHeaderData(5,   Qt.Horizontal,"部门")
      self.qryModel.setHeaderData(6,   Qt.Horizontal,"工资")

   ##      self.qryModel.setHeaderData(self.fldNum["empNo"],  Qt.Horizontal,"工号")
   ##      self.qryModel.setHeaderData(self.fldNum["Name"],   Qt.Horizontal,"姓名")
   ##      self.qryModel.setHeaderData(self.fldNum["Gender"], Qt.Horizontal,"性别")
   ##      self.qryModel.setHeaderData(self.fldNum["Birthday"],  Qt.Horizontal,"出生日期")
   ##      self.qryModel.setHeaderData(self.fldNum["Province"],  Qt.Horizontal,"省份")
   ##      self.qryModel.setHeaderData(self.fldNum["Department"],Qt.Horizontal,"部门")
   ##      self.qryModel.setHeaderData(self.fldNum["Salary"], Qt.Horizontal,"工资")
      
      self.selModel=QItemSelectionModel(self.qryModel)  #关联选择模型
   ##选择行变化时
      self.selModel.currentRowChanged.connect(self.do_currentRowChanged)

      self.ui.tableView.setModel(self.qryModel) #设置数据模型
      self.ui.tableView.setSelectionModel(self.selModel) #设置选择模型
      
      self.ui.actOpenDB.setEnabled(False)

      self.ui.actRecInsert.setEnabled(True)
      self.ui.actRecDelete.setEnabled(True)
      self.ui.actRecEdit.setEnabled(True)
      self.ui.actScan.setEnabled(True)
      self.ui.actTestSQL.setEnabled(True)

   def __updateRecord(self,recNo): ##更新一条记录
      curRec=self.qryModel.record(recNo)  #获取当前记录
      empNo=curRec.value("EmpNo")   #获取EmpNo

      query=QSqlQuery(self.DB)   #查询出当前记录的所有字段
      query.prepare("SELECT * FROM employee WHERE EmpNo = :ID")
      query.bindValue(":ID",empNo)
      query.exec() #
      query.first()

      if (not query.isValid()): #是否为有效记录
         return

      curRec=query.record()   #获取当前记录的数据，QSqlRecord类型
      dlgData=QmyDialogData(self)   #创建对话框

      dlgData.setUpdateRecord(curRec)  #调用对话框函数更新数据和界面
      ret=dlgData.exec()  # 以模态方式显示对话框
      if (ret != QDialog.Accepted):
         return
      
      recData=dlgData.getRecordData()  #获得对话框返回的记录
      query.prepare('''UPDATE employee SET Name=:Name, Gender=:Gender,
                    Birthday=:Birthday, Province=:Province,
                    Department=:Department, Salary=:Salary,
                    Memo=:Memo, Photo=:Photo WHERE EmpNo = :ID''')

      query.bindValue(":Name",      recData.value("Name"))
      query.bindValue(":Gender",    recData.value("Gender"))
      query.bindValue(":Birthday",  recData.value("Birthday"))
      query.bindValue(":Province",  recData.value("Province"))
      query.bindValue(":Department",recData.value("Department"))
      query.bindValue(":Salary", recData.value("Salary"))
      query.bindValue(":Memo",   recData.value("Memo"))
      query.bindValue(":Photo",  recData.value("Photo"))
      
      query.bindValue(":ID",     empNo)

##      if (not query.exec_()):
      if (not query.exec()):  #PyQt 5.11.2以前应该使用exec_()函数
         QMessageBox.critical(self, "错误",
                     "记录更新错误\n"+query.lastError().text())
      else:
         self.qryModel.query().exec()  #数据模型重新查询数据，更新tableView显示
      
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##打开数据库
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
      if self.DB.open():   #打开数据库
         self.__openTable()  #查询数据
      else:         
         QMessageBox.warning(self, "错误", "打开数据库失败")


   @pyqtSlot()    ##插入记录
   def on_actRecInsert_triggered(self):
      query=QSqlQuery(self.DB)
      query.exec("select * from employee where EmpNo =-1") #实际不查询出记录，只查询字段信息

      curRec=query.record()   #获取当前记录,实际为空记录,但有字段信息
      curRec.setValue("EmpNo",self.qryModel.rowCount()+3000)

      dlgData= QmyDialogData(self)
      dlgData.setInsertRecord(curRec)  #插入记录
      
      ret=dlgData.exec()  #以模态方式显示对话框
      if (ret != QDialog.Accepted):
         return
      
      recData=dlgData.getRecordData()

      query.prepare('''INSERT INTO employee (EmpNo,Name,Gender,Birthday,
                    Province,Department,Salary,Memo,Photo) 
                    VALUES(:EmpNo,:Name, :Gender,:Birthday,:Province,
                    :Department,:Salary,:Memo,:Photo)''')

      query.bindValue(":EmpNo",     recData.value("EmpNo"))
      query.bindValue(":Name",      recData.value("Name"))
      query.bindValue(":Gender",    recData.value("Gender"))
      query.bindValue(":Birthday",  recData.value("Birthday"))

      query.bindValue(":Province",  recData.value("Province"))
      query.bindValue(":Department",recData.value("Department"))

      query.bindValue(":Salary",    recData.value("Salary"));
      query.bindValue(":Memo",      recData.value("Memo"));
      query.bindValue(":Photo",     recData.value("Photo"));

      res=query.exec()   #执行SQL语句
      if (res==False):
         QMessageBox.critical(self, "错误",
                  "插入记录错误\n"+query.lastError().text())
      else:     #插入，删除记录后需要重新设置SQL语句查询
         sqlStr=self.qryModel.query().executedQuery()  #执行过的SELECT语句
         self.qryModel.setQuery(sqlStr)   #reset 重新查询数据

   @pyqtSlot()    ##删除记录
   def on_actRecDelete_triggered(self):
      curRecNo=self.selModel.currentIndex().row()
      curRec=self.qryModel.record(curRecNo)  #获取当前记录
      if (curRec.isEmpty()): #当前为空记录
        return

      empNo=curRec.value("EmpNo")   #获取员工编号
      query=QSqlQuery(self.DB)
      query.prepare("DELETE  FROM employee WHERE EmpNo = :ID")
      query.bindValue(":ID",empNo)

      if (query.exec()==False):
         QMessageBox.critical(self, "错误",
               "删除记录出现错误\n"+query.lastError().text())
      else:   #插入，删除记录后需要重新设置SQL语句查询
         sqlStr=self.qryModel.query().executedQuery()  #执行过的SELECT语句
         self.qryModel.setQuery(sqlStr)   #reset 重新查询数据
      
   @pyqtSlot()    ##编辑记录
   def on_actRecEdit_triggered(self):
      curRecNo=self.selModel.currentIndex().row()
      self.__updateRecord(curRecNo)

   ##   @pyqtSlot()  ##双击编辑记录
   def on_tableView_doubleClicked(self,index):
      curRecNo=index.row()
      self.__updateRecord(curRecNo)

   @pyqtSlot()    ##遍历记录，涨工资
   def on_actScan_triggered(self):
      qryEmpList=QSqlQuery(self.DB) #员工工资信息列表
      qryEmpList.exec("SELECT empNo,Salary FROM employee ORDER BY empNo")
      
      qryUpdate=QSqlQuery(self.DB) #临时 QSqlQuery
      qryUpdate.prepare('''UPDATE employee SET Salary=:Salary WHERE EmpNo = :ID''')

      qryEmpList.first()
      while (qryEmpList.isValid()): #当前记录有效
         empID=qryEmpList.value("empNo")     #获取empNo
         salary=qryEmpList.value("Salary")   #获取Salary
         salary=salary+500          #涨工资

         qryUpdate.bindValue(":ID",empID)
         qryUpdate.bindValue(":Salary",salary)  #设置SQL语句参数
         qryUpdate.exec()  #执行update语句

         if not qryEmpList.next():   #移动到下一条记录，并判断是否到末尾了
            break

      self.qryModel.query().exec()  #数据模型重新查询数据，更新tableView的显示
      QMessageBox.information(self, "提示", "涨工资计算完毕")

   @pyqtSlot()    ##SQL语句测试
   def on_actTestSQL_triggered(self):
      query=QSqlQuery(self.DB)
      
##   # SQL语句测试1，  exec_() 和exec()都可以直接执行不带参数的SQL语句    
##      query.exec('''UPDATE employee SET Salary=3000 where Gender="女" ''')
##      query.exec_('''UPDATE employee SET Salary=4500 where Gender="女" ''')

   # SQL语句测试2，执行带参数的SQL语句，只能用 exec_()，不能用exec()
##      query.prepare('''UPDATE employee SET Salary=9000 where Gender=:Gender ''')
##      query.bindValue(":Gender","男")
##      query.exec()  

      query.exec('''UPDATE employee SET Salary=500+Salary ''')
##      query.bindValue(":Gender","男")
##      query.exec()  

##      query.prepare("UPDATE employee SET Department=?, Salary=?  where Name=?")
##      query.bindValue(0, "技术部")
##      query.bindValue(1, 5500)
##      query.bindValue(2, "张三")
##      query.exec_()  #只能用exec_()，而不能用exec()函数

##      self.qryModel.query().exec()  #不增减记录时更新显示

   ##    增减记录后的更新显示
      sqlStr=self.qryModel.query().executedQuery()  #执行过的SELECT语句
      self.qryModel.setQuery(sqlStr) #reset 重新查询数据

      print("SQL OK")
      
##  =============自定义槽函数===============================        
   def do_currentRowChanged(self, current, previous): ##行切换时触发
      if (current.isValid() ==False):
         return
      curRec=self.qryModel.record(current.row())  #获取当前记录,QSqlRecord类型
      empNo=curRec.value("EmpNo")      #不需要加 toInt()函数
      self.ui.statusBar.showMessage("当前记录：工号=%d"%empNo)

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
