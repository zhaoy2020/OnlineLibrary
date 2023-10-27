import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,
              QLabel,QTableWidgetItem,QAbstractItemView)

from enum import Enum  ##枚举类型

from PyQt5.QtCore import  pyqtSlot, Qt,QDate

from PyQt5.QtGui import  QFont, QBrush, QIcon

from ui_MainWindow import Ui_MainWindow

class CellType(Enum):    ##各单元格的类型
   ctName=1000
   ctSex =1001
   ctBirth =1002
   ctNation=1003
   ctScore=1004
   ctPartyM=1005

class FieldColNum(Enum):   ##各字段在表格中的列号
   colName=0
   colSex=1
   colBirth=2
   colNation=3
   colScore=4
   colPartyM=5

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
##      self.setWindowTitle("Demo3_10，QTableWidget的使用")

      self.LabCellIndex=QLabel("当前单元格坐标：",self)
      self.LabCellIndex.setMinimumWidth(250)
      self.LabCellType=QLabel("当前单元格类型：",self)
      self.LabCellType.setMinimumWidth(200)
      self.LabStudID=QLabel("学生ID：",self)
      self.LabStudID.setMinimumWidth(200)
      self.ui.statusBar.addWidget(self.LabCellIndex)  #加到状态栏
      self.ui.statusBar.addWidget(self.LabCellType)
      self.ui.statusBar.addWidget(self.LabStudID)

      self.ui.tableInfo.setAlternatingRowColors(True) #交替行颜色
      self.__tableInitialized=False   #表格数据未初始化


##  ==============自定义功能函数============
   def __createItemsARow(self,rowNo,name,sex,birth,nation,isParty,score): ##创建一行的items
      StudID=201805000+rowNo  #学号

      #姓名
      item=QTableWidgetItem(name,CellType.ctName.value)
      item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
      font=item.font()
      font.setBold(True)
      item.setFont(font)
      item.setData(Qt.UserRole,StudID) #关联数据
      self.ui.tableInfo.setItem(rowNo,FieldColNum.colName.value,item)

      #性别
      if (sex=="男"):
         icon=QIcon(":/icons/images/boy.ico")
      else:
         icon=QIcon(":/icons/images/girl.ico")
      item=QTableWidgetItem(sex,CellType.ctSex.value)
      item.setIcon(icon)
      item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
      self.ui.tableInfo.setItem(rowNo,FieldColNum.colSex.value,item)

      #出生日期
      strBitrh=birth.toString("yyyy-MM-dd")  #日期转换为字符串
      item=QTableWidgetItem(strBitrh,CellType.ctBirth.value)
      item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
      self.ui.tableInfo.setItem(rowNo,FieldColNum.colBirth.value,item)

      #民族
      item=QTableWidgetItem(nation,CellType.ctNation.value)
      item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
      if (nation != "汉族"):
         item.setForeground(QBrush(Qt.blue)) 
      self.ui.tableInfo.setItem(rowNo,FieldColNum.colNation.value,item)

      #分数
      strScore=str(score)
      item=QTableWidgetItem(strScore,CellType.ctScore.value)
      item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
      self.ui.tableInfo.setItem(rowNo,FieldColNum.colScore.value,item)

      #党员
      item=QTableWidgetItem("党员",CellType.ctPartyM.value)
      item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
      if (isParty==True):
         item.setCheckState(Qt.Checked)
      else:
         item.setCheckState(Qt.Unchecked)
      item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable) #不允许编辑文字
      item.setBackground(QBrush(Qt.yellow))      #Qt::green  lightGray  yellow
      self.ui.tableInfo.setItem(rowNo,FieldColNum.colPartyM.value,item)   #为单元格设置Item
        
       
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##“设置表头”按钮
   def on_btnSetHeader_clicked(self): 
      headerText=["姓 名","性 别","出生日期","民 族","分数","是否党员"] 
      self.ui.tableInfo.setColumnCount(len(headerText)) #列数
      ##        self.ui.tableInfo.setHorizontalHeaderLabels(headerText)   #简单的表头文字，无格式
        
      for i in range(len(headerText)):
         headerItem=QTableWidgetItem(headerText[i])
         font=headerItem.font()
      ##  font.setBold(True)
         font.setPointSize(11)
         headerItem.setFont(font)
         headerItem.setForeground(QBrush(Qt.red))  #前景色，即文字颜色
         self.ui.tableInfo.setHorizontalHeaderItem(i,headerItem)

   @pyqtSlot()    ##设置行数
   def on_btnSetRows_clicked(self):  
      self.ui.tableInfo.setRowCount(self.ui.spinRowCount.value()) #设置数据区行数
      self.ui.tableInfo.setAlternatingRowColors(self.ui.chkBoxRowColor.isChecked()) #设置交替行背景颜色
    

   @pyqtSlot()    ##初始化表格数据
   def on_btnIniData_clicked(self):  
      self.ui.tableInfo.clearContents() #清除表格内容

      birth=QDate(1998,6,23)
      isParty=True
      nation="汉族"
      score=70
        
      rowCount=self.ui.tableInfo.rowCount()   #表格行数
      for i in range(rowCount):
         strName="学生%d"%i
         if ((i % 2)==0):
            strSex="男"
         else:
            strSex="女"
         self.__createItemsARow(i,strName,strSex,
                                birth,nation,isParty,score)
         birth=birth.addDays(20)
         isParty=not isParty

      self.__tableInitialized=True   #表格数据已初始化
            

   @pyqtSlot()    ##插入行
   def on_btnInsertRow_clicked(self):      
      curRow=self.ui.tableInfo.currentRow()   #当前行号
      self.ui.tableInfo.insertRow(curRow)
      birth=QDate.fromString("1998-4-5","yyyy-M-d")
      self.__createItemsARow(curRow, "新学生", "男",birth,"苗族",True,65)
        
   @pyqtSlot()    ##添加行
   def on_btnAppendRow_clicked(self):  
      curRow=self.ui.tableInfo.rowCount()   
      self.ui.tableInfo.insertRow(curRow)
      birth=QDate.fromString("1999-1-10","yyyy-M-d")
      self.__createItemsARow(curRow, "新生", " 女",birth,"土家族",False,86)
        
   @pyqtSlot()    ##删除当前行
   def on_btnDelCurRow_clicked(self):  
      curRow=self.ui.tableInfo.currentRow()   #当前行号
      self.ui.tableInfo.removeRow(curRow)
        
   @pyqtSlot()    ##清空表格内容
   def on_btnClearContents_clicked(self): 
      self.ui.tableInfo.clearContents()

   @pyqtSlot()    ##自动行高
   def on_btnAutoHeight_clicked(self): 
      self.ui.tableInfo.resizeRowsToContents()
        
   @pyqtSlot()    ##自动列宽
   def on_btnAutoWidth_clicked(self): 
      self.ui.tableInfo.resizeColumnsToContents()

   @pyqtSlot(bool)  ##表格可编辑
   def on_chkBoxEditable_clicked(self,checked):  
      if (checked):
         trig=QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
      else:
         trig=QAbstractItemView.NoEditTriggers #不允许编辑
      self.ui.tableInfo.setEditTriggers(trig) #不允许编辑

   @pyqtSlot(bool)      ##交替行颜色
   def on_chkBoxRowColor_clicked(self,checked):    
      self.ui.tableInfo.setAlternatingRowColors(checked)

   @pyqtSlot(bool)   ##是否显示水平表头
   def on_chkBoxHeaderH_clicked(self,checked):  
      self.ui.tableInfo.horizontalHeader().setVisible(checked)

   @pyqtSlot(bool) #是否显示垂直表头
   def on_chkBoxHeaderV_clicked(self,checked):    
      self.ui.tableInfo.verticalHeader().setVisible(checked)

   @pyqtSlot()    ##选择行为：行选择
   def on_radioSelectRow_clicked(self): 
      selMode=QAbstractItemView.SelectRows
      self.ui.tableInfo.setSelectionBehavior(selMode)
        
   @pyqtSlot()    ##选择行为：单元格选择
   def on_radioSelectItem_clicked(self):
      selMode=QAbstractItemView.SelectItems
      self.ui.tableInfo.setSelectionBehavior(selMode)


   @pyqtSlot()    ##读取表格到文本
   def on_btnReadToText_clicked(self): 
      self.ui.textEdit.clear()
      rowCount=self.ui.tableInfo.rowCount()  #行数
      colCount=self.ui.tableInfo.columnCount() #列数
      for i in range(rowCount):
         strText="第 %d 行： " %(i+1)
         for j in range(colCount-1):
            cellItem=self.ui.tableInfo.item(i,j)
            strText =strText+cellItem.text()+"   "
         cellItem=self.ui.tableInfo.item(i,colCount-1) #最后一列
         if (cellItem.checkState() == Qt.Checked):
            strText=strText+"党员"
         else:
            strText=strText+"群众"
         self.ui.textEdit.appendPlainText(strText)


   @pyqtSlot(int,int,int,int)   ##当前单元格发生变化
   def on_tableInfo_currentCellChanged(self,currentRow,currentColumn,
                                       previousRow,previousColumn):
      if (self.__tableInitialized ==False):   #表格数据未初始化
         return
      item=self.ui.tableInfo.item(currentRow,currentColumn)#当前单元格
      if (item == None):
         return

      self.LabCellIndex.setText("当前单元格：%d 行，%d 列"
                               %(currentRow,currentColumn))

      itemCellType=item.type()    #获取单元格的类型
      self.LabCellType.setText("当前单元格类型：%d" %itemCellType)

      item2=self.ui.tableInfo.item(currentRow,FieldColNum.colName.value)
      studID=item2.data(Qt.UserRole)   #读取用户自定义数据
      self.LabStudID.setText("学生ID：%d" %studID)
        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================        
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
