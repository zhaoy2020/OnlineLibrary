import sys,os

from PyQt5.QtWidgets import  (QApplication, QMainWindow,
                   QLabel, QAbstractItemView, QFileDialog)

##from PyQt5.QtWidgets import  QLabel, QAbstractItemView, QFileDialog

from PyQt5.QtCore import  Qt, pyqtSlot,QItemSelectionModel, QModelIndex
##from PyQt5.QtCore import   QItemSelectionModel, QModelIndex


from PyQt5.QtGui import QStandardItemModel, QStandardItem

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      self.setCentralWidget(self.ui.splitter)

      self.__ColCount=6  #列数=6
      self.itemModel = QStandardItemModel(5,self.__ColCount,self)# 数据模型,10行6列

      self.selectionModel = QItemSelectionModel(self.itemModel) #Item选择模型
      self.selectionModel.currentChanged.connect(self.do_curChanged)

      self.__lastColumnTitle="测井取样"
      self.__lastColumnFlags=(Qt.ItemIsSelectable
                              | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)

      ##tableView设置
      self.ui.tableView.setModel(self.itemModel) #设置数据模型
      self.ui.tableView.setSelectionModel(self.selectionModel)    #设置选择模型

      oneOrMore=QAbstractItemView.ExtendedSelection
      self.ui.tableView.setSelectionMode(oneOrMore) #可多选

      itemOrRow=QAbstractItemView.SelectItems
      self.ui.tableView.setSelectionBehavior(itemOrRow)   #单元格选择
      
      self.ui.tableView.verticalHeader().setDefaultSectionSize(22)#缺省行高
      self.ui.tableView.setAlternatingRowColors(True)   #交替行颜色

      self.ui.tableView.setEnabled(False)  #先禁用tableView
      self.__buildStatusBar()
      

##  ==============自定义功能函数============
   def __buildStatusBar(self):   ##构建状态栏
      self.LabCellPos = QLabel("当前单元格：",self)
      self.LabCellPos.setMinimumWidth(180)
      self.ui.statusBar.addWidget(self.LabCellPos)

      self.LabCellText = QLabel("单元格内容：",self)
      self.LabCellText.setMinimumWidth(150)
      self.ui.statusBar.addWidget(self.LabCellText)

      self.LabCurFile = QLabel("当前文件：",self)
      self.ui.statusBar.addPermanentWidget(self.LabCurFile)
        

   def __iniModelFromStringList(self,allLines):  ##从字符串列表构建模型
      rowCnt=len(allLines) #文本行数，第1行是标题
      self.itemModel.setRowCount(rowCnt-1) #实际数据行数

      headerText=allLines[0].strip() #第1行是表头,去掉末尾的换行符 "\n"
      headerList=headerText.split("\t")      #转换为字符串列表
      self.itemModel.setHorizontalHeaderLabels(headerList)  #设置表头标题

      self.__lastColumnTitle=headerList[len(headerList)-1] # 最后一列表头的标题，即“测井取样”

      lastColNo=self.__ColCount-1  #最后一列的列号
      for i in range(rowCnt-1):  
         lineText=allLines[i+1].strip()  #一行的文字，\t分隔
         strList=lineText.split("\t")    #分割为字符串列表 
         for j in range(self.__ColCount-1):  #不含最后一列
            item=QStandardItem(strList[j])
            self.itemModel.setItem(i,j,item)  #设置模型的item
             
         item=QStandardItem(self.__lastColumnTitle)  #最后一列
         item.setFlags(self.__lastColumnFlags)
         item.setCheckable(True)
         if (strList[lastColNo]=="0"):
            item.setCheckState(Qt.Unchecked)
         else:
            item.setCheckState(Qt.Checked)

         self.itemModel.setItem(i,lastColNo,item) #设置最后一列的item
            

   def  __setCellAlignment(self, align=Qt.AlignHCenter):
      if (not self.selectionModel.hasSelection()): #没有选择的项
         return
      selectedIndex=self.selectionModel.selectedIndexes()  #模型索引列表
      count=len(selectedIndex)
      for i in range(count):
         index=selectedIndex[i]  #获取其中的一个模型索引
         item=self.itemModel.itemFromIndex(index)    #获取一个单元格的项数据对象
         item.setTextAlignment(align) #设置文字对齐方式
        
    
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()  ##“打开文件”
   def on_actOpen_triggered(self):  
   ##        curPath=QDir.currentPath() #获取当前路径
      curPath=os.getcwd()   #获取当前路径
      
      filename,flt=QFileDialog.getOpenFileName(self,"打开一个文件",curPath,
                  "井斜数据文件(*.txt);;所有文件(*.*)")
      if (filename==""):
         return

      self.LabCurFile.setText("当前文件："+filename)
      self.ui.plainTextEdit.clear()

      aFile=open(filename,'r')
      allLines=aFile.readlines()  #读取所有行,list类型,每行末尾带有 \n
      aFile.close()

      for strLine in allLines:
         self.ui.plainTextEdit.appendPlainText(strLine.strip())

      self.__iniModelFromStringList(allLines)  
      self.ui.tableView.setEnabled(True)  #tableView可用
      self.ui.actAppend.setEnabled(True)  #更新Actions的enable属性
      self.ui.actInsert.setEnabled(True)
      self.ui.actDelete.setEnabled(True)
      self.ui.actSave.setEnabled(True)
      self.ui.actModelData.setEnabled(True)


   @pyqtSlot()  ##保存文件
   def on_actSave_triggered(self): 
##      curPath=QDir.currentPath() #获取当前路径
      curPath=os.getcwd()   #获取当前路径     
      filename,flt=QFileDialog.getSaveFileName(self,"保存文件",curPath,
                  "井斜数据文件(*.txt);;所有文件(*.*)")
      if (filename==""):
         return

      self.on_actModelData_triggered()  #更新数据到plainTextEdit

      aFile=open(filename,'w')  #以写方式打开
      aFile.write(self.ui.plainTextEdit.toPlainText())
      aFile.close()
        

   @pyqtSlot()   ##在最后添加一行
   def on_actAppend_triggered(self):   
      itemlist=[]    # QStandardItem 对象列表
      for i in range(self.__ColCount-1):  #不包括最后一列
         item=QStandardItem("0")
         itemlist.append(item)
         
      item=QStandardItem(self.__lastColumnTitle) #最后一列
      item.setCheckable(True)
      item.setFlags(self.__lastColumnFlags)
      itemlist.append(item)

      self.itemModel.appendRow(itemlist)  #添加一行
      curIndex=self.itemModel.index(self.itemModel.rowCount()-1,0)
      self.selectionModel.clearSelection()
      self.selectionModel.setCurrentIndex(curIndex,QItemSelectionModel.Select)

   @pyqtSlot()  ##插入一行
   def on_actInsert_triggered(self):   
      itemlist=[]  #  QStandardItem 对象列表
      for i in range(self.__ColCount-1): #不包括最后一列
         item=QStandardItem("0")
         itemlist.append(item)
         
      item=QStandardItem(self.__lastColumnTitle) #最后一列
      item.setFlags(self.__lastColumnFlags)
      item.setCheckable(True)
      item.setCheckState(Qt.Checked)
      itemlist.append(item)

      curIndex=self.selectionModel.currentIndex(); #获取当前选中项的模型索引
      self.itemModel.insertRow(curIndex.row(),itemlist);  #在当前行的前面插入一行
      self.selectionModel.clearSelection()
      self.selectionModel.setCurrentIndex(curIndex,QItemSelectionModel.Select)
        

   @pyqtSlot()   ##删除当前行
   def on_actDelete_triggered(self): 
      curIndex=self.selectionModel.currentIndex()  #获取当前选择单元格的模型索引
      self.itemModel.removeRow(curIndex.row())     #删除当前行

   @pyqtSlot()    ##左对齐
   def on_actAlignLeft_triggered(self):  
      self.__setCellAlignment(Qt.AlignLeft | Qt.AlignVCenter)

   @pyqtSlot()    ##中间对齐
   def on_actAlignCenter_triggered(self):  
      self.__setCellAlignment(Qt.AlignHCenter| Qt.AlignVCenter)
        
   @pyqtSlot()    ##右对齐
   def on_actAlignRight_triggered(self):   
      self.__setCellAlignment(Qt.AlignRight | Qt.AlignVCenter)

   @pyqtSlot(bool)   ##字体Bold
   def on_actFontBold_triggered(self,checked):
      if (not self.selectionModel.hasSelection()): #没有选择的项
         return
      selectedIndex=self.selectionModel.selectedIndexes()   #模型索引列表
      count=len(selectedIndex)
      for i in range(count):
         index=selectedIndex[i]     #获取其中的一个模型索引
         item=self.itemModel.itemFromIndex(index)    #获取一个单元格的项数据对象
         font=item.font()
         font.setBold(checked)
         item.setFont(font)

   @pyqtSlot()    ##模型数据显示到plainTextEdit里
   def on_actModelData_triggered(self):  
      self.ui.plainTextEdit.clear()
      lineStr=""
      for i in range(self.itemModel.columnCount()-1):  #表头,不含最后一列
         item=self.itemModel.horizontalHeaderItem(i)
         lineStr=lineStr+item.text()+"\t"
      item=self.itemModel.horizontalHeaderItem(self.__ColCount-1) #最后一列
      lineStr=lineStr+item.text()   #表头文字字符串
      self.ui.plainTextEdit.appendPlainText(lineStr)

      for i in range(self.itemModel.rowCount()):
         lineStr=""
         for j in range(self.itemModel.columnCount()-1): #不包括最后一列
            item=self.itemModel.item(i,j)
            lineStr=lineStr+item.text()+"\t"
         item=self.itemModel.item(i,self.__ColCount-1) #最后一列
         if (item.checkState()==Qt.Checked):
            lineStr=lineStr+"1"
         else:
            lineStr=lineStr+"0"
         self.ui.plainTextEdit.appendPlainText(lineStr)
        
    
##  =============自定义槽函数===============================        
   def do_curChanged(self,current,previous):
      if (current != None):        #当前模型索引有效
         text="当前单元格：%d行，%d列"%(current.row(),current.column())
         self.LabCellPos.setText(text)
         item=self.itemModel.itemFromIndex(current) #从模型索引获得Item
         self.LabCellText.setText("单元格内容："+item.text()) #显示item的文字内容

         font=item.font() #获取item的字体
         self.ui.actFontBold.setChecked(font.bold()) #更新actFontBold的check状态

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
    app = QApplication(sys.argv)    #创建GUI应用程序
    form=QmyMainWindow()            #创建窗体
    form.show()
    sys.exit(app.exec_())
