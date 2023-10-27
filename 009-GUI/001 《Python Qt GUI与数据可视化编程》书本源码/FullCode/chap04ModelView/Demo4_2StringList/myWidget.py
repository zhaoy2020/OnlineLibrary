import sys

from PyQt5.QtWidgets import  QApplication, QWidget,QAbstractItemView

from PyQt5.QtCore import  pyqtSlot, QStringListModel, Qt, QModelIndex

##from PyQt5.QtWidgets import  QAbstractItemView

##from PyQt5.QtGui import  

from ui_Widget import Ui_Widget


class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.__provinces=["北京","上海","天津","河北",
                       "山东","四川","重庆","广东","河南"]
      self.model=QStringListModel(self)
      self.model.setStringList(self.__provinces)
      self.ui.listView.setModel(self.model)
##        trig=(QAbstractItemView.DoubleClicked |QAbstractItemView.SelectedClicked)
##        self.ui.listView.setEditTriggers(trig)

      self.ui.listView.setEditTriggers(QAbstractItemView.DoubleClicked
              | QAbstractItemView.SelectedClicked)

##  =================自定义功能函数=================================
        
##  ==========由connectSlotsByName() 自动连接的槽函数===============        
   @pyqtSlot()    ##重设模型数据内容
   def on_btnList_Reset_clicked(self):   
      self.model.setStringList(self.__provinces); 

   @pyqtSlot()  ##添加项
   def on_btnList_Append_clicked(self):   
      lastRow=self.model.rowCount()
      self.model.insertRow(lastRow)   #在尾部插入一空行
      index=self.model.index(lastRow,0)  #获取最后一行的ModelIndex
      self.model.setData(index,"new item",Qt.DisplayRole) #设置显示文字
      self.ui.listView.setCurrentIndex(index)  #设置当前选中的行

   @pyqtSlot()    ##插入项
   def on_btnList_Insert_clicked(self):   
      index=self.ui.listView.currentIndex()   #当前 modelIndex
      self.model.insertRow(index.row())
      self.model.setData(index,"inserted item", Qt.DisplayRole)  #设置显示文字
      ##        self.model.setData(index,Qt.AlignRight, Qt.TextAlignmentRole) #设置对齐方式，不起作用
      self.ui.listView.setCurrentIndex(index) #设置当前选中的行


   @pyqtSlot()  ##删除当前项
   def on_btnList_Delete_clicked(self):   
      index=self.ui.listView.currentIndex() #获取当前 modelIndex
      self.model.removeRow(index.row()); #删除当前行
        
   @pyqtSlot()    ##清空列表
   def on_btnList_Clear_clicked(self): 
      count=self.model.rowCount()
      self.model.removeRows(0,count)

   @pyqtSlot()    ##清空文本
   def on_btnText_Clear_clicked(self):
      self.ui.plainTextEdit.clear()

   @pyqtSlot()    ##显示数据模型的内容
   def on_btnText_Display_clicked(self):  
      strList=self.model.stringList()  #列表类型
      self.ui.plainTextEdit.clear()
      for strLine in strList:
         self.ui.plainTextEdit.appendPlainText(strLine)

   def on_listView_clicked(self,index):
      self.ui.LabInfo.setText("当前项index: row=%d, column=%d"
                             %(index.row(),index.column()))
        
    
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":      
   app = QApplication(sys.argv) 
   form=QmyWidget()           
   form.show()
   sys.exit(app.exec_())
