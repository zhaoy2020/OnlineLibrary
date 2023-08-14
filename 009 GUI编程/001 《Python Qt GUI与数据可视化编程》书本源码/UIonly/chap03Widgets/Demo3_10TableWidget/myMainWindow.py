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
      pass



##  ==============自定义功能函数============
   def __createItemsARow(self,rowNo,name,sex,birth,nation,isParty,score): ##创建一行的items
      pass
   
       
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##“设置表头”按钮
   def on_btnSetHeader_clicked(self): 
      pass


   @pyqtSlot()    ##设置行数
   def on_btnSetRows_clicked(self):  
      pass
    

   @pyqtSlot()    ##初始化表格数据
   def on_btnIniData_clicked(self):  
      pass
            

   @pyqtSlot()    ##插入行
   def on_btnInsertRow_clicked(self):      
      pass

        
   @pyqtSlot()    ##添加行
   def on_btnAppendRow_clicked(self):  
      pass
        
   @pyqtSlot()    ##删除当前行
   def on_btnDelCurRow_clicked(self):  
      pass

        
   @pyqtSlot()    ##清空表格内容
   def on_btnClearContents_clicked(self): 
      pass


   @pyqtSlot()    ##自动行高
   def on_btnAutoHeight_clicked(self): 
      pass

        
   @pyqtSlot()    ##自动列宽
   def on_btnAutoWidth_clicked(self): 
      pass


   @pyqtSlot(bool)  ##表格可编辑
   def on_chkBoxEditable_clicked(self,checked):  
      pass


   @pyqtSlot(bool)      ##交替行颜色
   def on_chkBoxRowColor_clicked(self,checked):    
      pass


   @pyqtSlot(bool)   ##是否显示水平表头
   def on_chkBoxHeaderH_clicked(self,checked):  
      pass


   @pyqtSlot(bool) #是否显示垂直表头
   def on_chkBoxHeaderV_clicked(self,checked):    
      pass


   @pyqtSlot()    ##选择行为：行选择
   def on_radioSelectRow_clicked(self): 
      pass

        
   @pyqtSlot()    ##选择行为：单元格选择
   def on_radioSelectItem_clicked(self):
      pass


   @pyqtSlot()    ##读取表格到文本
   def on_btnReadToText_clicked(self): 
      pass


   @pyqtSlot(int,int,int,int)   ##当前单元格发生变化
   def on_tableInfo_currentCellChanged(self,currentRow,currentColumn,previousRow,previousColumn):
      pass

        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================        
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
