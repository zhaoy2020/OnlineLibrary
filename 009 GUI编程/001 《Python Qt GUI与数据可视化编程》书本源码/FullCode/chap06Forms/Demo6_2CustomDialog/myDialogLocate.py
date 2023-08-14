import sys

from PyQt5.QtWidgets import  QApplication, QDialog

##from PyQt5.QtWidgets import  QDialog

##from PyQt5.QtCore import  QStringListModel

from PyQt5.QtCore import  pyqtSlot, pyqtSignal, Qt, QEvent

##from PyQt5.QtWidgets import  

##from PyQt5.QtGui import QCloseEvent, QShowEvent

from ui_QWDialogLocate import Ui_QWDialogLocate

class QmyDialogLocate(QDialog):
   changeActionEnable = pyqtSignal(bool)        #用于设置主窗口的Action的Enabled
   changeCellText = pyqtSignal(int, int, str)   #用于设置主窗口的单元格的内容

   def __init__(self, parent=None):
      super().__init__(parent)         #调用父类构造函数，创建窗体
      self.ui=Ui_QWDialogLocate()      #创建UI对象
      self.ui.setupUi(self)            #构造UI界面

      self.setAttribute(Qt.WA_DeleteOnClose)       #对话框关闭时自动删除
      self.setWindowFlag(Qt.WindowStaysOnTopHint)  # StayOnTop显示

   def __del__(self):
      print("QmyDialogLocate 对象被删除了")
      
        
##  ==============自定义功能函数============
   def setSpinRange(self,rowCount, colCount):   #设置SpinBox最大值
      self.ui.spinRow.setMaximum(rowCount-1)
      self.ui.spinColumn.setMaximum(colCount-1)

##  ==============事件处理函数===================
   def showEvent(self,event):    ##对话框显示事件
      self.changeActionEnable.emit(False)
      super().showEvent(event)

   def closeEvent(self,event):   ##对话框关闭事件
      self.changeActionEnable.emit(True)
      super().closeEvent(event)
        
##  ========由connectSlotsByName() 自动连接的槽函数========        
   @pyqtSlot()    ##“设定文字”按钮
   def on_btnSetText_clicked(self):
      row=self.ui.spinRow.value()      #行号
      col=self.ui.spinColumn.value()   #列号
      text=self.ui.editCellText.text() #文字

      self.changeCellText.emit(row,col,text)    #发射信号

      if (self.ui.chkBoxRow.isChecked()):       #行增
         self.ui.spinRow.setValue(1+self.ui.spinRow.value())

      if (self.ui.chkBoxColumn.isChecked()):    #列增
         self.ui.spinColumn.setValue(1+self.ui.spinColumn.value())
    
        
##  ===================自定义槽函数=========================        
   @pyqtSlot(int,int)  ##用于与主窗口的cellIndexChanged()信号关联
   def do_setSpinValue(self,rowNo, colNo):
      self.ui.spinRow.setValue(rowNo)
      self.ui.spinColumn.setValue(colNo)
        
   
##  ============窗体测试程序 ==============================
if  __name__ == "__main__":       
   app = QApplication(sys.argv)  
   form=QmyDialogLocate()         
   form.show()
   sys.exit(app.exec_())
