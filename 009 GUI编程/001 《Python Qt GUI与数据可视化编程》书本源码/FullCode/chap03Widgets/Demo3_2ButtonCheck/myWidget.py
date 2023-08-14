import sys

from PyQt5.QtWidgets import  QApplication, QWidget

from PyQt5.QtCore import  pyqtSlot, Qt

from PyQt5.QtGui import  QFont

from ui_Widget import Ui_Widget


class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)      #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()           #创建UI对象
      self.ui.setupUi(self)         #构造UI界面

        
##  =======由connectSlotsByName() 自动连接的槽函数=======================        
   def on_btnAlign_Left_clicked(self):     ##“居左”按钮
      self.ui.editInput.setAlignment(Qt.AlignLeft)

   def on_btnAlign_Center_clicked(self):   ##“居中”按钮
      self.ui.editInput.setAlignment(Qt.AlignCenter)
    
   def on_btnAlign_Right_clicked(self):    ##“居右”按钮
      self.ui.editInput.setAlignment(Qt.AlignRight)

   @pyqtSlot(bool)  ##“粗体”按钮
   def on_btnFont_Bold_clicked(self,checked): 
      font=self.ui.editInput.font()
      font.setBold(checked)
      self.ui.editInput.setFont(font)
        
   @pyqtSlot(bool)   ##“斜体”按钮
   def on_btnFont_Italic_clicked(self,checked):
      font=self.ui.editInput.font()
      font.setItalic(checked)
      self.ui.editInput.setFont(font)

   @pyqtSlot(bool)  ##“下划线”按钮
   def on_btnFont_UnderLine_clicked(self,checked): 
      font=self.ui.editInput.font()
      font.setUnderline(checked)
      self.ui.editInput.setFont(font)

   @pyqtSlot(bool)  ##"Readonly" 复选框
   def on_chkBox_Readonly_clicked(self,checked):    
      self.ui.editInput.setReadOnly(checked)

   @pyqtSlot(bool)   ##"Enabled"复选框
   def on_chkbox_Enable_clicked(self,checked):    
      self.ui.editInput.setEnabled(checked)
        
   @pyqtSlot(bool)   ##"ClearButtonEnabled"复选框
   def on_chkBox_ClearButton_clicked(self,checked):  
      self.ui.editInput.setClearButtonEnabled(checked)

##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ===============================        
if  __name__ == "__main__":         #用于当前窗体测试
    app = QApplication(sys.argv)    #创建GUI应用程序
    form=QmyWidget()                #创建窗体
    form.show()
    sys.exit(app.exec_())
