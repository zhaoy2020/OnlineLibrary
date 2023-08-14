import sys

from PyQt5.QtWidgets import  QApplication, QWidget

from PyQt5.QtCore import  pyqtSlot

from PyQt5.QtGui import  QIcon

from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面
      
        
##  ==========由connectSlotsByName() 自动连接的槽函数====================        
   def on_btnIniItems_clicked(self):   ##“初始化列表”按钮
      pass

   def on_btnClearItems_clicked(self):    ##“清除列表”按钮
      pass

   @pyqtSlot(bool)   ##“可编辑” CheckBox
   def on_chkBoxEditable_clicked(self,checked):  
      pass

   @pyqtSlot(str)    ##“简单的ComboBox”的当前项变化
   def on_comboBox_currentIndexChanged(self,curText):
      pass

   def on_btnIni2_clicked(self):   ##有用户数据的comboBox2的初始化
      pass

   @pyqtSlot(str)    ##当前项变化
   def on_comboBox2_currentIndexChanged(self,curText): 
      pass
        
##  =========自定义槽函数===================================        

   
##  ===========窗体测试程序 ================================        
if  __name__ == "__main__":         
   app = QApplication(sys.argv)   
   form=QmyWidget()      
   form.show()
   sys.exit(app.exec_())
