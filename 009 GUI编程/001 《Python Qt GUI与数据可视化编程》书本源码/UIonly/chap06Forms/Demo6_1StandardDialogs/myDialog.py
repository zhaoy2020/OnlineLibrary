import sys

from PyQt5.QtWidgets import  (QApplication, QDialog,QFileDialog,
                        QColorDialog,QFontDialog,QProgressDialog,
                        QLineEdit,QInputDialog,QMessageBox)

from PyQt5.QtCore import  Qt, pyqtSlot, QDir,QTime

from PyQt5.QtGui import QPalette, QColor, QFont

from ui_Dialog import Ui_Dialog


class QmyDialog(QDialog): 
   def __init__(self, parent=None):
      super().__init__(parent)      #调用父类构造函数，创建窗体
      self.ui=Ui_Dialog()           #创建UI对象
      self.ui.setupUi(self)         #构造UI界面

##  ==============自定义功能函数============

        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##"打开一个文件"
   def on_btnOpen_clicked(self):
      pass

            
   @pyqtSlot()    ##"打开多个文件"
   def on_btnOpenMulti_clicked(self):   
      pass


   @pyqtSlot()    ##“选择已有目录 ”
   def on_btnSelDir_clicked(self): 
      pass


   @pyqtSlot()    ##“保存文件”
   def on_btnSave_clicked(self):   
      pass

        
   @pyqtSlot()    ##"选择颜色"
   def on_btnColor_clicked(self): 
      pass

        
   @pyqtSlot()    ##"选择字体"
   def on_btnFont_clicked(self):   
      pass


   @pyqtSlot()    ##"进度对话框"
   def on_btnProgress_clicked(self): 
      pass


   @pyqtSlot()    ##“输入字符串”
   def on_btnInputString_clicked(self):    
      pass
        
        
   @pyqtSlot()    ##“输入整数”
   def on_btnInputInt_clicked(self):
      pass


   @pyqtSlot()    ##“输入浮点数”
   def on_btnInputFloat_clicked(self):
      pass


   @pyqtSlot()    ##"条目选择输入"
   def on_btnInputItem_clicked(self):
      pass

        
   @pyqtSlot()    ##"question"
   def on_btnMsgQuestion_clicked(self):
      pass


   @pyqtSlot()    ##"information"
   def on_btnMsgInformation_clicked(self):
      pass


   @pyqtSlot()    ##"warning"
   def on_btnMsgWarning_clicked(self):
      pass

   @pyqtSlot()    ##"critical"
   def on_btnMsgCritical_clicked(self):
      pass


   @pyqtSlot()    ##"about"
   def on_btnMsgAbout_clicked(self):
      pass

   @pyqtSlot()    ##"About Qt"
   def on_btnMsgAboutQt_clicked(self):
      pass

    
##  =============自定义槽函数===============================        
   def do_progress_canceled(self):     ##取消进度对话框
      pass
   
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":      
   app = QApplication(sys.argv) 
   form=QmyDialog()           
   form.show()
   sys.exit(app.exec_())
