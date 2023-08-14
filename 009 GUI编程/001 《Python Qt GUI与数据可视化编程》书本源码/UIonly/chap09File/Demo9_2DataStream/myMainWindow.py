import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow,QColorDialog,
                           QFileDialog, QMessageBox,QFontDialog)

from PyQt5.QtCore import  Qt, pyqtSlot,QDataStream,QFile,QDir,QIODevice


from PyQt5.QtGui import QPalette,QColor

from ui_MainWindow import Ui_MainWindow


class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      self.setWindowTitle("二进制文件流化读写")
      pass

      
##  ==============自定义功能函数============
   def __iniWrite(self):     ##初始化写文件操作
      pass


   def __delFileStream(self): ##结束写文件操作
      pass


   def __iniRead(self):  ##开始读文件操作
      pass

          
##  ==========由connectSlotsByName() 自动连接的槽函数==================
   
   @pyqtSlot()    ##选择测试用文件
   def on_btnFile_clicked(self):  
      pass


   @pyqtSlot()   ##写  int8
   def on_btnInt8_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读  int8
   def on_btnInt8_Read_clicked(self):  
      pass


   @pyqtSlot()   ##写  uint8
   def on_btnUInt8_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读  uint8
   def on_btnUInt8_Read_clicked(self):  
      pass


   @pyqtSlot()    ##写int16
   def on_btnInt16_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读 int16
   def on_btnInt16_Read_clicked(self):  
      pass

            
   @pyqtSlot()   ##写  uint16
   def on_btnUInt16_Write_clicked(self):  
      pass
          
   @pyqtSlot()    ##读  uint16
   def on_btnUIn16_Read_clicked(self):  
      pass

         
   @pyqtSlot()   ##写  int32
   def on_btnInt32_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读  int32
   def on_btnInt32_Read_clicked(self):  
      pass


   @pyqtSlot()   ##写  int64
   def on_btnInt64_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读  int64
   def on_btnInt64_Read_clicked(self):  
      pass


   @pyqtSlot()   ##写  int
   def on_btnInt_Write_clicked(self):  
      pass

         
   @pyqtSlot()    ##读  int
   def on_btnInt_Read_clicked(self):  
      pass


   @pyqtSlot()   ##写  bool
   def on_btnBool_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读  bool
   def on_btnBool_Read_clicked(self):  
      pass


   @pyqtSlot()   ##写  float
   def on_btnFloat_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读  float
   def on_btnFloat_Read_clicked(self):  
      pass


   @pyqtSlot()   ##写  double
   def on_btnDouble_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读  double
   def on_btnDouble_Read_clicked(self):  
      pass


   @pyqtSlot()    ##写 QString,与Python的str兼容
   def on_btnQStr_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读 QString,与Python的str兼容
   def on_btnQStr_Read_clicked(self):  
      pass


   @pyqtSlot()    ##写 String
   def on_btnStr_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读 String
   def on_btnStr_Read_clicked(self):  
      pass


##===字体的写入与读取
   @pyqtSlot()   ##选择字体
   def on_btnFont_In_clicked(self):  
      pass


   @pyqtSlot()   ##写  QVariant, QFont
   def on_btnFont_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读  QVariant, QFont
   def on_btnFont_Read_clicked(self):  
      pass


##===颜色的写入与读取
   @pyqtSlot()   ##选择颜色
   def on_btnColor_In_clicked(self):  
      pass


   @pyqtSlot()   ##写  QVariant, QColor
   def on_btnColor_Write_clicked(self):  
      pass

          
   @pyqtSlot()    ##读  QVariant, QColor
   def on_btnColor_Read_clicked(self):  
      pass


   @pyqtSlot()    ##读出编辑框全清空
   def on_actClearOutput_triggered(self):  
      pass


   @pyqtSlot()    ##连续写入文件
   def on_actSaveALL_triggered(self):  
      pass


   @pyqtSlot()    ##连续读取文件
   def on_actReadALL_triggered(self):  
      pass

      
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
