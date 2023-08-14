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

      self.ui.groupBox.setEnabled(False)
      self.ui.actSaveALL.setEnabled(False)
      self.ui.actReadALL.setEnabled(False)

      self.setWindowTitle("二进制文件流化读写")
      
      self.__testFileName=""   #测试用文件的文件名
      
##  ==============自定义功能函数============
   def __iniWrite(self):     ##初始化写文件操作
      self.fileDevice=QFile(self.__testFileName)    #创建文件对象
      if  not self.fileDevice.open(QIODevice.WriteOnly):
         del self.fileDevice    #删除对象
         return False

      self.fileStream=QDataStream(self.fileDevice)   #流对象

      self.fileStream.setVersion(QDataStream.Qt_5_12)   #设置版本号，写入和读取的版本号要兼容
      if self.ui.radio_BigEndian.isChecked():
         self.fileStream.setByteOrder(QDataStream.BigEndian)
      else:
         self.fileStream.setByteOrder(QDataStream.LittleEndian)

      ##必须要设置精度，float和double都按照这个精度
      precision=QDataStream.DoublePrecision
      if self.ui.radio_Single.isChecked(): 
         precision=QDataStream.SinglePrecision
      self.fileStream.setFloatingPointPrecision(precision)
      return True

   def __delFileStream(self): ##结束写文件操作
      self.fileDevice.close()
      del self.fileStream
      del self.fileDevice


   def __iniRead(self):  ##开始读文件操作
      if not QFile.exists(self.__testFileName):
         QMessageBox.critical(self,"错误","文件不存在")
         return False
      
      self.fileDevice=QFile(self.__testFileName)    #创建文件对象
      if  not self.fileDevice.open(QIODevice.ReadOnly):
         del self.fileDevice    #删除对象
         return False

      self.fileStream=QDataStream(self.fileDevice)
      self.fileStream.setVersion(QDataStream.Qt_5_12)   #设置流版本号，写入和读取的版本号要兼容

      if self.ui.radio_BigEndian.isChecked():
         self.fileStream.setByteOrder(QDataStream.BigEndian)
      else:
         self.fileStream.setByteOrder(QDataStream.LittleEndian)

      ##必须要设置精度，float和double都按照这个精度
      precision=QDataStream.DoublePrecision
      if self.ui.radio_Single.isChecked(): 
         precision=QDataStream.SinglePrecision
      self.fileStream.setFloatingPointPrecision(precision)

      return True
          
##  ==========由connectSlotsByName() 自动连接的槽函数==================
   
   @pyqtSlot()    ##选择测试用文件
   def on_btnFile_clicked(self):  
      curPath=QDir.currentPath()       #当前目录
      title="选择文件"                 #对话框标题
      filt="流数据文件(*.stream)"     #文件过滤器
      fileName,flt=QFileDialog.getSaveFileName(self,title,curPath,filt)
      if (fileName == ""):
         return

      self.__testFileName=fileName     #测试用文件
      self.ui.editFilename.setText(fileName)
      self.ui.groupBox.setEnabled(True)
      self.ui.actSaveALL.setEnabled(True)
      self.ui.actReadALL.setEnabled(True)


   @pyqtSlot()   ##写  int8
   def on_btnInt8_Write_clicked(self):  
      Value=self.ui.spin_Int8.value() #Python int
      if self.__iniWrite():
         try:
            self.fileStream.writeInt8(Value)
         except Exception as e:
            QMessageBox.critical(self, "writeInt8()出现错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  int8
   def on_btnInt8_Read_clicked(self):  
      if self.__iniRead():
         Value=self.fileStream.readInt8()
         self.ui.edit_Int8.setText("%d"%Value)
         self.__delFileStream()

   @pyqtSlot()   ##写  uint8
   def on_btnUInt8_Write_clicked(self):  
      Value=self.ui.spin_UInt8.value()
      if self.__iniWrite(): 
         try:
            self.fileStream.writeUInt8(Value)
         except Exception as e:
            QMessageBox.critical(self, "writeUInt8()出现错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  uint8
   def on_btnUInt8_Read_clicked(self):  
      if self.__iniRead():
         Value=self.fileStream.readUInt8()
         self.ui.edit_UInt8.setText("%d"%Value)
         self.__delFileStream()


   @pyqtSlot()    ##写int16
   def on_btnInt16_Write_clicked(self):  
      Value=self.ui.spin_Int16.value()       #Python的int
      if self.__iniWrite():
         try:
            self.fileStream.writeInt16(Value)   #以int16类型写入文件
         except Exception as e:
            QMessageBox.critical(self, "writeInt16()发生错误", str(e))
         finally:
            self.__delFileStream()
##            print("finally被执行")
          
   @pyqtSlot()    ##读 int16
   def on_btnInt16_Read_clicked(self):  
      if self.__iniRead():
         try:
            Value=self.fileStream.readInt16() 
            self.ui.edit_Int16.setText("%d"%Value)
         except Exception as e:
            QMessageBox.critical(self, "readInt16()发生错误", str(e))
         finally:
            self.__delFileStream()
##            print("finally被执行")
            
            
   @pyqtSlot()   ##写  uint16
   def on_btnUInt16_Write_clicked(self):  
      Value=self.ui.spin_UInt16.value()
      if self.__iniWrite():
         try:
            self.fileStream.writeUInt16(Value)
         except Exception as e:
            QMessageBox.critical(self, "writeUInt16()发生错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  uint16
   def on_btnUIn16_Read_clicked(self):  
      if self.__iniRead():
         Value=self.fileStream.readUInt16()
         self.ui.edit_UInt16.setText("%d"%Value)
         self.__delFileStream()

         
   @pyqtSlot()   ##写  int32
   def on_btnInt32_Write_clicked(self):  
      Value=self.ui.spin_Int32.value()
      if self.__iniWrite():
         try:
            self.fileStream.writeInt32(Value)
         except Exception as e:
            QMessageBox.critical(self, "writeInt32()发生错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  int32
   def on_btnInt32_Read_clicked(self):  
      if self.__iniRead():
         Value=self.fileStream.readInt32()
         self.ui.edit_Int32.setText("%d"%Value)
         self.__delFileStream()


   @pyqtSlot()   ##写  int64
   def on_btnInt64_Write_clicked(self):  
      Value=self.ui.spin_Int64.value()
      if self.__iniWrite():
         try:
            self.fileStream.writeInt64(Value)
         except Exception as e:
            QMessageBox.critical(self, "writeInt64()发生错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  int64
   def on_btnInt64_Read_clicked(self):  
      if self.__iniRead():
         Value=self.fileStream.readInt64()
         self.ui.edit_Int64.setText("%d"%Value)
         self.__delFileStream()


   @pyqtSlot()   ##写  int
   def on_btnInt_Write_clicked(self):  
      Value=self.ui.spin_Int.value()
      if self.__iniWrite():
         try:
            self.fileStream.writeInt(Value)
         except Exception as e:
            QMessageBox.critical(self, "writeInt()发生错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  int
   def on_btnInt_Read_clicked(self):  
      if self.__iniRead():
         Value=self.fileStream.readInt()
         self.ui.edit_Int.setText("%d"%Value)
         self.__delFileStream()


   @pyqtSlot()   ##写  bool
   def on_btnBool_Write_clicked(self):  
      Value=self.ui.chkBox_In.isChecked() #bool型
      if self.__iniWrite(): 
         self.fileStream.writeBool(Value)
         self.__delFileStream()
          
   @pyqtSlot()    ##读  bool
   def on_btnBool_Read_clicked(self):  
      if self.__iniRead():
         Value=self.fileStream.readBool() #bool型
         self.ui.chkBox_Out.setChecked(Value)
         self.__delFileStream()


   @pyqtSlot()   ##写  float
   def on_btnFloat_Write_clicked(self):  
      Value=self.ui.spin_Float.value()
      if self.__iniWrite():
         try:
            self.fileStream.writeFloat(Value)
         except Exception as e:
            QMessageBox.critical(self, "writeFloat()发生错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  float
   def on_btnFloat_Read_clicked(self):  
      if self.__iniRead():
         try:
            Value=self.fileStream.readFloat()
            self.ui.edit_Float.setText("%.4f"%Value)
         except Exception as e:
            QMessageBox.critical(self, "writeFloat()发生错误", str(e))
         finally:
            self.__delFileStream()


   @pyqtSlot()   ##写  double
   def on_btnDouble_Write_clicked(self):  
      Value=self.ui.spin_Double.value()
      if self.__iniWrite():
         try:
            self.fileStream.writeDouble(Value)
         except Exception as e:
            QMessageBox.critical(self, "writeDouble()发生错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  double
   def on_btnDouble_Read_clicked(self):  
      if self.__iniRead():
         try:
            Value=self.fileStream.readDouble()
            self.ui.edit_Double.setText("%.4f"%Value)
         except Exception as e:
            QMessageBox.critical(self, "readDouble()发生错误", str(e))
         finally:
            self.__delFileStream()


   @pyqtSlot()    ##写 QString,与Python的str兼容
   def on_btnQStr_Write_clicked(self):  
      Value=self.ui.editQStr_In.text()
      if self.__iniWrite():
         try:
            self.fileStream.writeQString(Value)
         except Exception as e:
            QMessageBox.critical(self, "writeQString()发生错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读 QString,与Python的str兼容
   def on_btnQStr_Read_clicked(self):  
      if self.__iniRead():
         try:
            Value=self.fileStream.readQString()
            self.ui.editQStr_Out.setText(Value)
         except Exception as e:
            QMessageBox.critical(self, "readQString()发生错误", str(e))
         finally:
            self.__delFileStream()

   @pyqtSlot()    ##写 String
   def on_btnStr_Write_clicked(self):  
      strV=self.ui.editStr_In.text()   #str类型
      if self.__iniWrite():
         try:
            bts=bytes(strV,encoding="utf-8")  #转换为bytes类型
            self.fileStream.writeString(bts)
         except Exception as e:
            QMessageBox.critical(self, "写入时发生错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读 String
   def on_btnStr_Read_clicked(self):  
      if self.__iniRead():
         try:
            Value=self.fileStream.readString()  #bytes类型
            strV=Value.decode("utf-8")          #从bytes类型解码为字符串，编码utf-8
            self.ui.editStr_Out.setText(strV)
         except Exception as e:
            QMessageBox.critical(self, "读取时发生错误", str(e))
         finally:
            self.__delFileStream()

##===字体的写入与读取
   @pyqtSlot()   ##选择字体
   def on_btnFont_In_clicked(self):  
      font=self.ui.btnFont_In.font()
      font,OK=QFontDialog.getFont(font,self) #选择字体
      if OK:
         self.ui.btnFont_In.setFont(font)

   @pyqtSlot()   ##写  QVariant, QFont
   def on_btnFont_Write_clicked(self):  
      font=self.ui.btnFont_In.font()    #QFont类型
      if self.__iniWrite(): 
         self.fileStream.writeQVariant(font)  #QFont类型
         self.__delFileStream()
          
   @pyqtSlot()    ##读  QVariant, QFont
   def on_btnFont_Read_clicked(self):  
      if self.__iniRead():
         try:
            font=self.fileStream.readQVariant()   #QFont类型
            self.ui.editFont_Out.setFont(font)
         except Exception as e:
            QMessageBox.critical(self, "读取时发生错误", str(e))
         finally:
            self.__delFileStream()


##===颜色的写入与读取
   @pyqtSlot()   ##选择颜色
   def on_btnColor_In_clicked(self):  
      plet=self.ui.btnColor_In.palette()  #QPalette
      color=plet.buttonText().color()     #QColor
      color= QColorDialog.getColor(color,self)
      if color.isValid():
         plet.setColor(QPalette.ButtonText,color)
         self.ui.btnColor_In.setPalette(plet)


   @pyqtSlot()   ##写  QVariant, QColor
   def on_btnColor_Write_clicked(self):  
      plet=self.ui.btnColor_In.palette()  
      color=plet.buttonText().color()     #QColor
      if self.__iniWrite(): 
         self.fileStream.writeQVariant(color)  #QColor
         self.__delFileStream()
          
   @pyqtSlot()    ##读  QVariant, QColor
   def on_btnColor_Read_clicked(self):  
      if self.__iniRead():
         try:
            color=self.fileStream.readQVariant()   #读取为QColor类型
            plet=self.ui.editColor_Out.palette() 
            plet.setColor(QPalette.Text,color)
            self.ui.editColor_Out.setPalette(plet)
         except Exception as e:
            QMessageBox.critical(self, "读取时发生错误", str(e))
         finally:
            self.__delFileStream()

   @pyqtSlot()    ##读出编辑框全清空
   def on_actClearOutput_triggered(self):  
      self.ui.edit_Int8.clear()
      self.ui.edit_UInt8.clear()
      self.ui.edit_Int16.clear()
      self.ui.edit_UInt16.clear()
      self.ui.edit_Int32.clear()
      self.ui.edit_Int64.clear()
      self.ui.edit_Int.clear()

      self.ui.edit_Float.clear()
      self.ui.edit_Double.clear()

      font=self.font()
      self.ui.editFont_Out.setFont(font)

      plet=self.palette()
      self.ui.editColor_Out.setPalette(plet)

   @pyqtSlot()    ##连续写入文件
   def on_actSaveALL_triggered(self):  
      if not self.__iniWrite():
         QMessageBox.critical(self,"错误","为写入打开文件时出错")
         return
   #数据写入部分
      Value=self.ui.spin_Int8.value()
      self.fileStream.writeInt8(Value)    #int8

      Value=self.ui.spin_UInt8.value()
      self.fileStream.writeUInt8(Value)   #uint8

      Value=self.ui.spin_Int16.value()
      self.fileStream.writeInt16(Value)   #int16

      Value=self.ui.spin_UInt16.value()
      self.fileStream.writeUInt16(Value)  #uint16

      Value=self.ui.spin_Int32.value()
      self.fileStream.writeInt32(Value)   #int32

      Value=self.ui.spin_Int64.value()
      self.fileStream.writeInt64(Value)   #int64

      Value=self.ui.spin_Int.value()
      self.fileStream.writeInt(Value)     #int

      Value=self.ui.chkBox_In.isChecked()
      self.fileStream.writeBool(Value)    #bool

      Value=self.ui.spin_Float.value()
      self.fileStream.writeFloat(Value)   #float

      Value=self.ui.spin_Double.value() 
      self.fileStream.writeDouble(Value)  #double

      str_Value=self.ui.editQStr_In.text()
      self.fileStream.writeQString(str_Value)   #QString
      
      str_Value=self.ui.editStr_In.text()       #str类型
      bts=bytes(str_Value,encoding="utf-8")     #转换为bytes类型
      self.fileStream.writeString(bts)    

      font=self.ui.btnFont_In.font()
      self.fileStream.writeQVariant(font)    #QFont

      plet=self.ui.btnColor_In.palette()
      color=plet.buttonText().color() 
      self.fileStream.writeQVariant(color)   #QColor

   #数据写入完成
      self.__delFileStream()
      QMessageBox.information(self,"消息","数据连续写入完成.")


   @pyqtSlot()    ##连续读取文件
   def on_actReadALL_triggered(self):  
      if not self.__iniRead():
         QMessageBox.critical(self,"错误","为读取打开文件时出错")
         return
      
   #数据读取部分
      Value=self.fileStream.readInt8()    #int8
      self.ui.edit_Int8.setText("%d"%Value)

      Value=self.fileStream.readUInt8()   #uint8
      self.ui.edit_UInt8.setText("%d"%Value)

      Value=self.fileStream.readInt16()   #int16
      self.ui.edit_Int16.setText("%d"%Value)

      Value=self.fileStream.readUInt16()  #uint16
      self.ui.edit_UInt16.setText("%d"%Value)

      Value=self.fileStream.readInt32()   #int32
      self.ui.edit_Int32.setText("%d"%Value)

      Value=self.fileStream.readInt64()   #int64
      self.ui.edit_Int64.setText("%d"%Value)

      Value=self.fileStream.readInt()     #int
      self.ui.edit_Int.setText("%d"%Value)

      Value=self.fileStream.readBool()    #bool
      self.ui.chkBox_Out.setChecked(Value)

      Value=self.fileStream.readFloat()   #float
      self.ui.edit_Float.setText("%.4f"%Value)

      Value=self.fileStream.readDouble()  #double
      self.ui.edit_Double.setText("%.4f"%Value)

      str_Value=self.fileStream.readQString()  #str
      self.ui.editQStr_Out.setText(str_Value)

      byteStr=self.fileStream.readString()   #bytes
      str_Value=byteStr.decode("utf-8")      #从bytes类型解码为字符串
      self.ui.editStr_Out.setText(str_Value)

      font=self.fileStream.readQVariant()    #QFont
      self.ui.editFont_Out.setFont(font)

      color=self.fileStream.readQVariant()   #QColor
      plet=self.ui.editColor_Out.palette() 
      plet.setColor(QPalette.Text,color)
      self.ui.editColor_Out.setPalette(plet)
      
   #数据写入完成
      self.__delFileStream()
      QMessageBox.information(self,"消息","数据连续读取完成.")
      
##  =============自定义槽函数===============================        
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
