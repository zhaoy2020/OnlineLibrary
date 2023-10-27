import sys,struct

from PyQt5.QtWidgets import  QApplication, QMainWindow

from PyQt5.QtWidgets import  QFileDialog, QMessageBox

from PyQt5.QtCore import  Qt, pyqtSlot,QDataStream,QFile,QDir,QIODevice

from PyQt5.QtGui import QPalette,QColor

from ui_MainWindow import Ui_MainWindow


class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.ui.groupBox.setEnabled(False)
      self.ui.actSaveALL.setEnabled(False)
      self.ui.actReadALL.setEnabled(False)
      
      self.__testFileName=""
      self.__typeSize={"char":1, "bool":1,   "int8":1,   "uint8":1,
                       "int16":2,"uint16":2, "int32":4,  "uint32":4,
                       "int64":8,"uint64":8, "int":4,    "uint":4,
                       "float":4,"single":4, "double":8}
      
##  ==============自定义功能函数============
   def __iniWrite(self):    ##开始写文件操作
      self.fileDevice=QFile(self.__testFileName)  #创建文件对象
      if  not self.fileDevice.open(QIODevice.WriteOnly):
         del self.fileDevice  #删除对象
         return False

      self.fileStream=QDataStream(self.fileDevice)

      self.fileStream.setVersion(QDataStream.Qt_5_12)    #设置流版本号，写入和读取的版本号要兼容
      if self.ui.radio_BigEndian.isChecked():
         self.fileStream.setByteOrder(QDataStream.BigEndian)
      else:
         self.fileStream.setByteOrder(QDataStream.LittleEndian)

      if self.ui.radio_Single.isChecked():   #必须要设置，float和double都按照这个精度
         self.fileStream.setFloatingPointPrecision(QDataStream.SinglePrecision)
      else:
         self.fileStream.setFloatingPointPrecision(QDataStream.DoublePrecision)
      return True

   def __delFileStream(self):   ##结束写文件操作
      self.fileDevice.close()
      del self.fileStream
      del self.fileDevice


   def __iniRead(self):    ##开始写文件操作
      if not QFile.exists(self.__testFileName):
         QMessageBox.critical(self,"错误","文件不存在")
         return False
      
      self.fileDevice=QFile(self.__testFileName)   #创建文件对象
      if  not self.fileDevice.open(QIODevice.ReadOnly):
         del self.fileDevice    #删除对象
         return False

      self.fileStream=QDataStream(self.fileDevice)

      self.fileStream.setVersion(QDataStream.Qt_5_12)    #设置版本号，写入和读取的版本号要兼容
      if self.ui.radio_BigEndian.isChecked():
         self.fileStream.setByteOrder(QDataStream.BigEndian)
      else:
         self.fileStream.setByteOrder(QDataStream.LittleEndian)

      if self.ui.radio_Single.isChecked():   #必须要设置，float和double都按照这个精度
         self.fileStream.setFloatingPointPrecision(QDataStream.SinglePrecision)
      else:
         self.fileStream.setFloatingPointPrecision(QDataStream.DoublePrecision)
      return True
          
##  ==========由connectSlotsByName() 自动连接的槽函数==================
   @pyqtSlot()
   def on_btnFile_clicked(self):  
      curPath=QDir.currentPath() #获取系统当前目录
      title="选择文件"         #对话框标题
      filt="原始数据文件(*.raw)" #文件过滤器
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
      Value=self.ui.spin_Int8.value()   # Python的int类型
      if self.__iniWrite():
         try:
            bts=struct.pack('b',Value)     # 'b'=signed char=int8
## writeRawData(self, bytes) -> int            
            self.fileStream.writeRawData(bts)  
         except Exception as e:
            print(e)
            QMessageBox.critical(self, "写int8过程出现错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  int8
   def on_btnInt8_Read_clicked(self):  
      if self.__iniRead():
##readRawData(self, int) -> bytes
         try:
            bts=self.fileStream.readRawData(self.__typeSize["int8"])
## unpack(fmt, buffer) -> (v1, v2, ...)
            Value,=struct.unpack('b',bts)  # 'b'=signed char=int8
            self.ui.edit_Int8.setText("%d"%Value)
         except Exception as e:
            QMessageBox.critical(self, "读int8过程出现错误", str(e))
         finally:
            self.__delFileStream()

   @pyqtSlot()   ##写  uint8
   def on_btnUInt8_Write_clicked(self):  
      Value=self.ui.spin_UInt8.value()
      if self.__iniWrite():
         try:
            bts=struct.pack('B',Value)   # 'B'=unsigned char=uint8
            self.fileStream.writeRawData(bts)
         except Exception as e:
            QMessageBox.critical(self, "写uint8过程出现错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  uint8
   def on_btnUInt8_Read_clicked(self):  
      if self.__iniRead():
         try:
            bts=self.fileStream.readRawData(self.__typeSize["uint8"])
            Value,=struct.unpack('B',bts)    # 'B'=unsigned char=uint8
            self.ui.edit_UInt8.setText("%d"%Value)
         except Exception as e:
            QMessageBox.critical(self, "读uint8过程出现错误", str(e))
         finally:
            self.__delFileStream()

   @pyqtSlot()   ##写  int16
   def on_btnInt16_Write_clicked(self):  
      Value=self.ui.spin_Int16.value()
      if self.__iniWrite():
         try:
            bts=struct.pack('h',Value)    # 'h'=short, 2字节,=int16
            self.fileStream.writeRawData(bts)
         except Exception as e:
            QMessageBox.critical(self, "写int16过程出现错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  int16
   def on_btnInt16_Read_clicked(self):  
      if self.__iniRead():
         try:
            bts=self.fileStream.readRawData(self.__typeSize["int16"])
            Value,=struct.unpack('h',bts)     # 'h'=short, 2字节,=int16
            self.ui.edit_Int16.setText("%d"%Value)
         except Exception as e:
            QMessageBox.critical(self, "读int16过程出现错误", str(e))
         finally:
            self.__delFileStream()
            
   @pyqtSlot()    ##写  uint16
   def on_btnUInt16_Write_clicked(self):  
      Value=self.ui.spin_UInt16.value()
      if self.__iniWrite():
         try:
            bts=struct.pack('H',Value)    # 'H'=unsigned short, 2字节=uint16
            self.fileStream.writeRawData(bts)
         except Exception as e:
            QMessageBox.critical(self, "写uint16过程出现错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  uint16
   def on_btnUIn16_Read_clicked(self):  
      if self.__iniRead():
         try:
            bts=self.fileStream.readRawData(self.__typeSize["uint16"])
            Value,=struct.unpack('H',bts)    # 'H'=unsigned short, 2字节,=uint16
            self.ui.edit_UInt16.setText("%d"%Value)
         except Exception as e:
            QMessageBox.critical(self, "读uint16过程出现错误", str(e))
         finally:
            self.__delFileStream()

         
   @pyqtSlot()   ##写  int32
   def on_btnInt32_Write_clicked(self):  
      Value=self.ui.spin_Int32.value()
      if self.__iniWrite():
         try:
            bts=struct.pack('>l',Value)    # '>l'=大字节序 int32
            self.fileStream.writeRawData(bts)
         except Exception as e:
            QMessageBox.critical(self, "写int32过程出现错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  int32
   def on_btnInt32_Read_clicked(self):  
      if self.__iniRead():
         try:
            bts=self.fileStream.readRawData(self.__typeSize["int32"])
            Value,=struct.unpack('>l',bts)    # '>l'= 大字节序int32
            self.ui.edit_Int32.setText("%d"%Value)
         except Exception as e:
            QMessageBox.critical(self, "读int32过程出现错误", str(e))
         finally:
            self.__delFileStream()


   @pyqtSlot()    ##写 int64
   def on_btnInt64_Write_clicked(self):  
      Value=self.ui.spin_Int64.value()    #Python的int类型
      if self.__iniWrite():
         try:
            bts=struct.pack('q',Value)    # 'q'= long long, 8字节=int64
            self.fileStream.writeRawData(bts)
         except Exception as e:
            QMessageBox.critical(self, "写int64过程出现错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读 int64
   def on_btnInt64_Read_clicked(self):  
      if self.__iniRead():
         try:
            bts=self.fileStream.readRawData(self.__typeSize["int64"])
            Value,=struct.unpack('q',bts)     # 'q'= long long, 8字节=int64
            self.ui.edit_Int64.setText("%d"%Value)
         except Exception as e:
            QMessageBox.critical(self, "读int64过程出现错误", str(e))
         finally:
            self.__delFileStream()


   @pyqtSlot()   ##写  int
   def on_btnInt_Write_clicked(self):  
      Value=self.ui.spin_Int.value()
      if self.__iniWrite():
         try:
            bts=struct.pack('i',Value)  # 'i'=  int, 4字节=int
            self.fileStream.writeRawData(bts)
         except Exception as e:
            QMessageBox.critical(self, "写int过程出现错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  int
   def on_btnInt_Read_clicked(self):  
      if self.__iniRead():
         try:
            bts=self.fileStream.readRawData(self.__typeSize["int"])
            Value,=struct.unpack('i',bts)  # 'i'=  int, 4字节=int
            self.ui.edit_Int.setText("%d"%Value)
         except Exception as e:
            QMessageBox.critical(self, "读int过程出现错误", str(e))
         finally:
            self.__delFileStream()

   @pyqtSlot()   ##写  bool
   def on_btnBool_Write_clicked(self):  
      Value=self.ui.chkBox_In.isChecked()
      if self.__iniWrite(): 
         bts=struct.pack('?',Value)     # '?'=  bool, 1字节
         self.fileStream.writeRawData(bts)
         self.__delFileStream()
          
   @pyqtSlot()    ##读  bool
   def on_btnBool_Read_clicked(self):  
      if self.__iniRead():
         bts=self.fileStream.readRawData(self.__typeSize["bool"])
         Value,=struct.unpack('?',bts)    # '?'=  bool, 1字节
         self.ui.chkBox_Out.setChecked(Value)
         self.__delFileStream()


   @pyqtSlot()   ##写  float
   def on_btnFloat_Write_clicked(self):  
      Value=self.ui.spin_Float.value()       #float型
      if self.__iniWrite():
         try:
            bts=struct.pack('f',Value)          # 'f'=  float, 4字节
            self.fileStream.writeRawData(bts)
         except Exception as e:
            QMessageBox.critical(self, "写float过程出现错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()    ##读  float
   def on_btnFloat_Read_clicked(self):  
      if self.__iniRead():
         try:
            bts=self.fileStream.readRawData(self.__typeSize["float"])
            Value,=struct.unpack('f',bts)       # 'f'=  float, 4字节
            self.ui.edit_Float.setText("%.4f"%Value)
         except Exception as e:
            QMessageBox.critical(self, "读float过程出现错误", str(e))
         finally:
            self.__delFileStream()


   @pyqtSlot()    ##写 double
   def on_btnDouble_Write_clicked(self):  
      Value=self.ui.spin_Double.value()
      if self.__iniWrite():
         try:
            bts=struct.pack('d',Value)     # 'd'=  double, 8字节=double
            self.fileStream.writeRawData(bts)
         except Exception as e:
            QMessageBox.critical(self, "写double过程出现错误", str(e))
         finally:
            self.__delFileStream()
          
   @pyqtSlot()     ##读 double
   def on_btnDouble_Read_clicked(self):  
      if self.__iniRead():
         try:
            bts=self.fileStream.readRawData(self.__typeSize["double"])
            Value,=struct.unpack('d',bts)    # 'd'=  double, 8字节
            self.ui.edit_Double.setText("%.4f"%Value)
         except Exception as e:
            QMessageBox.critical(self, "读double过程出现错误", str(e))
         finally:
            self.__delFileStream()


## BigEndian和LittleEndian会影响字符串前面表示长度的整数的存储顺序
   @pyqtSlot()   ## writeBytes写  String
   def on_btnStr_Write_clicked(self):  
      strV=self.ui.editStr_In.text()    #str类型
      if self.__iniWrite():
         bts=bytes(strV,encoding="utf-8")    #转换为bytes类型
         self.fileStream.writeBytes(bts)     #writeBytes(self, bytes) -> QDataStream
         self.__delFileStream()
          
   @pyqtSlot()    ## readBytes读  String
   def on_btnStr_Read_clicked(self):  
      if self.__iniRead():
         try:
            Value=self.fileStream.readBytes()   #readBytes(self) -> bytes
            strV=Value.decode("utf-8")          #从bytes类型解码为字符串，utf-8码
            self.ui.editStr_Out.setText(strV)
         except Exception as e:
            QMessageBox.critical(self, "读String过程出现错误", str(e))
         finally:
            self.__delFileStream()


   @pyqtSlot()   ## writeRawData 写String, 未知长度无法读出
   def on_btnStr_WriteRaw_clicked(self):  
      strV=self.ui.editStr_RawIn.text()      #str类型
      if self.__iniWrite():
         bts=bytes(strV,encoding="utf-8")    #转换为bytes类型
         self.fileStream.writeRawData(bts)
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

      self.ui.editStr_Out.clear()


   @pyqtSlot()    ##连续写入文件
   def on_actSaveALL_triggered(self):  
      if not self.__iniWrite():
         QMessageBox.critical(self,"错误","为写入打开文件时出错")
         return
   #数据写入部分
      Value=self.ui.spin_Int8.value()
      bts=struct.pack('b',Value)    # 'b'=signed char, int8
      self.fileStream.writeRawData(bts)

      Value=self.ui.spin_UInt8.value()
      bts=struct.pack('B',Value)    # 'B'=unsigned char, uint8
      self.fileStream.writeRawData(bts)

      Value=self.ui.spin_Int16.value()
      bts=struct.pack('h',Value)    # 'h'=short, int16
      self.fileStream.writeRawData(bts)

      Value=self.ui.spin_UInt16.value()
      bts=struct.pack('H',Value)    # 'H'=unsigned short, uint16
      self.fileStream.writeRawData(bts)

      Value=self.ui.spin_Int32.value()
      bts=struct.pack('l',Value)    # 'l'= long, 4字节,int32
      self.fileStream.writeRawData(bts)

      Value=self.ui.spin_Int64.value()
      bts=struct.pack('q',Value)    # 'q'= long long, 8字节int64
      self.fileStream.writeRawData(bts)
      
      Value=self.ui.spin_Int.value()
      bts=struct.pack('i',Value)    # 'i'=  int, 4字节int
      self.fileStream.writeRawData(bts)

      Value=self.ui.chkBox_In.isChecked()
      bts=struct.pack('?',Value)    # '?'=  bool, 1字节
      self.fileStream.writeRawData(bts)

      Value=self.ui.spin_Float.value()
      bts=struct.pack('f',Value)    # 'f'=  float, 4字节
      self.fileStream.writeRawData(bts)

      Value=self.ui.spin_Double.value() 
      bts=struct.pack('d',Value)    # 'd'=  double, 8字节
      self.fileStream.writeRawData(bts)

      strV=self.ui.editStr_In.text()      #str类型
      bts=bytes(strV,encoding="utf-8")    #转换为bytes类型
      self.fileStream.writeBytes(bts) 

   #数据写入完成
      self.__delFileStream()
      QMessageBox.information(self,"消息","数据连续写入完成.")


   @pyqtSlot()    ##连续读取文件
   def on_actReadALL_triggered(self):  
      if not self.__iniRead():
         QMessageBox.critical(self,"错误","为读取打开文件时出错")
         return
   #数据读取部分
      bts=self.fileStream.readRawData(self.__typeSize["int8"])
      Value,=struct.unpack('b',bts)    # 'b'=signed char,int8
      self.ui.edit_Int8.setText("%d"%Value)

      bts=self.fileStream.readRawData(self.__typeSize["uint8"])
      Value,=struct.unpack('B',bts)    # 'B'=unsigned char
      self.ui.edit_UInt8.setText("%d"%Value)

      bts=self.fileStream.readRawData(self.__typeSize["int16"])
      Value,=struct.unpack('h',bts)    #  'h'=short, 2字节
      self.ui.edit_Int16.setText("%d"%Value)

      bts=self.fileStream.readRawData(self.__typeSize["uint16"])
      Value,=struct.unpack('H',bts)    #  'H'=unsigned short, 2字节
      self.ui.edit_UInt16.setText("%d"%Value)

      bts=self.fileStream.readRawData(self.__typeSize["int32"])
      Value,=struct.unpack('l',bts)    #  'l'= long, 4字节
      self.ui.edit_Int32.setText("%d"%Value)

      bts=self.fileStream.readRawData(self.__typeSize["int64"])
      Value,=struct.unpack('q',bts)    #  'q'= long long, 8字节
      self.ui.edit_Int64.setText("%d"%Value)

      bts=self.fileStream.readRawData(self.__typeSize["int"])
      Value,=struct.unpack('i',bts)    # 'i'=  int, 4字节
      self.ui.edit_Int.setText("%d"%Value)

      bts=self.fileStream.readRawData(self.__typeSize["bool"])
      Value,=struct.unpack('?',bts)    # '?'=  bool, 1字节
      self.ui.chkBox_Out.setChecked(Value)

      bts=self.fileStream.readRawData(self.__typeSize["float"])
      Value,=struct.unpack('f',bts)    # 'f'=  float, 4字节
      self.ui.edit_Float.setText("%.4f"%Value)

      bts=self.fileStream.readRawData(self.__typeSize["double"])
      Value,=struct.unpack('d',bts)    #  'd'=  double, 8字节
      self.ui.edit_Double.setText("%.4f"%Value)

      Value=self.fileStream.readBytes()   # bytes类型
      strV=Value.decode("utf-8")          #从bytes解码为字符串，utf-8码
      self.ui.editStr_Out.setText(strV)
      
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
