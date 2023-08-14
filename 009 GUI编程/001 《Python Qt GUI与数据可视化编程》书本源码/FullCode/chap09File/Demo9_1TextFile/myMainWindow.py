# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow

from PyQt5.QtWidgets import  QFileDialog,QMessageBox

from PyQt5.QtCore import  pyqtSlot,QDir,QFile,QIODevice,QTextStream


##from PyQt5.QtGui import

##from PyQt5.QtSql import 

##from PyQt5.QtMultimedia import

##from PyQt5.QtMultimediaWidgets import


from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setCentralWidget(self.ui.textEdit)
      

##  ==============自定义功能函数========================
   def __openByIODevice(self,fileName):  ##用QFile打开文件
      fileDevice=QFile(fileName)
      if not fileDevice.exists():  #判断文件是否存在
         return False

      if not fileDevice.open(QIODevice.ReadOnly | QIODevice.Text):
         return False

###整个文件一次性读取的方式，可行
##      qtBytes=fileDevice.readAll() #返回QByteArray类型
##      pyBytes=bytes(qtBytes.data())  # 将QByteArray转换为bytes类型
##      text=pyBytes.decode("utf-8")  #用utf-8编码为字符串
##      self.ui.textEdit.setPlainText(text)
      
##  逐行读取方式，可行
      try:
         self.ui.textEdit.clear()
         while not fileDevice.atEnd():
            qtBytes = fileDevice.readLine()  # 返回QByteArray类型
            pyBytes=bytes(qtBytes.data())    # QByteArray转换为bytes类型
            lineStr=pyBytes.decode("utf-8")  #bytes转换为str型
            lineStr=lineStr.strip()          #去除结尾增加的空行
            self.ui.textEdit.appendPlainText(lineStr)
      finally:
         fileDevice.close()
         
      return True


   def __saveByIODevice(self,fileName):  ##用QFile保存文件
      fileDevice=QFile(fileName)
      if not fileDevice.open(QIODevice.WriteOnly | QIODevice.Text):
         return False

      try:
         text=self.ui.textEdit.toPlainText()  #返回str类型
         strBytes=text.encode("utf-8")        # str转换为bytes类型
         fileDevice.write(strBytes)           #写入文件
      finally:
         fileDevice.close()
         
      return  True

   def __openByStream(self,fileName):  ##用QTextStream打开文件
      fileDevice=QFile(fileName)
      if not fileDevice.exists():   #判断文件是否存在
         return False

      if not fileDevice.open(QIODevice.ReadOnly | QIODevice.Text):
         return False
      
      try:
         fileStream=QTextStream(fileDevice) 
         fileStream.setAutoDetectUnicode(True)  #自动检测Unicode
         fileStream.setCodec("utf-8")  #必须设置编码,否则不能正常显示汉字
      
   # 一次性全部读出
   ##   text=fileStream.readAll()  #读取出来就是str
   ##   self.ui.textEdit.setPlainText(text)

   #逐行读取方式
         self.ui.textEdit.clear()
         while not fileStream.atEnd():
            lineStr=fileStream.readLine()    #读取文件的一行,读取出来就是str
            self.ui.textEdit.appendPlainText(lineStr)    #添加到文本框显示
      finally:
         fileDevice.close() #关闭文件
      return  True

   def __saveByStream(self,fileName):  ##用 QTextStream 保存文件
      fileDevice=QFile(fileName)
      if not fileDevice.open(QIODevice.WriteOnly | QIODevice.Text):
         return False

      try:
         fileStream=QTextStream(fileDevice)  #用文本流读取文件
         fileStream.setAutoDetectUnicode(True)  #自动检测Unicode
         fileStream.setCodec("utf-8")  #必须设置编码,否则不能正常显示汉字

         text=self.ui.textEdit.toPlainText()  #返回是str类型 
         fileStream<<text   #使用流操作符
   ##         fileStream<<"\n***************在尾部添加的第1行"
      finally:
         fileDevice.close()
      return  True
      

##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============
   
   @pyqtSlot()   ##用QFile 打开文件
   def on_actQFile_Open_triggered(self):
      curPath=QDir.currentPath()    #获取系统当前目录
      title="打开一个文件"    #对话框标题
      filt="程序文件(*.h *.cpp *.py);;文本文件(*.txt);;所有文件(*.*)"   #文件过滤器
      fileName,flt=QFileDialog.getOpenFileName(self,title,curPath,filt)
      if (fileName == ""):
         return

      if self.__openByIODevice(fileName):
         self.ui.statusBar.showMessage(fileName)
      else:
         QMessageBox.critical(self,"错误","打开文件失败")

   @pyqtSlot()   ##用QFile 另存文件
   def on_actQFile_Save_triggered(self):
      curPath=QDir.currentPath()    #获取系统当前目录
      title="另存为一个文件"        #对话框标题
      filt="Python程序(*.py);; C++程序(*.h *.cpp);;文本文件(*.txt);;所有文件(*.*)"  #文件过滤器
      fileName,flt=QFileDialog.getSaveFileName(self,title,curPath,filt)
      if (fileName==""):
         return

      if self.__saveByIODevice(fileName):
         self.ui.statusBar.showMessage(fileName)
      else:
         QMessageBox.critical(self,"错误","保存文件失败")

   @pyqtSlot()   ##用QTextStream 打开文件
   def on_actStream_Open_triggered(self):
      curPath=QDir.currentPath() #获取系统当前目录
      title="打开一个文件"       #对话框标题
      filt="程序文件(*.h *.cpp *.py);;文本文件(*.txt);;所有文件(*.*)"   #文件过滤器
      fileName,flt=QFileDialog.getOpenFileName(self,title,curPath,filt)
      if (fileName == ""):
         return

      if self.__openByStream(fileName):
         self.ui.statusBar.showMessage(fileName)
      else:
         QMessageBox.critical(self,"错误","打开文件失败")


   @pyqtSlot()   ##用QTextStream 另存文件
   def on_actStream_Save_triggered(self):
      curPath=QDir.currentPath()  #获取系统当前目录
      title="另存为一个文件"      #对话框标题
      filt="Python程序(*.py);; C++程序(*.h *.cpp);;所有文件(*.*)"   #文件过滤器
      fileName,flt=QFileDialog.getSaveFileName(self,title,curPath,filt)
      if (fileName==""):
         return

      if self.__saveByStream(fileName):
         self.ui.statusBar.showMessage(fileName)
      else:
         QMessageBox.critical(self,"错误","保存文件失败")


   @pyqtSlot()   ##用 python 的file()打开文件
   def on_actPY_Open_triggered(self):
##      curPath=os.getcwd()   #获取系统当前目录
      curPath=QDir.currentPath()    #获取系统当前目录
      title="打开一个文件"          #对话框标题
      filt="程序文件(*.h *.cpp *.py);;所有文件(*.*)"    #文件过滤器
      fileName,flt=QFileDialog.getOpenFileName(self,title,curPath,filt)
      if (fileName == ""):
         return

      self.ui.textEdit.clear()
   ##      aFile=codecs.open(fileName,encoding='utf-8')#可用
   ##      aFile=codecs.open(fileName,'r', "utf-8")  #打开文件，用utf-8码
      fileDevice=open(fileName,mode='r', encoding='utf-8')  #打开文件

      try:
         for eachLine in fileDevice:  #每次读取一行
            lineStr=eachLine.strip()  #必须用strip()去掉末尾的'\0'
            self.ui.textEdit.appendPlainText(lineStr)
         self.ui.statusBar.showMessage(fileName)
      finally:
         fileDevice.close()

         
   @pyqtSlot()   ##用 python的file() 保存文件
   def on_actPY_Save_triggered(self):
   ##      curPath=os.getcwd()   #获取系统当前目录
      curPath=QDir.currentPath()    #获取系统当前目录
      title="另存为一个文件"         #对话框标题
      filt="Python程序(*.py);; C++程序(*.h *.cpp);;所有文件(*.*)"   #文件过滤器
      fileName,flt=QFileDialog.getSaveFileName(self,title,curPath,filt)
      if (fileName==""):
         return

      text=self.ui.textEdit.toPlainText()  
      fileDevice=open(fileName,mode='w', encoding='utf-8')
      try:
         fileDevice.write(text)
         self.ui.statusBar.showMessage(fileName)
      finally:
         fileDevice.close()
        
##  =============自定义槽函数===============================        


   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
