import sys,codecs,os

from PyQt5.QtWidgets import  QApplication, QWidget,QFontDialog

##from PyQt5.QtWidgets import  

from PyQt5.QtCore import  Qt

##from PyQt5.QtGui import QStandardItemModel

from ui_QWFormDoc import Ui_QWFormDoc

class QmyFormDoc(QWidget):

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_QWFormDoc()      #创建UI对象
      self.ui.setupUi(self)       #构造UI界面

      self.setWindowTitle("New Doc")         #窗口标题
      self.setAttribute(Qt.WA_DeleteOnClose) #MDI子窗口会被自动删除

      self.__currentFile=""      #当前文件名
      self.__fileOpened=False    #是否已打开文件

   def __del__(self):   ##析构函数
      print("QmyFormDoc 对象被删除了")

##  ==============自定义功能函数============
   def loadFromFile(self, aFileName):  ##打开文件
      aFile=codecs.open(aFileName,encoding='utf-8')
      try:
         for eachLine in aFile:  #每次读取一行
            self.ui.plainTextEdit.appendPlainText(eachLine.strip())
      finally:
         aFile.close()

      self.__currentFile=aFileName
      self.__fileOpened=True

      baseFilename=os.path.basename(aFileName)  #去掉目录后的文件名
      self.setWindowTitle(baseFilename)

   def currentFileName(self): ##返回当前文件名
      return self.__currentFile

   def isFileOpened(self): ##文件是否打开
      return self.__fileOpened

   def textCut(self):   ## cut 操作
      self.ui.plainTextEdit.cut()

   def textCopy(self):  ## copy 操作
      self.ui.plainTextEdit.copy()
        
   def textPaste(self): ## paste 操作
      self.ui.plainTextEdit.paste()

   def textSetFont(self):  ##设置字体
      iniFont=self.ui.plainTextEdit.font() #获取文本框的字体
      font,OK=QFontDialog.getFont(iniFont) #选择字体, 注意与C++版本不同
      if (OK):     #选择有效
         self.ui.plainTextEdit.setFont(font)
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
        
    
##  =============自定义槽函数===============================        
        
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyFormDoc()
   form.show()
   sys.exit(app.exec_())
