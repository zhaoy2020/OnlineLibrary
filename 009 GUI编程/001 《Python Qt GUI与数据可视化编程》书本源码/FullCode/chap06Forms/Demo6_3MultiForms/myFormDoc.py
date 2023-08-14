import sys,os  #codecs

from PyQt5.QtWidgets import  (QApplication, QWidget,QFileDialog,
                         QToolBar, QVBoxLayout,QFontDialog)

from PyQt5.QtCore import  pyqtSlot, pyqtSignal,Qt

from PyQt5.QtGui import QPalette,  QFont

from ui_QWFormDoc import Ui_QWFormDoc

class QmyFormDoc(QWidget):
   docFileChanged=pyqtSignal(str)  ##自定义信号，打开文件时发射

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_QWFormDoc()     #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.__curFile=""
      self.__buildUI()  #构建工具栏
      self.setAutoFillBackground(True)
        
   def __del__(self):   ##析构函数
      print("QmyFormDoc 对象被删除了")

   def __buildUI(self):    ##使用UI可视化设计的Actions创建工具栏
      locToolBar = QToolBar("文档",self)    #创建工具栏
      locToolBar.addAction(self.ui.actOpen)
      locToolBar.addAction(self.ui.actFont)
      locToolBar.addSeparator()
      locToolBar.addAction(self.ui.actCut)
      locToolBar.addAction(self.ui.actCopy)
      locToolBar.addAction(self.ui.actPaste)
      locToolBar.addAction(self.ui.actUndo)
      locToolBar.addAction(self.ui.actRedo)
      locToolBar.addSeparator()
      locToolBar.addAction(self.ui.actClose)
      locToolBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

      Layout = QVBoxLayout()
      Layout.addWidget(locToolBar)  #设置工具栏和编辑器上下布局
      Layout.addWidget(self.ui.plainTextEdit)
      Layout.setContentsMargins(2,2,2,2)  #减小边框的宽度
      Layout.setSpacing(2)
      self.setLayout(Layout)  #设置布局
        

##  ==============自定义功能函数============
        
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()    ##打开文件
   def on_actOpen_triggered(self): 
      curPath=os.getcwd()  #获取当前路径
      filename,flt=QFileDialog.getOpenFileName(self,"打开一个文件",curPath,
                  "文本文件(*.cpp *.h *.py);;所有文件(*.*)")
      if (filename==""):
         return

      self.__curFile=filename
      self.ui.plainTextEdit.clear()

      aFile=open(filename,'r',encoding='utf-8')
      try:
         for eachLine in aFile:  #每次读取一行
            self.ui.plainTextEdit.appendPlainText(eachLine.strip())
      finally:
         aFile.close()

      baseFilename=os.path.basename(filename) #去掉目录后的文件名
      self.setWindowTitle(baseFilename)
      self.docFileChanged.emit(baseFilename)  #发射信号，传递文件名

   @pyqtSlot()    ##设置字体
   def on_actFont_triggered(self):
      iniFont=self.ui.plainTextEdit.font()   #获取文本框的字体
      font,OK=QFontDialog.getFont(iniFont)   #选择字体, 注意与C++版本不同
      if (OK):     #选择有效
         self.ui.plainTextEdit.setFont(font)
        
##  =============自定义槽函数===============================        
        
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyFormDoc()
   form.show()
   sys.exit(app.exec_())
