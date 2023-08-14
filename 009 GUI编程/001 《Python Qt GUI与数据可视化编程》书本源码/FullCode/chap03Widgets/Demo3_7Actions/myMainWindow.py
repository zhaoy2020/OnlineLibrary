import sys

from PyQt5.QtWidgets import (QApplication, QMainWindow,QActionGroup,
                 QLabel, QProgressBar, QSpinBox, QFontComboBox)

from PyQt5.QtCore import  Qt, pyqtSlot

from PyQt5.QtGui import  QTextCharFormat, QFont

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 
   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面

      self.__buildUI()    #动态创建组件，添加到工具栏和状态栏
      self.__spinFontSize.valueChanged[int].connect(
               self.do_fontSize_Changed)  #字体大小设置
      self.__comboFontName.currentIndexChanged[str].connect(
               self.do_fontName_Changed)  #字体选择

      self.setCentralWidget(self.ui.textEdit)

##  ============自定义功能函数================================        

   def __buildUI(self):    ##窗体上动态添加组件
      # 创建状态栏上的组件
      self.__LabFile=QLabel(self)     # QLabel组件显示信息
      self.__LabFile.setMinimumWidth(150)
      self.__LabFile.setText("文件名： ")
      self.ui.statusBar.addWidget(self.__LabFile)

      self.__progressBar1=QProgressBar(self)      # progressBar1
      self.__progressBar1.setMaximumWidth(200)
      self.__progressBar1.setMinimum(5)
      self.__progressBar1.setMaximum(50)
      sz=self.ui.textEdit.font().pointSize()  #字体大小
      self.__progressBar1.setValue(sz)
      self.ui.statusBar.addWidget(self.__progressBar1)

      self.__LabInfo=QLabel(self)     # QLabel组件显示字体名称
      self.__LabInfo.setText("选择字体名称： ")
      self.ui.statusBar.addPermanentWidget(self.__LabInfo)
        
      #为actLang_CN和actLang_EN创建QActionGroup，互斥型选择
      actionGroup= QActionGroup(self)
      actionGroup.addAction(self.ui.actLang_CN)
      actionGroup.addAction(self.ui.actLang_EN)
      actionGroup.setExclusive(True)      #互斥型分组
      self.ui.actLang_CN.setChecked(True)

      #创建工具栏上的组件
      self.__spinFontSize=QSpinBox(self)  #字体大小spinbox
      self.__spinFontSize.setMinimum(5)
      self.__spinFontSize.setMaximum(50)
      sz=self.ui.textEdit.font().pointSize()
      self.__spinFontSize.setValue(sz)
      self.__spinFontSize.setMinimumWidth(50)
      self.ui.mainToolBar.addWidget(self.__spinFontSize) #SpinBox添加到工具栏

      self.__comboFontName=QFontComboBox(self)  #字体 combobox
      self.__comboFontName.setMinimumWidth(100)
      self.ui.mainToolBar.addWidget(self.__comboFontName) 

      self.ui.mainToolBar.addSeparator()  #添加一个分隔条
      self.ui.mainToolBar.addAction(self.ui.actClose)  #添加 actClose作为“关闭”按钮
        
##  ===========由connectSlotsByName() 自动连接的槽函数=====================      
   @pyqtSlot(bool)   ##设置粗体 
   def on_actFont_Bold_triggered(self, checked):   
      fmt=self.ui.textEdit.currentCharFormat()
      if (checked == True):
         fmt.setFontWeight(QFont.Bold)
      else:
         fmt.setFontWeight(QFont.Normal)
      self.ui.textEdit.mergeCurrentCharFormat(fmt)

   @pyqtSlot(bool)   ##设置斜体 
   def on_actFont_Italic_triggered(self,checked):    
      fmt=self.ui.textEdit.currentCharFormat()
      fmt.setFontItalic(checked)
      self.ui.textEdit.mergeCurrentCharFormat(fmt)
        
   @pyqtSlot(bool)     ##设置下划线 
   def on_actFont_UnderLine_triggered(self,checked):  
      fmt=self.ui.textEdit.currentCharFormat()
      fmt.setFontUnderline(checked)
      self.ui.textEdit.mergeCurrentCharFormat(fmt)

   def on_textEdit_copyAvailable(self, avi): ##文本框内容可copy
      self.ui.actEdit_Cut.setEnabled(avi)
      self.ui.actEdit_Copy.setEnabled(avi)
      self.ui.actEdit_Paste.setEnabled(self.ui.textEdit.canPaste())
                                         
   def on_textEdit_selectionChanged(self):     ##文本选择内容发生变化
      fmt=self.ui.textEdit.currentCharFormat()
      self.ui.actFont_Bold.setChecked(fmt.font().bold())
      self.ui.actFont_Italic.setChecked(fmt.fontItalic())
      self.ui.actFont_UnderLine.setChecked(fmt.fontUnderline())

   def on_textEdit_customContextMenuRequested(self,pos):  ##标准右键菜单
      popMenu=self.ui.textEdit.createStandardContextMenu()
      popMenu.exec(pos) #显示快捷菜单

   @pyqtSlot(bool)   ##设置工具栏按钮样式 
   def on_actSys_ToggleText_triggered(self,checked):  
      if(checked):
         st=Qt.ToolButtonTextUnderIcon
      else:
         st=Qt.ToolButtonIconOnly
      self.ui.mainToolBar.setToolButtonStyle(st)

   def on_actFile_New_triggered(self):     ##新建文件，不实现具体功能
      self.__LabFile.setText(" 新建文件 ")

   def on_actFile_Open_triggered(self):    ##打开文件，不实现具体功能
      self.__LabFile.setText(" 打开的文件 ")
        
   def on_actFile_Save_triggered(self):    ##保存文件，不实现具体功能
      self.__LabFile.setText(" 文件已保存 ")
        
##  =============自定义槽函数===============================        

   @pyqtSlot(int)    ##设置字体大小,关联 __spinFontSize
   def do_fontSize_Changed(self, fontSize):      
      fmt=self.ui.textEdit.currentCharFormat()
      fmt.setFontPointSize(fontSize)
      self.ui.textEdit.mergeCurrentCharFormat(fmt)
      self.__progressBar1.setValue(fontSize)    #注意它属于self, 而不是self.ui

   @pyqtSlot(str)   ## 选择字体名称，关联__comboFontName
   def do_fontName_Changed(self, fontName):  
      fmt=self.ui.textEdit.currentCharFormat()
      fmt.setFontFamily(fontName)
      self.ui.textEdit.mergeCurrentCharFormat(fmt)
      self.__LabInfo.setText("字体名称：%s   "%fontName)
   
##  ===========窗体测试程序=================================        
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
