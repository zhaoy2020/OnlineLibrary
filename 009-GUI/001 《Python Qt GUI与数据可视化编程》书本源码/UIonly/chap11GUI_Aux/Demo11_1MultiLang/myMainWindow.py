# -*- coding: utf-8 -*-

import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QActionGroup

##from PyQt5.QtWidgets import   
from PyQt5.QtGui import  QTextCharFormat, QFont

from PyQt5.QtCore import (Qt, pyqtSlot,QCoreApplication,
                  QTranslator,QSettings,QT_TRANSLATE_NOOP)

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 
   _tr = QCoreApplication.translate  #替代符

   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()    #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      text=self._tr("QmyMainWindow","文件名: ")
      self.ui.statusBar.showMessage(text)

      actionGroup= QActionGroup(self)
      actionGroup.addAction(self.ui.actLang_CN)
      actionGroup.addAction(self.ui.actLang_EN)
      actionGroup.setExclusive(True)

      self.__translator=None  #QTranslator对象

      self.setCentralWidget(self.ui.textEdit)

##  ============自定义功能函数================================
   def   setTranslator(self,translator,Language):
      self.__translator=translator
      if Language=="EN":
         self.ui.actLang_EN.setChecked(True)
      else:
         self.ui.actLang_CN.setChecked(True)
        
##  ===========由connectSlotsByName() 自动连接的槽函数=====================      
   @pyqtSlot()    ##英语界面
   def on_actLang_EN_triggered(self): 
      QCoreApplication.removeTranslator(self.__translator)  
      self.__translator=QTranslator()
      self.__translator.load("appLang_EN.qm")
      QCoreApplication.installTranslator(self.__translator)
      self.ui.retranslateUi(self)

      regSettings=QSettings(QCoreApplication.organizationName(),
                           QCoreApplication.applicationName()) 
      regSettings.setValue("Language","EN") #保存设置

   @pyqtSlot()    ##汉语界面
   def on_actLang_CN_triggered(self):  
      QCoreApplication.removeTranslator(self.__translator)  
      self.__translator=QTranslator()
      self.__translator.load("appLang_CN.qm")
      QCoreApplication.installTranslator(self.__translator)
      self.ui.retranslateUi(self)

      regSettings=QSettings(QCoreApplication.organizationName(),
                           QCoreApplication.applicationName())
      regSettings.setValue("Language","CN") #保存设置

   @pyqtSlot(bool)      ##设置粗体 
   def on_actFont_Bold_triggered(self, checked):    
      fmt=self.ui.textEdit.currentCharFormat()
      if (checked == True):
         fmt.setFontWeight(QFont.Bold)
      else:
         fmt.setFontWeight(QFont.Normal)
      self.ui.textEdit.mergeCurrentCharFormat(fmt)

   @pyqtSlot(bool)      ##设置斜体
   def on_actFont_Italic_triggered(self,checked):       
      fmt=self.ui.textEdit.currentCharFormat()
      fmt.setFontItalic(checked)
      self.ui.textEdit.mergeCurrentCharFormat(fmt)
        
   @pyqtSlot(bool)      ##设置下划线   
   def on_actFont_UnderLine_triggered(self,checked):   
      fmt=self.ui.textEdit.currentCharFormat()
      fmt.setFontUnderline(checked)
      self.ui.textEdit.mergeCurrentCharFormat(fmt)

   def on_textEdit_copyAvailable(self, avi):    ##文本框内容可copy
      self.ui.actEdit_Cut.setEnabled(avi)
      self.ui.actEdit_Copy.setEnabled(avi)
      self.ui.actEdit_Paste.setEnabled(self.ui.textEdit.canPaste())
                                         
   def on_textEdit_selectionChanged(self):      ##文本选择内容发生变化
      fmt=self.ui.textEdit.currentCharFormat()
      self.ui.actFont_Bold.setChecked(fmt.font().bold())
      self.ui.actFont_Italic.setChecked(fmt.fontItalic())
      self.ui.actFont_UnderLine.setChecked(fmt.fontUnderline())

   def on_textEdit_customContextMenuRequested(self,pos):   ##标准右键菜单
      popMenu=self.ui.textEdit.createStandardContextMenu()
      popMenu.exec(pos)

   @pyqtSlot(bool)      ##设置工具栏按钮样式
   def on_actSys_ToggleText_triggered(self,checked):   
      btnStyle=Qt.ToolButtonIconOnly
      if(checked):
         btnStyle=Qt.ToolButtonTextUnderIcon
      self.ui.mainToolBar.setToolButtonStyle(btnStyle)

   def on_actFile_New_triggered(self):     ##新建文件，不实现具体功能
      text=self._tr("QmyMainWindow","新建文件")
      self.ui.statusBar.showMessage(text)

   def on_actFile_Open_triggered(self):    ##打开文件，不实现具体功能
      text=self._tr("QmyMainWindow","打开的文件:")
      self.ui.statusBar.showMessage(text)
        
   def on_actFile_Save_triggered(self):    ##保存文件，不实现具体功能
      text=self._tr("QmyMainWindow","文件已保存")
      self.ui.statusBar.showMessage(text)
        
        
##  =============自定义槽函数===============================        
        
   
##  ===========窗体测试程序=================================        
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序

# 读取注册表里的语言设置
   QCoreApplication.setOrganizationName("mySoft")
   QCoreApplication.setApplicationName("Demo11_1")

##organization="mySoft"  #用于注册表
##appName="Demo11_1" #HKEY_CURRENT_USER/Software/mySoft/Demo11_1
   regSettings=QSettings(QCoreApplication.organizationName(),
                       QCoreApplication.applicationName())  #创建
   Language=regSettings.value("Language","EN")  #读取Language键的值，缺省"EN"
   trans=QTranslator()
   if Language=="EN":
      trans.load("appLang_EN.qm")
   else:
      trans.load("appLang_CN.qm")
   QCoreApplication.installTranslator(trans)

   form=QmyMainWindow()                #创建窗体
   form.setTranslator(trans,Language)  #赋值给QmyMainWindow的类变量translator
   form.show()
   sys.exit(app.exec_())
