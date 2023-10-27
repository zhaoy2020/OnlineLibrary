# -*- coding: utf-8 -*-

##  GUI应用程序主程序入口

import sys

from PyQt5.QtWidgets import  QApplication

from PyQt5.QtCore import  QTranslator,QSettings,QCoreApplication

from myMainWindow import QmyMainWindow
    
app = QApplication(sys.argv)    #创建GUI应用程序

# 读取注册表里的语言设置
QCoreApplication.setOrganizationName("WWB")
QCoreApplication.setApplicationName("Demo11_1")

##organization="WWB"  #用于注册表
##appName="Demo11_1" #HKEY_CURRENT_USER/WWB/Demo11_1
regSettings=QSettings(QCoreApplication.organizationName(),
                      QCoreApplication.applicationName()) #创建
Language=regSettings.value("Language","EN")  #读取Language键的值，缺省"EN"

trans=QTranslator()
if Language=="EN":
   trans.load("appLang_EN.qm")
else:
   trans.load("appLang_CN.qm")

QCoreApplication.installTranslator(trans)

mainform=QmyMainWindow()        #创建主窗体
QmyMainWindow.translator=trans  #赋值给QmyMainWindow的类变量translator
mainform.show()                 #显示主窗体

sys.exit(app.exec_())



