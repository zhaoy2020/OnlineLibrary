##  GUI应用程序主程序入口

import sys

from PyQt5.QtWidgets import  QApplication

from myDialog import QmyDialog
    
app = QApplication(sys.argv)     #创建GUI应用程序
mainform=QmyDialog()             #创建主窗体
mainform.show()                  #显示主窗体

sys.exit(app.exec_()) 
