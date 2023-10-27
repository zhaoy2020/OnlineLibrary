import sys,os

from PyQt5.QtWidgets import  QApplication, QWidget

##from PyQt5.QtCore import  Qt

##from PyQt5.QtWidgets import  QMessageBox

from PyQt5.QtGui import  QPixmap

from ui_Widget import Ui_Widget


class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

      self.setAcceptDrops(True)
      self.ui.plainTextEdit.setAcceptDrops(True)   #不处理drop,由父窗体处理
      self.ui.LabPic.setAcceptDrops(False)
      self.ui.LabPic.setScaledContents(True)       #图片适应Label大小

##  =================自定义功能函数=================================
        
##  ==========由connectSlotsByName() 自动连接的槽函数===============        

##  ==================event处理函数=================================
   def dragEnterEvent(self,event):
      self.ui.plainTextEdit.clear()
      self.ui.plainTextEdit.appendPlainText("dragEnterEvent事件 mimeData()->formats()")
      for  strLine in event.mimeData().formats():   #mimeData().formats()
         self.ui.plainTextEdit.appendPlainText(strLine)

      self.ui.plainTextEdit.appendPlainText("\ndragEnterEvent事件 mimeData()->urls()")
      for url in event.mimeData().urls():    # mimeData().urls()
         self.ui.plainTextEdit.appendPlainText(url.path())

      if (event.mimeData().hasUrls()):
         filename=event.mimeData().urls()[0].fileName()  #只有文件名
         basename,ext=os.path.splitext(filename)         #文件名和后缀
         ext=ext.upper()
         if (ext ==".JPG"):  #只接收JPG文件
            event.acceptProposedAction()
         else:
            event.ignore()
      else:
         event.ignore()
        
   def dropEvent(self,event):
      filename=event.mimeData().urls()[0].path()  #完整文件名
      cnt=len(filename)

      realname=filename[1:cnt]    #去掉最左边的“/”
      pixmap = QPixmap(realname)
      self.ui.LabPic.setPixmap(pixmap)
      event.accept()
        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":     
    app = QApplication(sys.argv)
    form=QmyWidget()            
    form.show()
    sys.exit(app.exec_())
