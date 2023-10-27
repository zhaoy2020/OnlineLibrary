import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow

from PyQt5.QtCore import  pyqtSlot, Qt

##from PyQt5.QtWidgets import  

from PyQt5.QtGui import QPainter, QPixmap

from ui_MainWindow import Ui_MainWindow

from myFormDoc import QmyFormDoc

from myFormTable import QmyFormTable

class QmyMainWindow(QMainWindow):
   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面

      self.ui.tabWidget.setVisible(False)
      self.ui.tabWidget.clear()           #清除所有页面
      self.ui.tabWidget.setTabsClosable(True)    #Page有关闭按钮
      self.ui.tabWidget.setDocumentMode(True)

      self.setCentralWidget(self.ui.tabWidget)
      self.setWindowState(Qt.WindowMaximized)   #窗口最大化显示
      self.setAutoFillBackground(True)          #自动绘制背景

      self.__pic=QPixmap("sea1.jpg")  #载入背景图片到内存，提高绘制速度

##  ==============自定义功能函数============

##  =============event事件处理函数============
   def paintEvent(self,event):
      painter=QPainter(self)
##       pic=QPixmap("sea1.jpg")
      painter.drawPixmap(0,self.ui.mainToolBar.height(), self.width(),
         self.height()-self.ui.mainToolBar.height()-self.ui.statusBar.height(),
         self.__pic)
      super().paintEvent(event)
        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
   @pyqtSlot()     ## "嵌入式Widget"
   def on_actWidgetInsite_triggered(self):
      formDoc=QmyFormDoc(self)
   ##        formDoc=QmyFormDoc()   #也可以显示
      formDoc.setAttribute(Qt.WA_DeleteOnClose)   #关闭时自动删除
      formDoc.docFileChanged.connect(self.do_docFileChanged)

      title= "Doc %d"%self.ui.tabWidget.count()
      curIndex=self.ui.tabWidget.addTab(formDoc,title)   #添加到tabWidget
      self.ui.tabWidget.setCurrentIndex(curIndex)
      self.ui.tabWidget.setVisible(True)

   @pyqtSlot()   ##独立Widget窗口
   def on_actWidget_triggered(self):
      formDoc=QmyFormDoc(self)   #必须传递self，否则无法显示
      formDoc.setAttribute(Qt.WA_DeleteOnClose)
      formDoc.setWindowTitle("基于QWidget的窗体，关闭时删除")
      formDoc.setWindowFlag(Qt.Window,True)

##      formDoc.setWindowFlag(Qt.CustomizeWindowHint,True)
##      formDoc.setWindowFlag(Qt.WindowMinMaxButtonsHint,False)
##      formDoc.setWindowFlag(Qt.WindowCloseButtonHint,True)
##      formDoc.setWindowFlag(Qt.WindowStaysOnTopHint,True)
      
      formDoc.setWindowOpacity(0.9)   #窗口透明度
      formDoc.show()

   @pyqtSlot()  ##"嵌入式MainWindow"
   def on_actWindowInsite_triggered(self):
   ##        formTable=QmyFormTable()  #无差别
      formTable=QmyFormTable(self)
      formTable.setAttribute(Qt.WA_DeleteOnClose)

      title= "Table %d"%self.ui.tabWidget.count()
      curIndex=self.ui.tabWidget.addTab(formTable,title)
      self.ui.tabWidget.setCurrentIndex(curIndex)
      self.ui.tabWidget.setVisible(True)

   @pyqtSlot()     ##"独立MainWindow窗口"
   def on_actWindow_triggered(self):
      formTable=QmyFormTable(self)#必须传递self，否则无法显示
##        formTable=QmyFormTable()  #不可行，一显示就删除了
      formTable.setAttribute(Qt.WA_DeleteOnClose)
      formTable.setWindowTitle("基于QMainWindow的窗口，关闭时删除")
      formTable.show()

   def on_tabWidget_currentChanged(self,index): #tabWidget当前页面变化
      hasTabs=self.ui.tabWidget.count()>0       #再无页面时
      self.ui.tabWidget.setVisible(hasTabs)

   def on_tabWidget_tabCloseRequested(self,index):  #分页关闭时关闭窗体
      if (index<0):
         return
      aForm=self.ui.tabWidget.widget(index)
      aForm.close()
        
    
##  =============自定义槽函数===============================        
   @pyqtSlot(str)       ##显示文件名
   def do_docFileChanged(self,shotFilename):   
      index=self.ui.tabWidget.currentIndex()
      self.ui.tabWidget.setTabText(index,shotFilename)
       
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":
   app = QApplication(sys.argv)
   form=QmyMainWindow()
   form.show()
   sys.exit(app.exec_())
