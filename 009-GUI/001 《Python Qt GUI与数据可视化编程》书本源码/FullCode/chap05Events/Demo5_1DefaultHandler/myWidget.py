import sys

from PyQt5.QtWidgets import  QApplication, QWidget,QMessageBox

from PyQt5.QtCore import  pyqtSlot, Qt, QEvent

from PyQt5.QtGui import   QPainter, QPixmap,QCursor

from ui_Widget import Ui_Widget

class QmyWidget(QWidget): 
   def __init__(self, parent=None):
      super().__init__(parent)   #调用父类构造函数，创建窗体
      self.ui=Ui_Widget()        #创建UI对象
      self.ui.setupUi(self)      #构造UI界面

##  =================自定义功能函数=================================
        
        
##  ==========由connectSlotsByName() 自动连接的槽函数===============
      

##  ==================event处理函数=================================
   def event(self,event):
##      if(event.type()== QEvent.Paint):
##         return True  #不再绘制背景
      return super().event(event)


   def paintEvent(self,event):  ##绘制窗口背景图片
      painter=QPainter(self)
      pic=QPixmap("sea1.jpg")
      painter.drawPixmap(0,0,self.width(),self.height(),pic)
##      super().paintEvent(event)


   def resizeEvent(self,event):  ##窗体改变大小
      W=self.width()
      H=self.height()

      Wbtn=self.ui.btnTest.width()
      Hbtn=self.ui.btnTest.height()

      self.ui.btnTest.setGeometry((W-Wbtn)/2, (H-Hbtn)/2, Wbtn,Hbtn)
        
   def closeEvent(self,event):   ##窗体关闭时询问
      dlgTitle="Question消息框"
      strInfo="closeEvent事件触发，确定要退出吗？"
      defaultBtn=QMessageBox.NoButton    #缺省按钮
      result=QMessageBox.question(self, dlgTitle, strInfo,
                  QMessageBox.Yes | QMessageBox.No,defaultBtn)

      if (result==QMessageBox.Yes):
         event.accept() #窗口可关闭
      else:
         event.ignore() #窗口不能被关闭

   def keyPressEvent(self,event):      ##键盘按键按下
   ##   def keyReleaseEvent(self,event):  
      rect=self.ui.btnMove.geometry()

      if event.key() in set([Qt.Key_A,Qt.Key_Left]):
         self.ui.btnMove.setGeometry(rect.left()-20,rect.top(),rect.width(),rect.height())
      elif event.key() in set([Qt.Key_D, Qt.Key_Right]):
         self.ui.btnMove.setGeometry(rect.left()+20,rect.top(),rect.width(),rect.height())
      elif event.key() in set([Qt.Key_W, Qt.Key_Up]):
         self.ui.btnMove.setGeometry(rect.left(),rect.top()-20,rect.width(),rect.height())
      elif event.key() in set([Qt.Key_S, Qt.Key_Down]):
         self.ui.btnMove.setGeometry(rect.left(),rect.top()+20,rect.width(),rect.height())
##      elif (event.key()==Qt.Key_Q) and (event.modifiers() & Qt.ControlModifier):
##         self.ui.btnMove.setGeometry(rect.left()-100,rect.top(),rect.width(),rect.height())
      
##       event.accept() #被处理,不会再传播到父窗体
##       super().keyPressEvent(event)
##       super().keyReleaseEvent(event)

   def hideEvent(self,event):   #窗体最小化时触发
      print("hideEvent 事件触发")
        
   def showEvent(self,event):   #窗体显示时触发
      print("showEvent 事件触发")

   def mousePressEvent(self,event):  ##鼠标按键按下
      pt=event.pos() #鼠标位置，QPoint
##      pt=event.localPos()   #鼠标位置，QPointF,与pos()相同
##      pt=event.windowPos()  #鼠标位置，QPointF,与pos()相同
      
##      pt=event.globalPos()  #鼠标位置，QPointF,全局位置，全屏的位置
##      pt=event.screenPos()  #鼠标位置，QPointF,全屏位置，与globalPos()相同
      
##      if (event.buttons() & Qt.LeftButton) and (event.buttons() & Qt.RightButton): #左右键同时被按下
      if (event.button() == Qt.LeftButton):     #鼠标左键按下
         self.ui.LabMove.setText("(x,y)=(%d,%d)"%(pt.x(),pt.y()))
         rect=self.ui.LabMove.geometry()
         self.ui.LabMove.setGeometry(pt.x(),pt.y(),rect.width(),rect.height())
      super().mousePressEvent(event)
       
        
##  =============自定义槽函数===============================        

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)     #创建GUI应用程序
   form=QmyWidget()                 #创建窗体
   form.show()
   sys.exit(app.exec_())
