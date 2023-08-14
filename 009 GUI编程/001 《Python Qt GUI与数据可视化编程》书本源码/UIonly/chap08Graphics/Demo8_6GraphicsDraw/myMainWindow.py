import sys, random

from PyQt5.QtCore import  pyqtSlot,Qt,QPointF

from PyQt5.QtGui import  QBrush, QPolygonF,QPen,QFont,QTransform

from PyQt5.QtWidgets import  (QApplication, QMainWindow,QColorDialog,
            QFontDialog, QInputDialog, QLabel,QGraphicsScene,
            QGraphicsView,QGraphicsItem,QGraphicsRectItem,
            QGraphicsEllipseItem, QGraphicsPolygonItem,
            QGraphicsLineItem, QGraphicsItemGroup, QGraphicsTextItem)


from myGraphicsView import QmyGraphicsView

from ui_MainWindow import Ui_MainWindow

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面
      self.setWindowTitle("Demo8_6, Graphics View绘图")
      pass


##  ==============自定义功能函数============
   def __buildStatusBar(self):   ##构造状态栏
      pass


   def __iniGraphicsSystem(self):   ##初始化 Graphics View系统
      pass


   def __setItemProperties(self,item,desc):   ##item是具体类型的QGraphicsItem
      pass


   def __setBrushColor(self,item):  ##设置填充颜色
      pass
      
##==============event 处理函数=======================

        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
##==============创建基本图件==============
   @pyqtSlot()    ##添加一个矩形
   def on_actItem_Rect_triggered(self):
      pass


   @pyqtSlot()    ##添加一个椭圆
   def on_actItem_Ellipse_triggered(self):
      pass


   @pyqtSlot()    ##添加一个圆
   def on_actItem_Circle_triggered(self):
      pass


   @pyqtSlot()    ##添加三角形
   def on_actItem_Triangle_triggered(self):
      pass


   @pyqtSlot()    ##添加梯形
   def on_actItem_Polygon_triggered(self):
      pass

      
   @pyqtSlot()    ##添加直线
   def on_actItem_Line_triggered(self):
      pass

      
   @pyqtSlot()    ##添加文字
   def on_actItem_Text_triggered(self):
      pass


##=============图件的编辑操作===================
   @pyqtSlot()    ##放大
   def on_actZoomIn_triggered(self):
      pass


   @pyqtSlot()    ##缩小
   def on_actZoomOut_triggered(self):
      pass

      
   @pyqtSlot()    ##"恢复"取消所有缩放和旋转变换
   def on_actRestore_triggered(self):
      pass

      
   @pyqtSlot()    ##左旋转
   def on_actRotateLeft_triggered(self):
      pass


   @pyqtSlot()    ##右旋转
   def on_actRotateRight_triggered(self):
      pass


   @pyqtSlot()    ##前置
   def on_actEdit_Front_triggered(self):
      pass

      
   @pyqtSlot()    ##后置
   def on_actEdit_Back_triggered(self):
      pass


   @pyqtSlot()    ##组合
   def on_actGroup_triggered(self):
      pass


   @pyqtSlot()    ##打散组合
   def on_actGroupBreak_triggered(self):
      pass


   @pyqtSlot()    ##删除所有选中的绘图项
   def on_actEdit_Delete_triggered(self):
      pass

        
##  =============自定义槽函数===============================        
   ##鼠标移动事件，point是 GraphicsView的坐标,物理坐标
   def  do_mouseMove(self,point):      ##鼠标移动
      pass

   
   def  do_mouseClicked(self,point):   ##鼠标单击
      pass


   def do_mouseDoubleClick(self,point):  ##鼠标双击
      pass


   def do_keyPress(self,event):     ##按键操作
      pass
   
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
    app = QApplication(sys.argv)    #创建GUI应用程序
    form=QmyMainWindow()            #创建窗体
    form.show()
    sys.exit(app.exec_())
