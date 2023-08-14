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

      self.__buildStatusBar()    #构造状态栏
      self.__iniGraphicsSystem() #初始化 graphics View系统
      
      self.__ItemId=1   #绘图项自定义数据的key
      self.__ItemDesc=2 #绘图项自定义数据的key

      self.__seqNum=0  #每个图形项设置一个序号
      self.__backZ=0   #后置序号
      self.__frontZ=0  #前置序号

##  ==============自定义功能函数============
   def __buildStatusBar(self):   ##构造状态栏
      self.__labViewCord=QLabel("View 坐标：")
      self.__labViewCord.setMinimumWidth(150)
      self.ui.statusBar.addWidget(self.__labViewCord)

      self.__labSceneCord=QLabel("Scene 坐标：")
      self.__labSceneCord.setMinimumWidth(150)
      self.ui.statusBar.addWidget(self.__labSceneCord)

      self.__labItemCord=QLabel("Item 坐标：")
      self.__labItemCord.setMinimumWidth(150)
      self.ui.statusBar.addWidget(self.__labItemCord)

      self.__labItemInfo=QLabel("ItemInfo: ")
      self.ui.statusBar.addPermanentWidget(self.__labItemInfo)
      

   def __iniGraphicsSystem(self):   ##初始化 Graphics View系统
      self.view=QmyGraphicsView(self)   #创建图形视图组件
      self.setCentralWidget(self.view)

      self.scene=QGraphicsScene(-300,-200,600,200) #创建QGraphicsScene
      self.view.setScene(self.scene)   #与view关联

      self.view.setCursor(Qt.CrossCursor)     #设置鼠标
      self.view.setMouseTracking(True)
      self.view.setDragMode(QGraphicsView.RubberBandDrag)

      ##  4个信号与槽函数的关联
      self.view.mouseMove.connect(self.do_mouseMove)     #鼠标移动
      self.view.mouseClicked.connect(self.do_mouseClicked)   #左键按下
      self.view.mouseDoubleClick.connect(self.do_mouseDoubleClick) #鼠标双击
      self.view.keyPress.connect(self.do_keyPress)    #左键按下

   def __setItemProperties(self,item,desc):   ##item是具体类型的QGraphicsItem
      item.setFlag(QGraphicsItem.ItemIsFocusable)
      item.setFlag(QGraphicsItem.ItemIsMovable)
      item.setFlag(QGraphicsItem.ItemIsSelectable)

      self.__frontZ=1+self.__frontZ   
      item.setZValue(self.__frontZ)  #叠放次序
      item.setPos(-150+random.randint(1,200),-200+random.randint(1,200))

      self.__seqNum=1+self.__seqNum  
      item.setData(self.__ItemId,self.__seqNum) #图件编号
      item.setData(self.__ItemDesc,desc)        #图件描述

      self.scene.addItem(item)
      self.scene.clearSelection()
      item.setSelected(True)

   def __setBrushColor(self,item):  ##设置填充颜色
      color=item.brush().color()
      color=QColorDialog.getColor(color,self,"选择填充颜色")
      if color.isValid():
         item.setBrush(QBrush(color))
      
##==============event 处理函数=======================

        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
##==============创建基本图件==============
   @pyqtSlot()    ##添加一个矩形
   def on_actItem_Rect_triggered(self):
      item=QGraphicsRectItem(-50,-25,100,50)
      item.setBrush(QBrush(Qt.yellow)) #设置填充颜色
      self.__setItemProperties(item,"矩形")

   @pyqtSlot()    ##添加一个椭圆
   def on_actItem_Ellipse_triggered(self):
      item=QGraphicsEllipseItem(-50,-30,100,60) 
      item.setBrush(QBrush(Qt.blue))  #设置填充颜色
      self.__setItemProperties(item,"椭圆")

   @pyqtSlot()    ##添加一个圆
   def on_actItem_Circle_triggered(self):
      item=QGraphicsEllipseItem(-50,-50,100,100) 
      item.setBrush(QBrush(Qt.cyan))  #设置填充颜色
      self.__setItemProperties(item,"圆形")

   @pyqtSlot()    ##添加三角形
   def on_actItem_Triangle_triggered(self):
      item=QGraphicsPolygonItem()
      points=[QPointF(0,-40), QPointF(60,40), QPointF(-60,40)]
      item.setPolygon(QPolygonF(points))
      item.setBrush(QBrush(Qt.magenta))  #设置填充颜色
      self.__setItemProperties(item,"三角形")

   @pyqtSlot()    ##添加梯形
   def on_actItem_Polygon_triggered(self):
      item=QGraphicsPolygonItem()
      points=[QPointF(-40,-40), QPointF(40,-40), QPointF(100,40),QPointF(-100,40)]
      item.setPolygon(QPolygonF(points))
      item.setBrush(QBrush(Qt.green))  #设置填充颜色
      self.__setItemProperties(item,"梯形")
      
   @pyqtSlot()    ##添加直线
   def on_actItem_Line_triggered(self):
      item=QGraphicsLineItem(-100,0,100,0)
      pen=QPen(Qt.red)
      pen.setWidth(4)
      item.setPen(pen)    #设置线条属性
      self.__setItemProperties(item,"直线")
      
   @pyqtSlot()    ##添加文字
   def on_actItem_Text_triggered(self):
      strText,OK=QInputDialog.getText(self,"输入","请输入文字")
      if (not OK):
         return
      item=QGraphicsTextItem(strText)
      font=self.font()
      font.setPointSize(20)
      font.setBold(True)
      item.setFont(font)   #设置字体
      item.setDefaultTextColor(Qt.black)  #设置颜色
      self.__setItemProperties(item,"文字")

##=============图件的编辑操作===================
   @pyqtSlot()    ##放大
   def on_actZoomIn_triggered(self):
      items=self.scene.selectedItems()    # QGraphicsItem的列表
      cnt=len(items)  #选中的图形项的个数
      if cnt==1:  #只有一个图形项
         item=items[0]
         item.setScale(0.1+item.scale())
      else:
         self.view.scale(1.1,1.1)

   @pyqtSlot()    ##缩小
   def on_actZoomOut_triggered(self):
      items=self.scene.selectedItems()    # QGraphicsItem的列表
      cnt=len(items)    #选中的图形项的个数
      if cnt==1:        #只有一个图形项
         item=items[0]
         item.setScale(item.scale()-0.1)
      else:
         self.view.scale(0.9,0.9)   
      
   @pyqtSlot()    ##"恢复"取消所有缩放和旋转变换
   def on_actRestore_triggered(self):
      items=self.scene.selectedItems()    # QGraphicsItem的列表
      cnt=len(items)
      if cnt==1:  #单个图形项
         item=items[0]
         item.setScale(1)        #缩放还原
         item.setRotation(0)     #旋转还原
##         item.resetTransform()   #不起作用,BUG, PyQt 5.12
      else:
         self.view.resetTransform()
      
   @pyqtSlot()    ##左旋转
   def on_actRotateLeft_triggered(self):
      items=self.scene.selectedItems()    # QGraphicsItem的列表
      cnt=len(items)
      if cnt==1:
         item=items[0]
         item.setRotation(-30+item.rotation())
      else:
         self.view.rotate(-30)

   @pyqtSlot()    ##右旋转
   def on_actRotateRight_triggered(self):
      items=self.scene.selectedItems()    # QGraphicsItem的列表
      cnt=len(items)
      if cnt==1:
         item=items[0]
         item.setRotation(30+item.rotation())
      else:
         self.view.rotate(30)

   @pyqtSlot()    ##前置
   def on_actEdit_Front_triggered(self):
      items=self.scene.selectedItems()    # QGraphicsItem的列表
      cnt=len(items)
      if cnt>0 :
         item=items[0]
         self.__frontZ=1+self.__frontZ
         item.setZValue(self.__frontZ)
      
   @pyqtSlot()    ##后置
   def on_actEdit_Back_triggered(self):
      items=self.scene.selectedItems()    # QGraphicsItem的列表
      cnt=len(items)
      if cnt>0 :
         item=items[0]
         self.__backZ=self.__backZ-1
         item.setZValue(self.__backZ)

   @pyqtSlot()    ##组合
   def on_actGroup_triggered(self):
      items=self.scene.selectedItems()    # QGraphicsItem的列表
      cnt=len(items)  #选中的图形项个数
      if (cnt<=1):
        return 

      group =QGraphicsItemGroup() #创建组合
      self.scene.addItem(group)   #组合添加到场景中
      for i in range(cnt):
         item=items[i]
         item.setSelected(False)    #清除选择虚线框
         item.clearFocus()
         group.addToGroup(item)     #添加到组合

      group.setFlag(QGraphicsItem.ItemIsFocusable)
      group.setFlag(QGraphicsItem.ItemIsMovable)
      group.setFlag(QGraphicsItem.ItemIsSelectable)

      self.__frontZ=1+self.__frontZ
      group.setZValue(self.__frontZ)
      self.scene.clearSelection()
      group.setSelected(True)

   @pyqtSlot()    ##打散组合
   def on_actGroupBreak_triggered(self):
      items=self.scene.selectedItems()   
      cnt=len(items)
      if (cnt==1):  #假设选中的是QGraphicsItemGroup
         group=items[0]
         self.scene.destroyItemGroup(group)

   @pyqtSlot()    ##删除所有选中的绘图项
   def on_actEdit_Delete_triggered(self):
      items=self.scene.selectedItems()   
      cnt=len(items)
      for i in range(cnt):
         item=items[i]
         self.scene.removeItem(item)   #删除绘图项

        
##  =============自定义槽函数===============================        
   def  do_mouseMove(self,point):      ##鼠标移动
   ##鼠标移动事件，point是 GraphicsView的坐标,物理坐标
      self.__labViewCord.setText("View 坐标：%d,%d" %(point.x(),point.y()))
      pt=self.view.mapToScene(point)  #转换到Scene坐标
      self.__labSceneCord.setText("Scene 坐标：%.0f,%.0f"%(pt.x(),pt.y()))
   
   def  do_mouseClicked(self,point):   ##鼠标单击
      pt=self.view.mapToScene(point)  #转换到Scene坐标
      item=self.scene.itemAt(pt,self.view.transform())  #获取光标下的图形项
      if (item == None): 
         return
      pm=item.mapFromScene(pt)   #转换为绘图项的局部坐标
      self.__labItemCord.setText("Item 坐标：%.0f,%.0f"%(pm.x(),pm.y()))
      self.__labItemInfo.setText(str(item.data(self.__ItemDesc))
                    +", ItemId="+str(item.data(self.__ItemId)))

   def do_mouseDoubleClick(self,point):  ##鼠标双击
      pt=self.view.mapToScene(point)   #转换到Scene坐标,QPointF
      item=self.scene.itemAt(pt,self.view.transform())    #获取光标下的绘图项
      if (item == None): 
         return

      className=str(type(item))  #将类名称转换为字符串
   ##      print(className)
      
      if (className.find("QGraphicsRectItem") >=0):      #矩形框
         self.__setBrushColor(item)
      elif (className.find("QGraphicsEllipseItem")>=0):  #椭圆和圆都是 QGraphicsEllipseItem
         self.__setBrushColor(item)
      elif (className.find("QGraphicsPolygonItem")>=0):  #梯形和三角形
         self.__setBrushColor(item)
      elif (className.find("QGraphicsLineItem")>=0):     #直线，设置线条颜色
         pen=item.pen()
         color=item.pen().color()
         color=QColorDialog.getColor(color,self,"选择线条颜色")
         if color.isValid():
            pen.setColor(color)
            item.setPen(pen)
      elif (className.find("QGraphicsTextItem")>=0):     #文字，设置字体
         font=item.font()
         font,OK=QFontDialog.getFont(font)
         if OK:
            item.setFont(font)

   def do_keyPress(self,event):     ##按键操作
      items=self.scene.selectedItems() 
      cnt=len(items)
      if (cnt!=1):  
         return

      item=items[0]
      key=event.key()
      if (key==Qt.Key_Delete):       #删除
         self.scene.removeItem(item)
      elif (key==Qt.Key_Space):      #顺时针旋转90度
         item.setRotation(90+item.rotation())
      elif (key==Qt.Key_PageUp):     #放大
         item.setScale(0.1+item.scale())
      elif (key==Qt.Key_PageDown):  #缩小
         item.setScale(-0.1+item.scale())
      elif (key==Qt.Key_Left):      #左移
         item.setX(-1+item.x())
      elif (key==Qt.Key_Right):     #右移
         item.setX(1+item.x())
      elif (key==Qt.Key_Up):        #上移
         item.setY(-1+item.y())
      elif (key==Qt.Key_Down):      #下移
         item.setY(1+item.y())
   
   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
    app = QApplication(sys.argv)    #创建GUI应用程序
    form=QmyMainWindow()            #创建窗体
    form.show()
    sys.exit(app.exec_())
