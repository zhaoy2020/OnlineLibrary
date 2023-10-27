import sys

from PyQt5.QtWidgets import  (QApplication, QMainWindow, QGraphicsScene,
         QStatusBar,QWidget, QVBoxLayout, QGroupBox, QLabel,QGraphicsView, 
         QGraphicsItem,QGraphicsRectItem, QGraphicsEllipseItem)

from PyQt5.QtCore import  pyqtSlot,Qt,QRectF

from PyQt5.QtGui import  QPen,QBrush

from myGraphicsView import QmyGraphicsView   #自定义视图类

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体

      self.__buildUI()            #构造界面
      self.__iniGraphicsSystem()  #初始化 graphics View系统
      
      self.view.mouseMove.connect(self.do_mouseMovePoint)   #鼠标移动
      self.view.mouseClicked.connect(self.do_mouseClicked)  #左键按下

##  ==============自定义功能函数============
   def __buildUI(self):   ##构造界面
      self.resize(600,450)
      self.setWindowTitle("Demo8_5, View/Scene/Item关系和坐标变换")

      font=self.font()
      font.setPointSize(11)
      self.setFont(font)
      
      centralWidget =QWidget(self)   #中间工作区组件
      vLayoutMain =QVBoxLayout(centralWidget)  #垂直布局
      
      groupBox = QGroupBox(centralWidget)    #显示两个Label的groupBox
      vLayoutGroup = QVBoxLayout(groupBox)
      self.__labViewSize = QLabel(groupBox)
      self.__labViewSize.setText("view坐标，左上角(0,0)，宽度=，长度=")
      
      vLayoutGroup.addWidget(self.__labViewSize)
      self.__labSceneRect = QLabel(groupBox)
      self.__labSceneRect.setText("view.sceneRect=()")
      vLayoutGroup.addWidget(self.__labSceneRect)
      
      vLayoutMain.addWidget(groupBox)   #主布局添加groupBox
        
      self.view = QmyGraphicsView(centralWidget)  #绘图视图
      self.view.setCursor(Qt.CrossCursor)
      self.view.setMouseTracking(True)
   ##      self.view.setDragMode(QGraphicsView.RubberBandDrag)
      vLayoutMain.addWidget(self.view) #添加到主布局

      self.setCentralWidget(centralWidget)  #设置工作区中间组件
      
      statusBar = QStatusBar(self)  #状态栏
      self.setStatusBar(statusBar)

      self.__labViewCord=QLabel("View 坐标：")
      self.__labViewCord.setMinimumWidth(150)
      statusBar.addWidget(self.__labViewCord)

      self.__labSceneCord=QLabel("Scene 坐标：")
      self.__labSceneCord.setMinimumWidth(150)
      statusBar.addWidget(self.__labSceneCord)

      self.__labItemCord=QLabel("Item 坐标：")
      self.__labItemCord.setMinimumWidth(150)
      statusBar.addWidget(self.__labItemCord)


   def __iniGraphicsSystem(self):   ##初始化 graphics View系统
      rect=QRectF(-200,-100,400,200)  
      self.scene=QGraphicsScene(rect)    #scene逻辑坐标系定义
      self.view.setScene(self.scene)

   ## 画一个矩形框，大小等于scene
      item=QGraphicsRectItem(rect)  #矩形框正好等于scene的大小
      item.setFlag(QGraphicsItem.ItemIsSelectable) #可选，
      item.setFlag(QGraphicsItem.ItemIsFocusable)  #可以有焦点
      pen=QPen()
      pen.setWidth(2)
      item.setPen(pen)
      self.scene.addItem(item)

   ##一个位于scene中心的椭圆，测试局部坐标
   #矩形框内创建椭圆,绘图项的局部坐标，左上角(-100,-50)，宽200，高100
      item2= QGraphicsEllipseItem(-100,-50,200,100) 
      item2.setPos(0,0)
      item2.setBrush(QBrush(Qt.blue))
      item2.setFlag(QGraphicsItem.ItemIsSelectable) #可选，
      item2.setFlag(QGraphicsItem.ItemIsFocusable)  #可以有焦点
      item2.setFlag(QGraphicsItem.ItemIsMovable)    #可移动

      self.scene.addItem(item2)

   ##一个圆，中心位于scene的边缘
      item3=QGraphicsEllipseItem(-50,-50,100,100)   #矩形框内创建椭圆,绘图项的局部坐标
      item3.setPos(rect.right(),rect.bottom())
      item3.setBrush(QBrush(Qt.red))
      item3.setFlag(QGraphicsItem.ItemIsSelectable) #可选，
      item3.setFlag(QGraphicsItem.ItemIsFocusable)  #可以有焦点
      item3.setFlag(QGraphicsItem.ItemIsMovable)    #可移动
      self.scene.addItem(item3)
      
      self.scene.clearSelection()

##==============event 处理函数=======================
   def resizeEvent(self,event):
      self.__labViewSize.setText('view坐标，左上角(0,0)，宽度=%d，高度=%d'
                                 %(self.view.width(),self.view.height()))

      rectF=self.view.sceneRect()   #Scene的矩形区,QRectF
      self.__labSceneRect.setText('view.sceneRect=(%.0f,%.0f,%.0f,%.0f)'
             %(rectF.left(),rectF.top(),rectF.width(),rectF.height()))

        
##  ==========由connectSlotsByName() 自动连接的槽函数==================        
    
        
        
##  =============自定义槽函数===============================        
   def  do_mouseMovePoint(self,point):
   ##鼠标移动事件，point是 GraphicsView的坐标,物理坐标
      self.__labViewCord.setText("View坐标：%d,%d" %(point.x(),point.y()))
      pt=self.view.mapToScene(point)  #转换到Scene坐标
      self.__labSceneCord.setText("Scene坐标：%.0f,%.0f"%(pt.x(),pt.y()))

   def  do_mouseClicked(self,point):
      pt=self.view.mapToScene(point)  #转换到Scene坐标
      item=None
      item=self.scene.itemAt(pt,self.view.transform())  #获取光标下的绘图项
      if (item != None): #有绘图项
         pm=item.mapFromScene(pt)   #转换为绘图项的局部坐标
         self.__labItemCord.setText("Item坐标：%.0f,%.0f"%(pm.x(),pm.y()))

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":         #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
