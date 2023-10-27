## 自定义类 QmyFigureCanvas，父类QWidget
## 创建了FigureCanvas和NavigationToolbar，组成一个整体
## 便于可视化设计


from PyQt5.QtWidgets import  QWidget

import matplotlib as mpl

from  matplotlib.backends.backend_qt5agg import (FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar)

from PyQt5.QtWidgets import   QVBoxLayout

class QmyFigureCanvas(QWidget):
   
   def __init__(self, parent=None, toolbarVisible=True,showHint=False):
      super().__init__(parent)
      pass


##=====公共接口函数
   def setToolbarVisible(self,isVisible=True):  ##是否显示工具栏
      pass


   def setDataHintVisible(self,isVisible=True): ##是否显示工具栏最后的坐标提示标签
      pass

      
   def redraw(self):    ##重绘曲线,快捷调用
      pass

   
   def __changeActionLanguage(self):   ##汉化工具栏
      pass


   def do_scrollZoom(self,event):   ##通过鼠标滚轮缩放
      pass

