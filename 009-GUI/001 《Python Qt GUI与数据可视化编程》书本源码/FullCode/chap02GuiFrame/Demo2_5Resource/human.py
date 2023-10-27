# 自定义信号与槽的演示
import sys

from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal

class Human(QObject):
## 定义一个带str类型参数的信号
   nameChanged = pyqtSignal(str)
    
##    overload型信号，两种参数，一种int，一种str
   ageChanged = pyqtSignal([int],[str])

   def __init__(self,name='Mike',age=10,parent=None):
      super().__init__(parent)      # 调用父类构造函数
      self.setAge(age)
      self.setName(name)

   def   setAge(self,age):
      self.__age= age
      self.ageChanged.emit(self.__age)    # int参数信号
        
      if age<=18:
         ageInfo="你是 少年"
      elif (18< age <=35):
         ageInfo="你是 年轻人"
      elif (35< age <=55):
         ageInfo="你是 中年人"
      elif (55< age <=80):
         ageInfo="您是 老人"
      else:
         ageInfo="您是 寿星啊"
         
      self.ageChanged[str].emit(ageInfo)  #str参数信号
        
   def setName(self,name):
      self.__name = name
      self.nameChanged.emit(self.__name)


class Responsor(QObject):
   @pyqtSlot(int)
   def do_ageChanged_int(self,age):
      print("你的年龄是："+str(age))

   @pyqtSlot(str)
   def do_ageChanged_str(self,ageInfo):
      print(ageInfo)

#   @pyqtSlot(str)
   def do_nameChanged(self, name):
      print("Hello,"+name)

  
if  __name__ == "__main__":    #测试程序
   print("**创建对象时**")    
   boy=Human("Boy",16)   
   resp=Responsor()

   boy.nameChanged.connect(resp.do_nameChanged)

   # overload的信号如果都定义了槽函数，两个槽函数不能同名，连接时需要给信号加参数区分
   boy.ageChanged.connect(resp.do_ageChanged_int)        #缺省参数，int型
   boy.ageChanged[str].connect(resp.do_ageChanged_str)   # str型参数

   print("\n **建立连接后**")    
   boy.setAge(35)       #发射 两个ageChanged 信号
   boy.setName("Jack")  #发射nameChanged信号

   boy.ageChanged[str].disconnect(resp.do_ageChanged_str) #断开连接
   print("\n **断开ageChanged[str]的连接后**")    
   boy.setAge(10)   #发射 两个ageChanged 信号    
    
