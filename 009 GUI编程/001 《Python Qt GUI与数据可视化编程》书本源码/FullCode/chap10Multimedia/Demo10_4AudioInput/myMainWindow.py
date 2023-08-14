import sys

from PyQt5.QtWidgets import  QApplication, QMainWindow,QMessageBox

from PyQt5.QtCore import  pyqtSlot,pyqtSignal, QIODevice,QFile

##from PyQt5.QtWidgets import  

##from PyQt5.QtGui import

##from PyQt5.QtSql import 

from PyQt5.QtMultimedia import QAudioDeviceInfo, QAudio,QAudioFormat,QAudioInput

from ui_MainWindow import Ui_MainWindow

##from myAudioReader import QmyAudioReader

class QmyMainWindow(QMainWindow): 

   def __init__(self, parent=None):
      super().__init__(parent)    #调用父类构造函数，创建窗体
      self.ui=Ui_MainWindow()     #创建UI对象
      self.ui.setupUi(self)       #构造UI界面

      self.ui.progBar_Max.setMaximum(256)
      self.ui.progBar_Min.setMaximum(256)
      self.ui.progBar_Diff.setMaximum(256)

      self.ui.sliderVolumn.setMaximum(100)
      self.ui.sliderVolumn.setValue(100)

      self.ui.comboDevices.clear()
      self.__deviceList=QAudioDeviceInfo.availableDevices(QAudio.AudioInput) #音频输入设备列表
      for i in range(len(self.__deviceList)):
         device=self.__deviceList[i]    #QAudioDeviceInfo类
         self.ui.comboDevices.addItem(device.deviceName())

##      self.__deviceInfo =None   #当前设备信息,QAudioDeviceInfo
      self.audioDevice=None       #音频输入设备,QAudioInput
      self.BUFFER_SIZE=4000

      self.ioDevice=None      #第1种读取方法，内建的IODevice

##      self.externalReader=None        #第2种读取方法，外建的IODevice

      self.recordFile=QFile()      #第3种读取方法，使用QFile直接写入文件

      if len(self.__deviceList)>0:
         self.ui.comboDevices.setCurrentIndex(0) #触发comboDevices的信号currentIndexChanged()
##         self.__deviceInfo =deviceList[0]
      else:
         self.ui.actStart.setEnabled(False)
         self.ui.actDeviceTest.setEnabled(False)
         self.ui.groupBoxDevice.setTitle("支持的音频输入设置(无设备)")
      

##  ==============自定义功能函数========================
   def __getSampleTypeStr(self,sampleType):
      result="Unknown"
      if sampleType==QAudioFormat.SignedInt:
         result = "SignedInt"
      elif sampleType==QAudioFormat.UnSignedInt:
         result = "UnSignedInt"
      elif sampleType==QAudioFormat.Float:
         result = "Float"
      elif sampleType==QAudioFormat.Unknown:
         result = "Unknown"
         
      return result

   def __getByteOrderStr(self,endian):
      if (endian==QAudioFormat.LittleEndian):
         return "LittleEndian"
      else:
         return "BigEndian"
      

##  ==============event处理函数==========================
        
        
##  ==========由connectSlotsByName()自动连接的槽函数============
      
   @pyqtSlot(int)   ##选择音频输入设备
   def on_comboDevices_currentIndexChanged(self,index):        
      deviceInfo =self.__deviceList[index]   #当前音频设备,QAudioDeviceInfo类型
      self.ui.comboCodec.clear()             
      codecs = deviceInfo.supportedCodecs()  #支持的音频编码，字符串列表
      for strLine in codecs:
         self.ui.comboCodec.addItem(strLine)

      self.ui.comboSampleRate.clear()  #支持的采样率
      sampleRates = deviceInfo.supportedSampleRates()  #QList<int>
      for i in  sampleRates:
         self.ui.comboSampleRate.addItem("%d"% i)

      self.ui.comboChannels.clear()    #支持的通道数
      Channels = deviceInfo.supportedChannelCounts() #QList<int> 
      for i in Channels:
         self.ui.comboChannels.addItem("%d"%i )

      self.ui.comboSampleTypes.clear() #支持的采样点类型
      sampleTypes = deviceInfo.supportedSampleTypes() #QList<QAudioFormat::SampleType>
      for i in  sampleTypes:
         sampTypeStr=self.__getSampleTypeStr(i)
         self.ui.comboSampleTypes.addItem(sampTypeStr,i)
         
      self.ui.comboSampleSizes.clear() #采样点大小
      sampleSizes = deviceInfo.supportedSampleSizes() #QList<int>
      for i in  sampleSizes:
         self.ui.comboSampleSizes.addItem("%d"%i)

      self.ui.comboByteOrder.clear() #字节序
      endians = deviceInfo.supportedByteOrders()  #QList<QAudioFormat::Endian> 
      for i in endians:
         self.ui.comboByteOrder.addItem(self.__getByteOrderStr(i))

   @pyqtSlot()   ##使用内建IODevice
   def on_radioSaveMode_Inner_clicked(self):
      self.ui.groupBox_disp.setVisible(True)
      
   @pyqtSlot()   ##使用QFile对象(test.raw)
   def on_radioSaveMode_QFile_clicked(self):
      self.ui.groupBox_disp.setVisible(False)

   @pyqtSlot(int)   ##调节录音音量
   def on_sliderVolumn_valueChanged(self,value):
      self.ui.LabVol.setText("录音音量(%d%%)"%value)


   @pyqtSlot()   ##测试音频输入设备是否支持选择的设置
   def on_actDeviceTest_triggered(self):
      settings=QAudioFormat()
      settings.setCodec(self.ui.comboCodec.currentText())
      settings.setSampleRate(int(self.ui.comboSampleRate.currentText()))
      settings.setChannelCount(int(self.ui.comboChannels.currentText()))

      k=self.ui.comboSampleTypes.currentData()
      settings.setSampleType(k)  #QAudioFormat.SampleType

      settings.setSampleSize(int(self.ui.comboSampleSizes.currentText()))

      if (self.ui.comboByteOrder.currentText()=="LittleEndian"):
         settings.setByteOrder(QAudioFormat.LittleEndian)
      else:
         settings.setByteOrder(QAudioFormat.BigEndian)

      index=self.ui.comboDevices.currentIndex()
      deviceInfo =self.__deviceList[index]  #当前音频设备

      if deviceInfo.isFormatSupported(settings):
         QMessageBox.information(self,"消息","测试成功，设备支持此设置")
      else:
         QMessageBox.critical(self,"错误","测试失败，设备不支持此设置")

   @pyqtSlot()   ##开始音频输入
   def on_actStart_triggered(self):
      audioFormat=QAudioFormat() #使用固定格式
      audioFormat.setSampleRate(8000)
      audioFormat.setChannelCount(1)
      audioFormat.setSampleSize(8)
      audioFormat.setCodec("audio/pcm")
      audioFormat.setByteOrder(QAudioFormat.LittleEndian)
      audioFormat.setSampleType(QAudioFormat.UnSignedInt)

      index=self.ui.comboDevices.currentIndex()
      deviceInfo =self.__deviceList[index]  #当前音频设备

      if (False== deviceInfo.isFormatSupported(audioFormat)):
         QMessageBox.critical(self,"错误","测试失败，输入设备不支持此设置")
         return

      self.audioDevice = QAudioInput(deviceInfo,audioFormat) #音频输入设备
      self.audioDevice.setBufferSize(self.BUFFER_SIZE)   #设置的缓冲区大小,字节数，并不一定等于实际数据块大小
      self.audioDevice.stateChanged.connect(self.do_stateChanged)  #状态变化


   ##1. 使用 start()->QIODevice 启动，返回的内置的IODevice, pull mode，利用readyRead()信号读出数据
      if self.ui.radioSaveMode_Inner.isChecked():
         self.ioDevice=self.audioDevice.start()  #返回内建的IODevice
         self.ioDevice.readyRead.connect(self.do_IO_readyRead)

   ## 2. 自定义流设备QWAudioBlockReader，push mode, start(QIODevice)，不行
   ##      if self.ui.radioSaveMode_External.isChecked():
   ##         self.externalReader = QmyAudioReader()
   ##         self.externalReader.open(QIODevice.WriteOnly)
   ##         self.externalReader.updateBlockInfo.connect(self.do_updateBlockInfo)
   ##         self.audioDevice.start(self.externalReader)  #使用外建的IODevice
         
      
   ##3. 写入文件,用 start(QIODevice)启动
      if self.ui.radioSaveMode_QFile.isChecked():
         self.recordFile.setFileName("test.raw") 
         self.recordFile.open(QIODevice.WriteOnly)
         self.audioDevice.start(self.recordFile)  


   @pyqtSlot()   ##停止音频输入
   def on_actStop_triggered(self):
      self.audioDevice.stop()
      self.audioDevice.deleteLater()
      
   ##1. 使用 QIODevice  =start()返回的内置的IODevice, pull mode，利用readyRead()信号读出数据
      #无需处理,停止后self.ioDevice自动变为无效

   ##2. 外部IODevice接收音频输入数据的流设备
   ##      if self.ui.radioSaveMode_External.isChecked():
   ##         self.externalReader.close()
   ##         self.externalReader.updateBlockSize.disconnect(self.do_updateBlockInfo)
   ##         del self.externalReader  #删除外建设备
      
   ##3. 写入文件,start(QIODevice) 启动
      if self.ui.radioSaveMode_QFile.isChecked():
         self.recordFile.close() #关闭文件
      

        
##  =============自定义槽函数===============================        
##1. 使用QIODevice* start()返回的内置的IODevice, pull mode，利用readyRead()信号读出数据
   def do_IO_readyRead(self):    ##内建IODevice，读取缓冲区数据
      self.ui.LabBufferSize.setText("bufferSize()=%d"
                                    %self.audioDevice.bufferSize())

      byteCount = self.audioDevice.bytesReady() #可以读取的字节数
      self.ui.LabBytesReady.setText("bytesReady()=%d"%byteCount)

      if byteCount>self.BUFFER_SIZE:
         byteCount=self.BUFFER_SIZE
      
      buffer=self.ioDevice.read(byteCount)  #返回的是bytes类型，便于处理
   ##      buffer=self.ioDevice.readAll()  #不能用readAll()读取，返回的是QByteArray,需要再转换为bytes
      
   ##      print(type(buffer))
   ##      print(buffer)
      
      maxSize=len(buffer)
   ##      print(maxSize)

      self.ui.LabBlockSize.setText("IODevice数据字节数=%d"%maxSize)

      maxV=0
      minV=255
      for k in range(maxSize):
         V=buffer[k]  #取一个字节,整数
         if V>maxV:
            maxV=V   #求最大值
         if V<minV:
            minV=V   #求最小值

      self.ui.progBar_Max.setValue(maxV)
      self.ui.progBar_Min.setValue(minV)
      self.ui.progBar_Diff.setValue(maxV-minV)


   def do_stateChanged(self,state):    ##设备状态变化
      isStoped=(state== QAudio.StoppedState) #停止状态
      self.ui.groupBox_saveMode.setEnabled(isStoped)
      self.ui.sliderVolumn.setEnabled(isStoped)
      self.ui.actStart.setEnabled(isStoped)
      self.ui.actStop.setEnabled(not isStoped)
      self.ui.actDeviceTest.setEnabled(isStoped)
      self.ui.sliderVolumn.setEnabled(isStoped)

      if  state== QAudio.ActiveState:
         self.ui.statusBar.showMessage("state: ActiveState")
      elif state== QAudio.SuspendedState:
         self.ui.statusBar.showMessage("state: SuspendedState")
      elif state== QAudio.StoppedState:
         self.ui.statusBar.showMessage("state: StoppedState")
      elif state== QAudio.IdleState:
         self.ui.statusBar.showMessage("state: IdleState")
      elif state== QAudio.InterruptedState:
         self.ui.statusBar.showMessage("state: InterruptedState")
         

##   def do_updateBlockInfo(self,blockSize, maxValue, minValue ):
##      self.ui.LabBufferSize.setText("bufferSize()=%d"%self.audioDevice.bufferSize())
##
##      byteCount = self.audioDevice.bytesReady() #可以读取的字节数, unicode编码
##      self.ui.LabBytesReady.setText("bytesReady()=%d"%byteCount)
##
##      self.ui.LabBlockSize.setText("IODevice读取数据字节数=%d"%blockSize)
##
##      self.ui.progBar_Max.setValue(maxValue)
##      self.ui.progBar_Min.setValue(minValue)
##      self.ui.progBar_Diff.setValue(maxValue-minValue)

   
##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())
