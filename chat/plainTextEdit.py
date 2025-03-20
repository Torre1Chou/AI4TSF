from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtCore import Qt

class MyPlainTextEdit(QPlainTextEdit):  #父类为QPlainTextEdit

    def __init__(self,parent=None,main_window=None): #初始化函数
        super(MyPlainTextEdit, self).__init__(parent)
        self.main_window = main_window

    def keyPressEvent(self, event: QKeyEvent): #重写keyPressEvent方法
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:#ctrl+回车
            self.insertPlainText('\n')                                              #添加换行
        elif self.toPlainText() and event.key() == Qt.Key_Return:  #回车
            self.send_function() # 调用 send 函数
        else:
            super().keyPressEvent(event)

    def send_function(self):
        #self.setEnabled(False)          #主函数使用undoAvailable监听信号
        self.setUndoRedoEnabled(False)  #设置焦点
        self.setUndoRedoEnabled(True)   #设置焦点
        self.main_window.ui.pushButton_send.click()