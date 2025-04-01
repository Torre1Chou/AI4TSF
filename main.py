from PyQt5.QtWidgets import QApplication
import sys

from MainWindow import LoadWINDOW
from Ui_mainWindow import Ui_MainWindow

from qt_material import apply_stylesheet

import qdarkstyle
from qdarkstyle import LightPalette


if __name__ == "__main__":
    app = QApplication(sys.argv)
    load = LoadWINDOW()

    #apply_stylesheet(app, theme='light_blue.xml')
    
    app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette))




    sys.exit(app.exec_())
    