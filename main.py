from PyQt5.QtWidgets import QApplication, QDialog
import sys

from MainWindow import LoadWINDOW


from login.login import LoginWINDOW

from qt_material import apply_stylesheet


import qdarkstyle
from qdarkstyle import LightPalette


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 应用 qdarkstyle 样式到整个应用程序
    app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette))


    #载入登录界面
    login = LoginWINDOW()
    if login.exec_() == QDialog.Accepted:

        # 登录成功，获取用户名
        current_user = login.current_user
        print(f"登陆成功！当前登录用户: {current_user}")

        #app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette))
        main_window = LoadWINDOW(username=current_user)
        main_window.show()
        sys.exit(app.exec_())
    else:
        # 登录失败或取消，直接退出
        sys.exit()



    # load = LoadWINDOW()
    
    # app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette))


    sys.exit(app.exec_())
    