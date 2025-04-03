from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QCursor
from PyQt5 import QtGui

from PyQt5.QtWidgets import QMessageBox

import psycopg2

import os
from .Ui_LoginWindow import Ui_LoginWindow
from .utils import BasicFunction



class LoginWINDOW(QDialog, Ui_LoginWindow):
    """界面逻辑"""

    current_user = None # 存储当前登录的用户名

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        #self.ui = Ui_LoginWindow()
        self.setupUi(self)

        #基础控件功能设置
        self.pushButton.clicked.connect(self.showMinimized)  # 最小化
        self.pushButton_2.clicked.connect(self.close)  # 关闭窗口
        
        #隐藏窗口自带的标题栏
        self.setWindowFlags(Qt.FramelessWindowHint)
        #设置窗口背景透明
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.basic_function = BasicFunction(self)

        self.setWindowTitle("登录")

        self.setWindowFlag(Qt.WindowContextHelpButtonHint, on=False)  # 去掉 QDialog 帮助问号
        self.stackedWidget.setCurrentIndex(0)  # 默认登录页
        self.pushButtonLoginPage.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))  # 切换登录页
        self.pushButtonRegisterPage.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))  # 切换注册页
        self.pushButtonForget.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))  # 切换忘记密码页
    

        #登录页面
        self.pushButtonLogin.clicked.connect(self.login)


        #注册页面
        #发送验证码
        #self.pushButtonSend.clicked.connect(self.send_verify_code)


        self.pushButtonRegister.clicked.connect(self.register)

        #忘记密码页面
        #self.pushButtonSendVerify2.clicked.connect(self.send_verify_code)

        self.pushButtonForgetOk.clicked.connect(self.forget_password)

        # 窗口切换信号
        self.stackedWidget.currentChanged.connect(self.update_stacked_widget)
    
    def update_stacked_widget(self):
        #把每个输入框的内容清空
        self.lineEditUsername.clear()
        self.lineEditPassword.clear()
        self.lineEditRegisterName.clear()
        self.lineEditEmail.clear()
        self.lineEditRegisterPassword1.clear()
        self.lineEditRegisterPassword2.clear()
        self.lineEditVerify.clear()

    ## 登录按钮

    def login(self):
        username = self.lineEditUsername.text()
        password = self.lineEditPassword.text()

        # 连接数据库验证帐号密码
        user_info_dict = {}
        email_info_dict = {}
        conn = psycopg2.connect(database='tsf_database', user='postgres', password='123456', host='127.0.0.1', port='5432')
        cur = conn.cursor()
        cur.execute("select username, email, password from users;")  # 查询 username, email 和 password
        rows = cur.fetchall()
        for row in rows:
            user_info_dict[row[0]] = row[2]  # 使用 username 作为键，password 作为值
            email_info_dict[row[1]] = row[2]  # 使用 email 作为键，password 作为值
        conn.commit()
        conn.close()

        if len(username) == 0 or len(password) == 0:  # 账号或密码为空
            self.basic_function.info_message("用户名或密码错误")
            return
        elif user_info_dict.get(username) == password:  # 账号密码对应正确
            
            LoginWINDOW.current_user = username
            self.accept()
        elif email_info_dict.get(username) == password:  # 邮箱密码对应正确
            
            LoginWINDOW.current_user = username  # 此处仍然将输入的内容作为当前用户
            self.accept()
        else:  # 用户名或邮箱与密码均不匹配
            self.basic_function.info_message("用户名/邮箱或密码错误")


    # def login(self):

    #     username = self.ui.lineEditUsername.text()
    #     password = self.ui.lineEditPassword.text()

    #     # 连接数据库验证帐号密码
    #     user_info_dict = {}
    #     conn = psycopg2.connect(database='tsf_database', user='postgres', password='123456', host='127.0.0.1', port='5432')
    #     cur = conn.cursor()
    #     cur.execute("select username, password from users;")  # 查询 username 和 password
    #     rows = cur.fetchall()
    #     for row in rows:
    #         user_info_dict[row[0]] = row[1]  # 使用 username 作为键，password 作为值
    #     conn.commit()
    #     conn.close()

    #     if len(username) == 0 or len(password) == 0: # 账号或密码为空
    #         self.basic_function.info_message("用户名或密码错误")
    #         return
    #     elif user_info_dict.get(username) is None: # 账号不存在
    #         self.basic_function.info_message("用户名不存在")
    #         return
    #     elif user_info_dict[username][0] == password: # 账号密码对应正确
    #         global current_user
    #         current_user = username
    #         self.accept()
            
        

    ## 注册按钮
    def register(self):
        #registername = self.ui.lineEditRegisterName.text()
        registername = self.lineEditRegisterName.text()
        password_1 = self.lineEditRegisterPassword1.text()
        password_2 = self.lineEditRegisterPassword2.text()
        email = self.lineEditEmail.text()

        if (len(registername) == 0):
            self.basic_function.info_message("用户名不能为空")
            return
        elif (len(email) == 0):
            self.basic_function.info_message("邮箱不能为空")
            return
        elif (len(password_1) == 0 or len(password_2) == 0):
            self.basic_function.info_message("密码不能为空")
            return
        elif password_1 != password_2:
            self.basic_function.info_message("两次密码不一致")
            return           
        else: # 注册成功
            conn = psycopg2.connect(database='tsf_database', user='postgres', password='123456', host='127.0.0.1',
                                    port='5432')
            cur = conn.cursor()
            cur.execute(f"insert into users values ('{registername}', '{email}','{password_1}');")
            conn.commit()
            conn.close()
        
        # 注册成功后，判断是否选中直接登录,若未选中，则切换会登录页
        if self.checkBox_2.isChecked():
            self.accept()
        else:
            self.stackedWidget.setCurrentIndex(0)
            self.basic_function.info_message("注册成功，请登录")

    ## 忘记密码按钮
    def forget_password(self):
        username = self.lineEditForgetName.text()
        NewPassword = self.lineEditNewPassword.text()
        NewPassword2 = self.lineEditNewPassword2.text()

        if len(username) == 0 :
            self.basic_function.info_message("请填写正确的用户名")
            return
        elif len(NewPassword) == 0 or len(NewPassword2) == 0:
            self.basic_function.info_message("新密码不能为空")
            return
        elif NewPassword != NewPassword2:
            self.basic_function.info_message("两次密码不一致")
            return
        
        # 判断是否选中找回密码后直接登录,若未选中，则切换会登录页
        if self.checkBox_3.isChecked():
            self.accept()
        else:
            self.stackedWidget.setCurrentIndex(0)




    
    #发送验证码函数
    def send_verify_code(self):
        # 这里可以添加发送验证码的逻辑
        # 例如，使用 smtplib 发送邮件或使用第三方库发送短信验证码等
        self.basic_function.info_message("验证码已发送，请注意查收！")


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))