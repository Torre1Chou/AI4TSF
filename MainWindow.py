import sys
from PyQt5 import QtGui,QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication,QMainWindow,QTextBrowser,QStackedWidget
from PyQt5 import QtWidgets

from PyQt5.QtChart import QChart ,QChartView,QLineSeries
from pyqt5Custom import DragDropFile

import pandas as pd
import numpy as np
import os
import html

from datetime import datetime
import one_rc

from Ui_mainWindow import Ui_MainWindow
import matplotlib.pyplot as plt
from draw import ImageFigures, plot_data, CustomChartView,plot_muldata,plot_predata
#from draw import plot_acf

from draw import ACFFigureCanvas,DecomFigureCanvas

from bayes_opt import BayesianOptimization


from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose,STL

from scipy.signal import detrend

from chat.SendMessage import  Avatar,BubbleMessage
from chat.SendMessage import ChatWidget
from chat.SendMessage import MessageType
from chat.SendMessage import Notice 

from forecast.prediction import get_predictions

from chat.plainTextEdit import MyPlainTextEdit
from chat.chatbot import chatbot,chatbot_file


class LoadWINDOW(QMainWindow,Ui_MainWindow):

    def __init__(self, username = None,parent=None):
        super().__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("AI4TSF")
        self.setWindowIcon(QIcon(":/3/res/数据质控.png"))
        
        #隐藏窗口自带的标题栏
        self.setWindowFlags(Qt.FramelessWindowHint)
        #设置窗口背景透明
        self.setAttribute(Qt.WA_TranslucentBackground)

        #把label_user的文字设置为当前登录的用户名
        if username:
            self.ui.label_user.setText(f"(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧  欢迎 {username}！")

        #基础控件功能设置
        self.ui.pushButton_minus.clicked.connect(self.minusUI)
        self.ui.pushButton_closev.clicked.connect(self.closeUI)

        #设置布局管理器
        data_layout = QtWidgets.QVBoxLayout(self.ui.widget_data_image)
        self.ui.widget_data_image.setLayout(data_layout)
        interpolate_layout = QtWidgets.QVBoxLayout(self.ui.widget_interpolate_image)
        self.ui.widget_interpolate_image.setLayout(interpolate_layout)


        rolling_layout = QtWidgets.QVBoxLayout(self.ui.widget_rolling_image)
        self.ui.widget_rolling_image.setLayout(rolling_layout)
        fft_layout = QtWidgets.QVBoxLayout(self.ui.widget_fft_image)
        self.ui.widget_fft_image.setLayout(fft_layout)

        predict_layout = QtWidgets.QVBoxLayout(self.ui.widget_predict_image)
        self.ui.widget_predict_image.setLayout(predict_layout)

        ACF_layout = QtWidgets.QVBoxLayout(self.ui.widget_ACF)
        self.ui.widget_ACF.setLayout(ACF_layout)
        PACF_layout = QtWidgets.QVBoxLayout(self.ui.widget_PACF)
        self.ui.widget_PACF.setLayout(PACF_layout)

        decom_ori_layout = QtWidgets.QVBoxLayout(self.ui.widget_decom_ori)
        self.ui.widget_decom_ori.setLayout(decom_ori_layout)
        decom_trend_layout = QtWidgets.QVBoxLayout(self.ui.widget_decom_trend)
        self.ui.widget_decom_trend.setLayout(decom_trend_layout)
        decom_season_layout = QtWidgets.QVBoxLayout(self.ui.widget_decom_season)
        self.ui.widget_decom_season.setLayout(decom_season_layout)
        decom_resid_layout = QtWidgets.QVBoxLayout(self.ui.widget_decom_resid)
        self.ui.widget_decom_resid.setLayout(decom_resid_layout)


        #使用pyqtchart绘图
        self.original_chart = QChart()
        self.original_chart_view = QChartView(self.original_chart)
        self.ui.widget_data_image.layout().addWidget(self.original_chart_view)
        
        self.interpolate_chart = QChart()
        self.interpolate_chart_view = QChartView(self.interpolate_chart)
        self.ui.widget_interpolate_image.layout().addWidget(self.interpolate_chart_view)

        self.rolling_chart = QChart()
        self.rolling_chart_view = QChartView(self.rolling_chart)
        self.ui.widget_rolling_image.layout().addWidget(self.rolling_chart_view)
        self.fft_chart = QChart()
        self.fft_chart_view = QChartView(self.fft_chart)
        self.ui.widget_fft_image.layout().addWidget(self.fft_chart_view)



        #设置文件拖拽组件

        #设置布局管理器
        dropfile_layout = QtWidgets.QVBoxLayout(self.ui.widget_dropfile)
        self.ui.widget_dropfile.setLayout(dropfile_layout)

        # 将 DragDropFile 组件添加到 widget_dropfile 中
        self.dropfile = DragDropFile()
        self.dropfile.setFixedSize(300, 180)  # 设置组件大小
        self.ui.widget_dropfile.layout().addWidget(self.dropfile)
        
        #dropfile_layout.addWidget(self.ui.file_drag)


        # 监听文件拖放事件
        self.dropfile.fileDropped.connect(self.on_file_dropped)

        # 设置 dropfile 的样式
        # self.dropfile.borderRadius = 10  # 设置圆角
        # self.dropfile.borderWidth = 2    # 设置边框宽度
        # self.dropfile.setStyleSheet("border: 2px dashed #888;")  # 设置边框样式



        self.file_path = None
        self.original_data = None  # 初始化 original_data 属性
        #self.original_image = ImageFigures(self.ui.widget_data_image)

        # self.original_image = plt.figure()
        
        # # 将 widget_data_image 转换为 FigureCanvas
        # self.canvas = FigureCanvas(self.original_image)
        # layout = QVBoxLayout()
        # layout.addWidget(self.canvas)
        # self.ui.widget_data_image.setLayout(layout)

        #绘图的x轴和y轴的标签
        self.x_label = None
        self.y_label = None


        #预测结果数据
        self.predicted_data = None
        self.predicted_chart = QChart()
        self.predicted_chart_view = QChartView(self.predicted_chart)
        self.ui.widget_predict_image.layout().addWidget(self.predicted_chart_view)


        
        #数据处理部分

        #加载文件
        self.ui.pushButton_loadcsv.clicked.connect(lambda: self.switch_page(0))
        #self.ui.pushButton_loadcsv.clicked.connect(self.load_csv_file)
        
        #初始化选择变量的combobox
        self.ui.comboBoxChoosecolumn.addItems([])

        #数据领域关键词加载
        self.dataField = ''


        #数据填补
        self.ui.pushButton_interpolate_data.clicked.connect(lambda:self.switch_page(1))
        self.ui.pushButton_interpolate.clicked.connect(self.interpolate_csv_data)
        
        #数据平滑去噪
        self.ui.pushButton_denoise_data.clicked.connect(lambda:self.switch_page(2))

        self.ui.pushButton_rolling.clicked.connect(self.smooth_csv_data)
        self.ui.pushButton_fft.clicked.connect(self.Denoise_csv_data)
        
        #self.ui.pushButton_denoise_data.clicked.connect(self.Denoise_csv_data)

        

        #预测部分

        #模型列表

        

#后续模型在这里全部给出，在后面进行交互选择
        self.ui.comboBoxPredictModel.addItems(['DeepSeek','ARIMA','LLMTime GPT-3.5','timeGPT','Leddam'])

        #self.models = ['ARIMA',]
        #self.models = ['LLMTime GPT-3.5']
        #self.models = ['DeepSeek']
        self.hypers = {}

        #差分检验
        self.ui.pushButton_steady.clicked.connect(lambda:self.switch_page(3))
        self.ui.pushButton_steady.clicked.connect(self.Stationarity_csv_data)

        #时序分解按钮
        self.ui.pushButton_decomposition.clicked.connect(lambda:self.switch_page(4))
        self.ui.pushButton_decomposition.clicked.connect(self.decomposition_csv_data)

        #预测数据
        self.ui.pushButton_predict.clicked.connect(lambda:self.switch_page(5))
        #self.ui.pushButton_predict.clicked.connect(self.predict_data)
         

        #开始预测按钮
        self.ui.pushButton_start_predict.clicked.connect(self.predict_data)

        #设置预测的频率
        self.ui.comboBoxFreq.addItems(['D','H','T','S','W','MS','Q'])

        #设置插值方式选择
        self.ui.comboBoxInterpolate.addItems(['linear','quadratic','cubic','spline','polynomial'])

        #rolling窗口大小输入框设置只读
        self.ui.textEditRolling.setReadOnly(True)
        #fft阈值输入框设置只读
        self.ui.textEditFFT.setReadOnly(True)

        #预测数据保存路径
        self.savePath = ''
        
        #路径文本输入框显示默认路径
        self.ui.lineEditSavepath.setText('D:/SaveCSV/predicted_data.csv')

        #AI聊天部分

        #气泡数量
        #self.sum=0
        #记录气泡
        self.bubblelist = []       
        #存储信息
        self.text = ""
        self.kpss_text = ""
        self.decom_text = ''


        TEXT = MessageType.Text
        IMAGE = MessageType.Image
        #头像
        self.iconChat = QtGui.QPixmap(":/3/res/icon_chat.png")
        self.iconUser = QtGui.QPixmap(":/3/res/icon_cat.png")


        #设置选择聊天助手的combobox
        self.ui.comboBoxChatModel.addItems(['DeepSeek', 'GPT-3.5',])


        # 获取现有的布局（如果存在）
        if self.ui.chatwidget.layout() is None:
            chat_layout = QtWidgets.QVBoxLayout(self.ui.chatwidget)
        else:
            chat_layout = self.ui.chatwidget.layout()

        # 创建聊天窗口
        self.chat_ui = ChatWidget()
        chat_layout.addWidget(self.chat_ui)

        

        #获取当前时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        

        time_message = Notice(str(timestamp))
        self.chat_ui.add_message_item(time_message)
    

        bubble_message = BubbleMessage('你好，我是AI4TSF，一个低门槛时序数据的预测系统，有什么可以帮助你的吗？', self.iconChat, Type=TEXT, is_send=False)
        self.chat_ui.add_message_item(bubble_message)

        # bubble_message = BubbleMessage('你好啊💖', self.iconUser, Type=TEXT, is_send=True)
        # self.chat_ui.add_message_item(bubble_message)



        #设置聊天窗口样式 隐藏滚动条
        self.ui.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 信号与槽

        
        #监听输入框
        #self.ui.plainTextEdit.undoAvailable.connect(self.Enter2Send) 

        scrollbar = self.ui.scrollArea.verticalScrollBar()
        scrollbar.rangeChanged.connect(self.adjustScrollToMaxValue)

        chat_layout.addWidget(self.chat_ui)

        #发送消息按钮函数绑定
        self.ui.pushButton_send.clicked.connect(self.create_bubble)

        
        # 使用 MyPlainTextEdit 作为文本编辑控件
        self.ui.plainTextEdit = MyPlainTextEdit(self.ui.centralwidget, main_window=self)
        #self.ui.plainTextEdit.setGeometry(QtCore.QRect(740, 670, 411, 89))
        self.ui.horizontalLayout_7.addWidget(self.ui.plainTextEdit)


        self.show()

    ###############################################################
    #以下是组件编辑部分
    ###############################################################
    def on_file_dropped(self, file):
        # 当文件被拖放时，触发此函数
        print(f"File dropped: {file.path}")
        # 文件拖拽之后，调用 load_csv_file 函数
        self.load_csv_file(file.path)
        
    #把界面最小化
    def minusUI(self):
        self.showMinimized()

    #关闭页面
    def closeUI(self):
        self.close()

    def switch_page(self, index):
            """切换页面的函数"""
            self.ui.stackedWidget.setCurrentIndex(index)

    # #图片在加载时的动画效果
    # def show_loading_message(self, message, chart_view):
    #     """显示加载提示，关联到特定chart view"""
    #     # 先隐藏之前的加载提示
    #     self.hide_loading_message()
        
    #     # 创建加载标签，父组件设置为chart view
    #     self.loading_label = QtWidgets.QLabel(chart_view)
    #     self.loading_label.setText(message)
    #     self.loading_label.setAlignment(QtCore.Qt.AlignCenter)
        
    #     # 设置样式
    #     self.loading_label.setStyleSheet("""
    #         QLabel {
    #             background-color: rgba(255, 255, 255, 220);
    #             border: 2px solid #cccccc;
    #             border-radius: 10px;
    #             padding: 20px;
    #             font-size: 16px;
    #             min-width: 200px;
    #             min-height: 100px;
    #         }
    #     """)
        
    #     # 添加gif动画
    #     loading_gif_path = ":/res/gif/loading_circle.gif"  # Qt资源路径
    #     if QtCore.QFile.exists(loading_gif_path):
    #         self.movie = QtGui.QMovie(loading_gif_path)
    #         if self.movie.isValid():
    #             self.loading_label.setMovie(self.movie)
    #             self.movie.setCacheMode(QtGui.QMovie.CacheAll)
    #             self.movie.start()
        
    #     # 使用布局管理器确保居中
    #     self.loading_label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
    #     layout = QtWidgets.QVBoxLayout(chart_view)
    #     layout.addWidget(self.loading_label, 0, QtCore.Qt.AlignCenter)
    #     chart_view.setLayout(layout)
        
        
    #     # 显示在最上层
    #     self.loading_label.raise_()
    #     self.loading_label.show()

    def show_loading_message(self, message, chart_view):
        """显示加载提示，确保每次都能正常显示"""
        # 立即隐藏之前的加载提示
        self.hide_loading_message()
        
        # 创建新的加载标签（不设置父组件）
        self.loading_label = QtWidgets.QLabel()
        self.loading_label.setText(message)
        self.loading_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # 设置样式
        self.loading_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 220);
                border: 2px solid #cccccc;
                border-radius: 10px;
                padding: 20px;
                font-size: 16px;
                font-family: 'SimHei';
                min-width: 200px;
                min-height: 100px;
            }
        """)
        
        # 加载动画资源
        loading_gif_path = ":/res/gif/loading_circle.gif"
        if QtCore.QFile.exists(loading_gif_path):
            self.movie = QtGui.QMovie(loading_gif_path)
            if self.movie.isValid():
                self.movie.setCacheMode(QtGui.QMovie.CacheAll)
                self.loading_label.setMovie(self.movie)
                self.movie.start()
        
        # 创建覆盖层并添加到chart_view
        self.loading_overlay = QtWidgets.QWidget(chart_view)
        self.loading_overlay.setStyleSheet("background-color: transparent;")
        self.loading_overlay.setGeometry(0, 0, chart_view.width(), chart_view.height())
        
        # 使用布局确保居中
        layout = QtWidgets.QVBoxLayout(self.loading_overlay)
        layout.addWidget(self.loading_label, 0, QtCore.Qt.AlignCenter)
        self.loading_overlay.setLayout(layout)
        
        # 显示在最上层
        self.loading_overlay.raise_()
        self.loading_overlay.show()
        
        # 强制立即更新UI
        QtWidgets.QApplication.processEvents()


    def hide_loading_message(self):
        """隐藏加载提示"""
        try:
            # 停止并删除动画
            if hasattr(self, 'movie'):
                movie = getattr(self, 'movie', None)
                if movie:
                    movie.stop()
                    movie.deleteLater()
                delattr(self, 'movie')
            
            # 删除标签
            if hasattr(self, 'loading_label'):
                label = getattr(self, 'loading_label', None)
                if label:
                    label.deleteLater()
                delattr(self, 'loading_label')

            # 删除覆盖层
            if hasattr(self, 'loading_overlay'):
                overlay = getattr(self, 'loading_overlay', None)
                if overlay:
                    overlay.deleteLater()
                delattr(self, 'loading_overlay')  
        except Exception as e:
            print(f"隐藏加载提示时出错: {e}")
        finally:
            # 确保处理所有待处理事件
            QtWidgets.QApplication.processEvents()
       
    ###############################################################
    #以下是数据处理部分
    ###############################################################



# #这里需要获取当前选择的chatbot

#         curr_model = 'DeepSeek'
#         #把文件内容发送给聊天机器人
#         response = chatbot_file(curr_model,self.file_path)

#         #把聊天机器人的回复生成气泡消息
#         bubble_message = BubbleMessage(response, self.iconChat, Type=MessageType.Text, is_send=False)
#         self.chat_ui.add_message_item(bubble_message)

# #文件日期格式不正确

    #涉及多变量的上传文件函数
    def load_csv_file(self, file_path):
        self.file_path = file_path
        file_name = os.path.basename(self.file_path)
        self.file_name = file_name[:-4]

        # 读取 CSV 文件
        data = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        self.original_data = data
        self.original_data.sort_index(inplace=True)

        # 设置 x 和 y 坐标名
        self.x_label = self.original_data.index.name
        self.y_label = "Value"  # 多变量时，y 轴统一命名为 "Value"

        # 准备 y 数据字典
        y_data_dict = {column: self.original_data[column] for column in self.original_data.columns}

        #把文件中的列数据除去时间列添加到combobox中
        self.ui.comboBoxChoosecolumn.addItems(self.original_data.columns)
        #减去时间列
        #self.ui.comboBoxChoosecolumn.removeItem(0)

        #提取数据关键词
        self.dataField = self.ui.textEditDataField.toPlainText().strip()
        print(f"提取的数据关键词: {self.dataField}")

        # 显示"正在绘图中"提示，关联到chart
        self.show_loading_message("正在绘图中...", self.original_chart_view)
        
        # 强制界面更新
        QtWidgets.QApplication.processEvents()

        
        try:
            # 调用 plot_muldata 函数绘制多变量数据
            plot_muldata(self.original_chart, self.original_data.index, y_data_dict, "Original Data", self.x_label, self.y_label)
        finally:
            # 绘图完成后隐藏提示
            self.hide_loading_message()

#为了节约时间先把机器人交互部分注释掉
        # # # 把文件内容发送给聊天机器人
        # curr_model = self.ui.comboBoxChatModel.currentText()

        # response = chatbot_file(curr_model, self.file_path,self.dataField)

        # # 把聊天机器人的回复生成气泡消息
        # bubble_message = BubbleMessage(response, self.iconChat, Type=MessageType.Text, is_send=False)
        # self.chat_ui.add_message_item(bubble_message)



    def initialize_predicted_data(self, predict_length):
        if self.original_data is not None:
            # 创建 original_data 的副本
            original_data_copy = self.original_data.copy()


            #使用前端选择的freq
            curr_freq = self.ui.comboBoxFreq.currentText()

            # 确保时间索引是规则的
            if original_data_copy.index.freq is None:
                # 生成规则的时间索引
                full_index = pd.date_range(start=original_data_copy.index[0], 
                                        end=original_data_copy.index[-1], 
                                        freq=curr_freq)
                # 重新索引并填充缺失值
                original_data_copy = original_data_copy.reindex(full_index, fill_value=0)
            
            # 预测长度
            predict_len = int(predict_length)

            # 获取最后一个日期
            last_date = original_data_copy.index[-1]
            
            # 生成新的日期范围，使用推断或手动设置的频率
            new_dates = pd.date_range(start=last_date, periods=predict_len + 1, freq=curr_freq)[1:]

            # 创建新的预测数据，初始值为0
            # zero_col = pd.Series([0 for index in range(predict_len)])
            # new_data = pd.Series(zero_col, index=new_dates)

            # 如果 original_data 是 DataFrame，则 predict_data 也应该是 DataFrame
            if isinstance(self.original_data, pd.DataFrame):
                # 创建一个与 original_data 列名相同的 DataFrame
                zero_data = {col: [0] * predict_len for col in self.original_data.columns}
                new_data = pd.DataFrame(zero_data, index=new_dates)
            else:
                # 如果 original_data 是 Series，则 predict_data 也是 Series
                zero_col = pd.Series([0 for _ in range(predict_len)], index=new_dates)
                new_data = zero_col
            
            # 预测数据
            self.predicted_data = new_data


            # 确保时间索引具有 freq 属性
            self.predicted_data.index.freq = curr_freq
        
        else:
            print("Original data is not loaded")




    #从文件中读取数据
    def read_data(self):
        if self.file_path:
            self.original_data = pd.read_csv(self.file_path, index_col=0, parse_dates=True, squeeze=True)
            self.original_data.sort_index(inplace=True)
            #给出载入数据的行列索引
            print(self.original_data.index)
        else:
            print("No file selected")
        #文件日期格式不正确


    


    #对数据进行插值处理
    def interpolate_csv_data(self):


        # 确保时间索引是规则的
        if self.original_data.index.freq is None:
            # 获取当前选择的频率
            curr_freq = self.ui.comboBoxFreq.currentText()

            # 生成规则的时间索引
            full_index = pd.date_range(start=self.original_data.index[0], 
                                    end=self.original_data.index[-1], 
                                    freq=curr_freq)

            # 重新索引并填充缺失值
            self.original_data = self.original_data.reindex(full_index)

        old_data = self.original_data.copy()



        #根据数据特征选择插值方法
        interpolated_method = self.ui.comboBoxInterpolate.currentText()

        interpolated_data = self.original_data.interpolate(method=interpolated_method,order =2)

        #更新数据
        self.original_data = interpolated_data

        # 显示"正在绘图中"提示，关联到_chart
        self.show_loading_message("正在绘图中...", self.interpolate_chart_view)

        # 准备 y 数据字典
        y_data_dict = {column: self.original_data[column] for column in self.original_data.columns}

        # 把插值前后数据信息打印到文本框中
        self.ui.textEditInformation.setHtml(
            "<h3 style='color: #1E90FF; font-family: 黑体;'>=== 插值前后数据对比 ===</h3>"
            "<p style='color: #1E90FF; font-family: 黑体;'>插值前数据摘要：</p>"
            f"<p style='color: #FFA500; font-family: 黑体;'>{old_data.describe().to_html()}</p>"
            "<p style='color: #1E90FF; font-family: 黑体;'>插值后数据摘要：</p>"
            f"<p style='color: #FFA500; font-family: 黑体;'>{self.original_data.describe().to_html()}</p>"
        )
        

        try:
            #绘制插值后的数据
            plot_muldata(self.interpolate_chart, self.original_data.index, y_data_dict, "Interpolated Data", self.x_label, self.y_label)

        finally:
            # 绘图完成后隐藏提示
            self.hide_loading_message()

#为了节约时间先把机器人交互部分注释掉
        # # 生成简短的插值结果摘要
        # summary_before = old_data.describe()
        # summary_after = interpolated_data.describe()


        # #把插值处理的结果封装成气泡消息发送给聊天机器人
        # message = f"数据插值处理完成，插值方法为 {interpolated_method}，\n插值前数据摘要：\n{summary_before}\n插值后数据摘要：\n{summary_after}，根据前后数据摘要对比分析插值方法带来的影响。"


        # curr_model = self.ui.comboBoxChatModel.currentText()

        # response = chatbot(curr_model,message,self.dataField)

        # #把聊天机器人的回复生成气泡消息
        # bubble_message = BubbleMessage(response, self.iconChat, Type=MessageType.Text, is_send=False)
        # self.chat_ui.add_message_item(bubble_message)


    #对数据进行平滑去噪处理
    def smooth_csv_data(self):
        #进行rolling平均去噪

        # 显示"正在绘图中"提示，关联到chart
        self.show_loading_message("正在绘图中...", self.rolling_chart_view)

        # 强制界面更新
        QtWidgets.QApplication.processEvents()

        #贝叶斯优化函数
        def objective_function(window_size):
            window_size = int(window_size)  # 窗口大小必须为整数
            # 使用rolling平滑
            smoothed_data = self.original_data.rolling(window=window_size, min_periods=1).mean()
            # 计算均方误差（MSE）
            mse = ((self.original_data - smoothed_data) ** 2).mean().mean()  # 多列数据取平均MSE
            return -mse  # 贝叶斯优化默认最大化目标函数，因此取负MSE

        # 定义窗口大小的搜索范围
        pbounds = {'window_size': (7, 200)}

        # 运行贝叶斯优化
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=42
        )
        optimizer.maximize(init_points=10, n_iter=50)  # 初始点5个，迭代20次

        # 获取最优窗口大小
        best_window_size = int(optimizer.max['params']['window_size'])
        print(f"最优窗口大小: {best_window_size}")

        #在界面上显示最佳窗口大小
        self.ui.textEditRolling.setText(str(best_window_size))

        

        smoothed_data = self.original_data.rolling(window=best_window_size,min_periods=1).mean()

        # 使用前向填充和线性插值填充无效值
        smoothed_data = smoothed_data.ffill().bfill().interpolate(method='linear')

        self.original_data = smoothed_data

        # 准备 y 数据字典
        y_data_dict = {column: self.original_data[column] for column in self.original_data.columns}


        

        try:
            plot_muldata(self.rolling_chart, self.original_data.index, y_data_dict, "Rolling Data", self.x_label, self.y_label)
        finally:
            # 绘图完成后隐藏提示
            self.hide_loading_message()
        

    #对数据进行傅里叶去噪
    def Denoise_csv_data(self):

        # 显示"正在绘图中"提示，关联到chart
        self.show_loading_message("正在绘图中...", self.fft_chart_view)


        # 强制界面更新
        QtWidgets.QApplication.processEvents()

        def fourier_denoise(series, threshold=0.001):
            # 进行傅里叶变换
            length = len(series)
            fft = np.fft.fft(series.values, length)

            # 计算功率谱密度
            PSD = fft * np.conj(fft) / length

            # 处理无效值
            PSD = np.nan_to_num(PSD)

            # 保留高频
            mask = PSD > threshold
            fft = mask * fft

            # 逆傅里叶变换
            denoised_data = np.fft.ifft(fft)
            denoised_data = denoised_data.real

            # 将去噪后的数据转换回 pandas.Series，并保留原始索引
            denoised_series = pd.Series(denoised_data, index=series.index)
            return denoised_series

        # 一维傅里叶去噪
        def objective_function(threshold):
            denoised_data = self.original_data.apply(lambda col: fourier_denoise(col, threshold))
            noise = self.original_data - denoised_data
            signal_power = np.mean(denoised_data ** 2)
            noise_power = np.mean(noise ** 2)
            snr = 10 * np.log10(signal_power / noise_power)  # 计算信噪比
            smoothness = denoised_data.diff().var().mean()  # 计算平滑度
            return snr + smoothness  # 信噪比越高，平滑度越高，目标函数越大

        # 贝叶斯优化
        pbounds = {'threshold': (0, 0.1)}  # 阈值搜索范围
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=42
        )
        optimizer.maximize(init_points=10, n_iter=50)  # 初始点10个，迭代20次

        # 获取最优阈值
        best_threshold = optimizer.max['params']['threshold']
        print(f"最优阈值: {best_threshold}")

        # 在界面上显示最佳阈值
        self.ui.textEditFFT.setText(str(best_threshold))

        # 使用傅里叶去噪处理每个变量
        denoised_data = self.original_data.apply(lambda col: fourier_denoise(col , best_threshold))
        
        old_data = self.original_data.copy()

        # 更新数据
        self.original_data = denoised_data


        # 准备 y 数据字典
        y_data_dict = {column: self.original_data[column] for column in self.original_data.columns}

        # 将 denoised_data 转换回 pandas.Series，并保留原始索引
        #self.original_data = pd.Series(denoised_data, index=self.original_data.index)

        
        try:
            plot_muldata(self.fft_chart, self.original_data.index, y_data_dict, "FFT Data", self.x_label, self.y_label)

        finally:
            # 绘图完成后隐藏提示
            self.hide_loading_message()

        # 生成简短的插值结果摘要
        summary_before = old_data.describe()
        summary_after = denoised_data.describe()

#为了节约时间先把机器人交互部分注释掉
        # #把插值处理的结果封装成气泡消息发送给聊天机器人
        # message = f"数据进行了rolling去噪与傅里叶去噪处理完成，傅里叶去噪阈值为 {best_threshold}，\n去噪前数据摘要：\n{summary_before}\n去噪后数据摘要：\n{summary_after}，根据前后数据摘要对比分析去噪带来的影响。"

        # curr_model = self.ui.comboBoxChatModel.currentText()

        # response = chatbot(curr_model,message,self.dataField)

        # #把聊天机器人的回复生成气泡消息
        # bubble_message = BubbleMessage(response, self.iconChat, Type=MessageType.Text, is_send=False)
        # self.chat_ui.add_message_item(bubble_message)


    def Stationarity_csv_data(self):

        #清空kpss_text中数据
        self.kpss_kpss_text=''
        self.adf_text=''
        # 平稳性检验

        

        ChooseColumn = self.ui.comboBoxChoosecolumn.currentText()

        # 检查数据
        # print("数据统计信息：")
        # print(self.original_data[ChooseColumn].describe())
        # print("缺失值数量：", self.original_data[ChooseColumn].isnull().sum())
        # print('nan值数量：',self.original_data[ChooseColumn].isna().sum())  # 检查 NaN
        # print('无穷大值数量：',self.original_data[ChooseColumn].isin([np.inf, -np.inf]).sum())  # 检查无穷大值

        # # 处理缺失值
        # self.original_data[ChooseColumn] = self.original_data[ChooseColumn].dropna()

        # 差分处理
        #self.original_data[ChooseColumn] = self.original_data[ChooseColumn].diff().dropna()

        # 清理数据
        self.original_data[ChooseColumn] = self.original_data[ChooseColumn].replace([np.inf, -np.inf], np.nan).dropna()

        self.original_data[ChooseColumn] = self.original_data[ChooseColumn].replace(['', None], np.nan).dropna()



        retes = self.original_data[ChooseColumn]

        retes = retes.replace(['', None], np.nan).dropna()

        #进行一阶差分
        retes_diff = retes.diff().dropna()

        
        # ADF检验
        adf_result = adfuller(retes_diff,autolag ='AIC')

        #直接把ADF的结果接入聊天机器人中让其分析各项指标
        
        adf_statistic = adf_result[0]  # 检验统计量
        adf_p_value = adf_result[1]        # p 值
        adf_lags = adf_result[2]           # 滞后阶数
        adf_nobs = adf_result[3]           # 样本数量
        adf_critical_values = adf_result[4]  # 临界值
        adf_ic = adf_result[5]             # 信息准则

        #打印各项指标
        # print("p值结果：",adf_p_value)

        # print("临界值：",adf_critical_values)
        
        # #打印返回的adf结果
        # print("ADF检验结果：",adf_result)

        # 转义特殊字符
        adf_statistic_str = html.escape(f"{adf_statistic:.4f}")
        adf_p_value_str = html.escape(f"{adf_p_value:.6f}")
        adf_ic_str = html.escape(f"{adf_ic:.4f}")


        # 在 textEditADF 中输出带颜色的 ADF 内容
        self.ui.textEditADF.setHtml(
            "<h3 style='color: #1E90FF; font-family: 黑体;'>=== ADF 检验结果 ===</h3>"
            f"<p style='font-family: 黑体;'>检验统计量 (ADF Statistic): <b>{adf_statistic_str}</b></p>"
            f"<p style='font-family: 黑体;'>p 值 (p-value): <b style='color: {'red' if adf_p_value > 0.05 else 'green'}'>{adf_p_value_str}</b></p>"
            "<p style='font-family: 黑体;'>临界值 (Critical Values):</p>"
            "<ul style='font-family: 黑体;'>"
            + "".join(f"<li>{key}: <b>{value:.4f}</b></li>" for key, value in adf_critical_values.items()) +
            "</ul>"
            f"<p style='font-family: 黑体;'>信息准则 (IC): <b>{adf_ic_str}</b></p>"
            f"""<p style='font-family: 黑体;'>结论: <b style='color: {'red' if adf_p_value > 0.05 else 'green'}'>
        {'数据不平稳' if adf_p_value > 0.05 else '数据平稳'}</b></p>"""
        )

        # KPSS检验
        kpss_result = kpss(retes, regression='c')
        
        kpss_statistic = kpss_result[0]  # 检验统计量
        kpss_p_value = kpss_result[1]    # p 值
        kpss_lags = kpss_result[2]       # 滞后阶数
        kpss_critical_values = kpss_result[3]  # 临界值

        # 转义特殊字符
        kpss_statistic_str = html.escape(f"{kpss_statistic:.4f}")
        kpss_p_value_str = html.escape(f"{kpss_p_value:.6f}")

        # 在 textEditKPSS 中输出带颜色的 KPSS 内容
        self.ui.textEditKPSS.setHtml(
        "<h3 style='color: #1E90FF; font-family: 黑体;'>=== KPSS 检验结果 ===</h3>"
        f"<p style='font-family: 黑体;'>检验统计量 (KPSS Statistic): <b>{kpss_statistic_str}</b></p>"
        f"<p style='font-family: 黑体;'>p 值 (p-value): <b style='color: {'red' if kpss_p_value > 0.05 else 'green'}'>{kpss_p_value_str}</b></p>"
        "<p style='font-family: 黑体;'>临界值 (Critical Values):</p>"
        "<ul style='font-family: 黑体;'>"
        + "".join(f"<li>{key}: <b>{value:.4f}</b></li>" for key, value in kpss_critical_values.items()) +
        "</ul>"
        f"<p style='font-family: 黑体;'>信息准则 (IC): <b>{adf_ic_str}</b></p>"
        f"""<p style='font-family: 黑体;'>结论: <b style='color: {'red' if kpss_p_value > 0.05 else 'green'}'>
        {'数据不平稳' if kpss_p_value > 0.05 else '数据平稳'}</b></p>"""
        )


        #self.plot_acf_data()
        if hasattr(self, 'acf_canvas') and self.acf_canvas:
            self.ui.widget_ACF.layout().removeWidget(self.acf_canvas)
            self.acf_canvas.setParent(None)
            self.acf_canvas = None  # 释放变量

        # ========== 移除已有的 PACF 画布 ==========
        if hasattr(self, 'pacf_canvas') and self.pacf_canvas:
            self.ui.widget_PACF.layout().removeWidget(self.pacf_canvas)
            self.pacf_canvas.setParent(None)
            self.pacf_canvas = None  # 释放变量

        # 创建 ACF 画布并添加到 PyQt 界面
        self.acf_canvas = ACFFigureCanvas(self.ui.widget_ACF)  
        self.acf_canvas.plot_acf(retes, lags=40)
        self.ui.widget_ACF.layout().addWidget(self.acf_canvas)  # 添加到布局

        # 创建 pACF 画布并添加到 PyQt 界面
        self.pacf_canvas = ACFFigureCanvas(self.ui.widget_PACF)  
        self.pacf_canvas.plot_pacf(retes, lags=40)
        self.ui.widget_PACF.layout().addWidget(self.pacf_canvas)  # 添加到布局

#为了节约时间先把机器人交互部分注释掉
        # acf_result = acf(retes, nlags=40, fft=True)
        # pacf_result = pacf(retes, nlags=40)

        # #封装ADF和KPSS的结果
        # message = f'数据进行了ADF和KPSS检验，ADF检验结果：{adf_result}，KPSS检验结果：{kpss_result}，acf数据{acf_result}，pacf数据{pacf_result}。根据ADF和KPSS检验结果分析数据的平稳性,解释acf和pacf数据。'

        # curr_model = self.ui.comboBoxChatModel.currentText()
        # response = chatbot(curr_model,message,self.dataField)


        # #把聊天机器人的回复生成气泡消息
        # bubble_message = BubbleMessage(response, self.iconChat, Type=MessageType.Text, is_send=False)
        # self.chat_ui.add_message_item(bubble_message)


    ###############################################################
    #以下是预测部分
    ###############################################################
    
    #设置各个模型参数
    #def arima_parameters(self):

#需要添加一个选择模型的函数


    def decomposition_csv_data(self):

        #清空decom_text中的数据
        self.decom_text = ''

        ChooseColumn = self.ui.comboBoxChoosecolumn.currentText()

        retes = self.original_data[ChooseColumn]

        retes = retes.replace(['', None], np.nan).dropna()

        #对数化转换
        #retes = np.log(retes)

        #进行一阶差分
        #retes_diff = retes.diff().dropna()

        
        
        # # 计算均值和标准差
        # mean_value = np.mean(retes)
        # std_value = np.std(retes)

        # #打印数据均值和标准差
        # print('数据的均值',mean_value)
        # print('数据标准差',std_value)

        # # 检查均值是否接近零
        # if abs(mean_value) < 1e-10:
        #     self.ui.textEditDecom.setHtml("<p style='color: red;'>错误: 数据均值接近零，无法计算 CV 值。</p>")
        #     return
        

        # #通过计算变异系数CV来判断数据的平稳性
        # #cv = np.std(retes_diff) / np.mean(retes_diff)
        # cv = np.std(retes) / np.mean(retes)

        # # 判断 CV 值是否异常
        # if abs(cv) > 1000:  # 如果 CV 值异常大
        #     self.ui.textEditDecom.setHtml("<p style='color: red;'>警告: CV 值异常，请检查数据是否存在异常值。</p>")
        #     return

        # # 判断是否使用加法模型或乘法模型
        # if abs(cv) < 0.5:  # 使用绝对值避免负值影响
        #     model_choice = 'additive'
        #     message = "数据波动平稳，使用加法模型进行分解"
        #     color = "green"
        # else:
        #     model_choice = 'multiplicative'
        #     message = "数据波动随均值增长，使用乘法模型进行分解"
        #     color = "blue"

        # # 更新 UI 显示
        # self.ui.textEditDecom.setHtml(
        #     f"""<p>CV值: <b>{cv:.4f}</b></p>
        # <p>判定结果: <b style='color: {color}'>{message}</b></p>"""
        # )
        # # 更新 UI 显示
        # self.ui.textEditDecom.setHtml(
        #     f"""<p>CV值: [<b>{cv:.4f}</b>]    判定结果: <b style='color: {color}'>{message}</b></p>"""
        # )

        # 指定季节性周期（例如月度数据周期为 12）
        seasonal_period = 24  # 根据数据特性调整

        # #一阶差

        # 进行时间序列分解
        try:
            # 使用 STL 分解（更强大的分解方法）
            stl_result = STL(
                retes,
                seasonal=13,       # 季节性平滑窗口
                trend=29,         # 趋势平滑窗口
                period=seasonal_period,  # 季节性周期
                #鲁棒性未开
            ).fit()

            # # 打印分解结果的统计信息
            # print("趋势成分的均值:", np.mean(stl_result.trend))
            # print("季节性成分的均值:", np.mean(stl_result.seasonal))
            # print("残差成分的均值:", np.mean(stl_result.resid))

            # 移除已有的画布
            if hasattr(self, 'decom_ori_canvas') and self.decom_ori_canvas:
                self.ui.widget_decom_ori.layout().removeWidget(self.decom_ori_canvas)
                self.decom_ori_canvas.setParent(None)
                self.decom_ori_canvas = None

            if hasattr(self, 'decom_trend_canvas') and self.decom_trend_canvas:
                self.ui.widget_decom_trend.layout().removeWidget(self.decom_trend_canvas)
                self.decom_trend_canvas.setParent(None)
                self.decom_trend_canvas = None

            if hasattr(self, 'decom_season_canvas') and self.decom_season_canvas:
                self.ui.widget_decom_season.layout().removeWidget(self.decom_season_canvas)
                self.decom_season_canvas.setParent(None)
                self.decom_season_canvas = None

            if hasattr(self, 'decom_resid_canvas') and self.decom_resid_canvas:
                self.ui.widget_decom_resid.layout().removeWidget(self.decom_resid_canvas)
                self.decom_resid_canvas.setParent(None)
                self.decom_resid_canvas = None

            # 绘制分解结果
            self.decom_ori_canvas = DecomFigureCanvas(self.ui.widget_decom_ori)
            self.decom_ori_canvas.plot_decom(retes, "原始数据", "#4C72B0")
            self.ui.widget_decom_ori.layout().addWidget(self.decom_ori_canvas)

            self.decom_trend_canvas = DecomFigureCanvas(self.ui.widget_decom_trend)
            self.decom_trend_canvas.plot_decom(stl_result.trend, "趋势", "#DD8452")
            self.ui.widget_decom_trend.layout().addWidget(self.decom_trend_canvas)

            self.decom_season_canvas = DecomFigureCanvas(self.ui.widget_decom_season)
            self.decom_season_canvas.plot_decom(stl_result.seasonal, "季节性", "#55A868")
            self.ui.widget_decom_season.layout().addWidget(self.decom_season_canvas)

            self.decom_resid_canvas = DecomFigureCanvas(self.ui.widget_decom_resid)
            self.decom_resid_canvas.plot_scatter(stl_result.resid, "残差", "#8172B3")
            self.ui.widget_decom_resid.layout().addWidget(self.decom_resid_canvas)

        except Exception as e:
            #self.ui.textEditDecom.append(f"<p style='color: red;'>错误: 时间序列分解失败 - {str(e)}</p>")
            print(f"错误: 时间序列分解失败 - {str(e)}")


#为了节约时间先把机器人交互部分注释掉
        # # 计算汇总统计信息
        # trend_summary = stl_result.trend.describe()
        # seasonal_summary = stl_result.seasonal.describe()
        # residual_summary = stl_result.resid.describe()

        # # 格式化为文本
        # summary_text = (
        #     f"=== STL 分解结果汇总 ===\n"
        #     "趋势项 (Trend):\n" + trend_summary.to_string() + "\n\n"
        #     "季节项 (Seasonal):\n" + seasonal_summary.to_string() + "\n\n"
        #     "残差项 (Residual):\n" + residual_summary.to_string() + "\n"
        #     '分析各项指标。'
        # )

        # curr_model = self.ui.comboBoxChatModel.currentText()
        # response = chatbot(curr_model,summary_text,self.dataField)

        # #把聊天机器人的回复生成气泡消息
        # bubble_message = BubbleMessage(response, self.iconChat, Type=MessageType.Text, is_send=False)
        # self.chat_ui.add_message_item(bubble_message)
        



    def predict_data(self):
        deepseek_api = 'sk-wdkwczxanmwevdpenbguhirxnjxpoyvoaqxrekhftvuizsld'
        gpt_api = 'sk-J4QMUqSVow9TsSdAIABKOIclpsFSpyPh2PJXccvbzBEepliv'

        Predict_model = self.ui.comboBoxPredictModel.currentText()
        print("当前预测模型", Predict_model)

        # 设置 API 密钥
        if Predict_model == 'LLMTime GPT-3.5':
            API_key = gpt_api
        elif Predict_model == 'DeepSeek':
            API_key = deepseek_api
        else:
            API_key = deepseek_api#随便传一个


        # 显示"正在绘图中"提示，关联到chart
        self.show_loading_message("正在绘图中...", self.predicted_chart_view)


        # 强制界面更新
        QtWidgets.QApplication.processEvents()


        # 获取预测长度
        predict_length = self.ui.plainTextEditLenth.toPlainText()

        # 初始化 predicted_data，确保其类型与 original_data 一致
        self.initialize_predicted_data(predict_length)

        pre_length = int(predict_length)

        #获取选择的预测列数据
        ChooseColumn = self.ui.comboBoxChoosecolumn.currentText()


        origin_col_data = self.original_data
        predict_col_data = self.predicted_data

        # # 获取当前列的原始数据和预测数据
        # origin_col_data = self.original_data[ChooseColumn]
        # predict_col_data = self.predicted_data[ChooseColumn]

        #打印原始数据长度
        print("原始数据长度:",len(origin_col_data))

#还需要实现各个模型的多变量预测数据载入与模型内部对接

        #如果选择的是arima模型则只需要传入一列数据
        if Predict_model == 'ARIMA':
            origin_col_data_uni = self.original_data[ChooseColumn]
            predict_col_data_uni = self.predicted_data[ChooseColumn]

            # 获取预测结果
            predict_result = get_predictions(Predict_model,ChooseColumn, API_key, origin_col_data_uni, predict_col_data_uni, self.hypers)
        else:

            # 获取预测结果
            predict_result = get_predictions(Predict_model, ChooseColumn,API_key, origin_col_data, predict_col_data, self.hypers)

#如果choosecolumn=all，则对所有列进行预测

        # 检查 predict_result 的类型和内容
        print("Predict result type:", type(predict_result))
        print("Predict result content:", predict_result)

        # 更新 predicted_data 的对应列
        #self.predicted_data[ChooseColumn] = predict_result

        #打印预测结构的类型
        print("预测结果类型:",type(predict_result))

        # 将 predict_result 转换为 pandas.Series
        predict_result_series = pd.Series(predict_result, index=predict_col_data.index)

        # 更新 predicted_data 的对应列
        self.predicted_data[ChooseColumn] = predict_result_series

        #origin_col_data丢掉最后pre_length个数据
        #origin_col_data = origin_col_data[:-pre_length]

        # #把预测结果与原始数据对应列进行合并
        origin_choose_series = origin_col_data[ChooseColumn]


        #y_data = pd.concat([origin_col_data, predict_result], axis=0)
        y_data = pd.concat([origin_choose_series, predict_result_series], axis=0)

        #print("预测结果形状：",y_data.shape)

        # #把预测结果与原始数据进行合并
        # self.predicted_data = pd.concat([self.original_data, self.predicted_data], axis=1)

        # #打印合并后的数据
        # print("合并后的数据:",self.predicted_data)

        # #打印预测结果长度
        # print("预测结果长度:",len(self.predicted_data))

        #获取保存预测结果路径
        self.savePath = self.ui.lineEditSavepath.toPlainText()

        
        try:
            # # 绘制预测结果
            # if isinstance(self.predicted_data, pd.DataFrame):
        
                #plot_predata(self.predicted_chart, self.predicted_data.index, y_data_dict, f"Predicted Data ({Predict_model})", self.x_label, self.y_label,pre_length)
            plot_predata(self.predicted_chart, y_data.index, y_data, f"Predicted Model:{Predict_model} Predicted Column: {ChooseColumn}", self.x_label, self.y_label,pre_length)
        finally:
            # 绘图完成后隐藏提示
            self.hide_loading_message()  
#节约时间先把机器人交互部分注释掉
        # #把原始数据最后100个数据和预测数据输入到聊天机器人中
        # message = f"原始数据最后100个数据：\n{origin_col_data[-100:]}\n预测数据：\n{predict_result},结合原始数据的趋势周期性等等内容对预测结果进行评价。"
            
        # curr_model = self.ui.comboBoxChatModel.currentText()
        # response = chatbot(curr_model,message,self.dataField)

        # #把聊天机器人的回复生成气泡消息
        # bubble_message = BubbleMessage(response, self.iconChat, Type=MessageType.Text, is_send=False)
        # self.chat_ui.add_message_item(bubble_message)
        
        #print(f"Predicted Data ({Predict_model}):", self.predicted_data)









        # # 遍历 original_data 的每一列数据进行预测
        # for column in self.original_data.columns:
        #     # 获取当前列的原始数据和预测数据
        #     origin_col_data = self.original_data[column]
        #     predict_col_data = self.predicted_data[column]


        #     # 获取预测结果
        #     predict_result = get_predictions(Predict_model, API_key, origin_col_data, predict_col_data, self.hypers)

        #     # 检查 predict_result 的类型和内容
        #     print("Predict result type:", type(predict_result))
        #     print("Predict result content:", predict_result)

        #     # 确保 predict_result 是一个字典
        #     # if not isinstance(predict_result, dict):
        #     #     raise TypeError("predict_result must be a dictionary")

        #     # # 遍历每个模型的预测结果
        #     # for model_name, model_result in predict_result.items():
        #     #     # 检查模型结果是否包含 'median'
        #     #     if not isinstance(model_result, dict) or 'median' not in model_result:
        #     #         print(f"Skipping {model_name}: 'median' key not found")
        #     #         continue

        #     #     # 提取 median 预测结果
        #     #     predict_series = model_result['median']

        #     #     # 确保 predict_series 是一个 pandas.Series 或 numpy.ndarray
        #     #     if not isinstance(predict_series, (pd.Series, np.ndarray)):
        #     #         print(f"Warning: predict_series is not a Series or ndarray, got {type(predict_series)}")
        #     #         continue

        #     #     # 如果 predict_series 是 numpy.ndarray，将其转换为 pandas.Series
        #     #     if isinstance(predict_series, np.ndarray):
        #     #         predict_series = pd.Series(predict_series, index=self.predicted_data.index)

        #     #     # 确保 predict_series 的长度与 predicted_data 的长度一致
        #     #     if len(predict_series) != len(self.predicted_data):
        #     #         print(f"Warning: Length mismatch in predict_series. Expected {len(self.predicted_data)}, got {len(predict_series)}")
        #     #         continue

        #     #     # 更新 predicted_data 的对应列
        #     #     self.predicted_data[column] = predict_series

        #     #将predict_result字典中对应model的预测结果提取出来
        #     # if Predict_model in predict_result:
        #     #     predict_model_result = predict_result[Predict_model]
        #     #     #打印提取成功的预测结果
        #     #     print(f"提取出来 for {Predict_model}:", predict_result)
        #     # else:
        #     #     print(f"Skipping {Predict_model}: 'median' key not found")
        #     #     continue


        #     #更新 predicted_data 的对应列
        #     #self.predicted_data[column] = predict_model_result
        #     self.predicted_data[column] = predict_result

        # # 绘制预测结果
        # if isinstance(self.predicted_data, pd.DataFrame):
        #     # 多变量数据
        #     y_data_dict = {column: self.predicted_data[column] for column in self.predicted_data.columns}
        #     plot_muldata(self.predicted_chart, self.predicted_data.index, y_data_dict, f"Predicted Data ({Predict_model})", self.x_label, self.y_label)
        # # else:
        # #     # 单变量数据
        # #     plot_data(self.predicted_chart, self.predicted_data.index, predicted_data.values, f"Predicted Data ({model_name})", self.x_label, self.y_label)

        # print(f"Predicted Data ({Predict_model}):", self.predicted_data)

        # #print("更新的Predicted data:", self.predicted_data)

    #把预测得到的结果保存到本地
    def save_predicted_data(self):
        #获取保存预测结果路径
        self.savePath = self.ui.lineEditSavepath.toPlainText()

        #保存预测结果
        if hasattr(self, 'predicted_data') and self.predicted_data is not None:
            self.predicted_data.to_csv(self.savePath, index=True)
            print(f"Predicted data saved to {self.savePath}")
        else:
            print("Predicted data is not available.")


    ###############################################################
    #以下是聊天部分
    ###############################################################

    #接入聊天机器人
    def AskChatbot(self):
        
        #当button_send触发之后获取文本框中发送的信息
        current_text = self.ui.plainTextEdit.toPlainText()
    
        if not current_text.strip():
            print("No message to send")
            return


        curr_model = self.ui.comboBoxChatModel.currentText()
    
        response = chatbot(curr_model, current_text,self.dataField)


        #把聊天机器人的回复生成气泡消息
        bubble_message = BubbleMessage(response, self.iconChat, Type=MessageType.Text, is_send=False)
        self.chat_ui.add_message_item(bubble_message)
        
        #发送之后清空文本框
        self.ui.plainTextEdit.clear()
        

    
    #创建气泡
    def create_bubble(self):
        self.text=self.ui.plainTextEdit.toPlainText()
        #self.ui.plainTextEdit.setPlainText("发送问题")
        
        #点击发送问题之后调用气泡消息函数
        bubble_send = BubbleMessage(self.text, self.iconUser, Type=MessageType.Text, is_send=True)
        self.chat_ui.add_message_item(bubble_send)

        #调用聊天机器人
        self.AskChatbot()

        #self.adjustScrollToMaxValue()

                                    
    #窗口滚动到最底层
    def adjustScrollToMaxValue(self):
        scrollbar = self.ui.scrollArea.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    