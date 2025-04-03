import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QVBoxLayout, QGraphicsScene, QGraphicsView, QGraphicsTextItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtChart import QChart ,QChartView,QLineSeries,QDateTimeAxis,QValueAxis
from PyQt5.QtCore import Qt, QPointF, QDateTime, QRectF, QRect
from PyQt5.QtGui import QFont, QPainter, QPainterPath, QColor, QFontMetrics, QBrush, QPen
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsSimpleTextItem, QGraphicsView, QGraphicsScene
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QSplineSeries, QDateTimeAxis, QValueAxis

mpl.use('Qt5Agg')

# class ForcastingFigure(FigureCanvas):
#     def __init__(self, parent=None):
#         self.fig = Figure(figsize=(6, 4))
#         super().__init__(self.fig)

#     def plot_preds(self, train, test, pred_dict, model_name, show_samples=False):
#         pred = pred_dict['median']
#         pred = pd.Series(pred, index=test.index)
#         ax = self.figure.add_subplot(111)
#         self.figure.subplots_adjust(left=0.1, right=0.75)
#         ax.clear()
#         ax.plot(train)
#         ax.plot(pred, label=model_name, color='purple')
#         samples = pred_dict['samples']
#         lower = np.quantile(samples, 0.05, axis=0)
#         upper = np.quantile(samples, 0.95, axis=0)
#         ax.fill_between(pred.index, lower, upper, alpha=0.3, color='purple')
#         if show_samples:
#             samples = pred_dict['samples']
#             samples = samples.values if isinstance(samples, pd.DataFrame) else samples
#             for i in range(min(10, samples.shape[0])):
#                 ax.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
#         ax.legend(bbox_to_anchor=(1, 1), loc=3, borderaxespad=0)
#         if 'NLL/D' in pred_dict:
#             nll = pred_dict['NLL/D']
#             if nll is not None:
#                 ax.text(1.01, 0.85, f'NLL/D: {nll:.2f}', transform=ax.transAxes,
#                         bbox=dict(facecolor='white', alpha=0.5))
#         self.fig.canvas.draw()

class ImageFigures(FigureCanvas):
    def __init__(self, widget):
        self.fig = Figure(figsize=(3,3))
        super().__init__(self.fig)
        layout = QVBoxLayout(widget)
        layout.addWidget(self)

    def plotImage(self, x, y1, y2=None, title=None, legend1='Open', legend2=None):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.clear()
        ax.plot(x, y1)
        if y2 is not None:
            ax.plot(x, y2)
        ax.tick_params(axis='both', which='major', labelsize=5)
        if title is not None:
            ax.set_title(title)
        ax.legend([legend1, legend2])
        self.fig.canvas.draw()


class Callout(QGraphicsItem):
    def __init__(self, chart):
        super(Callout, self).__init__(chart)
        self.chart = chart
        self.text = ''
        self.anchor = QPointF()
        self.rect = QRectF()
        self.textRect = QRectF()
        self.font = QFont()

    def boundingRect(self):
        anchor = self.mapFromParent(self.chart.mapToPosition(self.anchor))
        rect = QRectF()
        rect.setLeft(min(self.rect.left(), anchor.x()))
        rect.setRight(max(self.rect.right(), anchor.x()))
        rect.setTop(min(self.rect.top(), anchor.y()))
        rect.setBottom(max(self.rect.bottom(), anchor.y()))
        return rect

    def paint(self, painter, option, widget):
        path = QPainterPath()
        path.addRoundedRect(self.rect, 5, 5)

        anchor = self.mapFromParent(self.chart.mapToPosition(self.anchor))
        if not (self.rect.contains(anchor) and not self.anchor.isNull()):
            point1 = QPointF()
            point2 = QPointF()

            above = anchor.y() <= self.rect.top()
            aboveCenter = anchor.y() > self.rect.top() and anchor.y() <= self.rect.center().y()
            belowCenter = anchor.y() > self.rect.center().y() and anchor.y() <= self.rect.bottom()
            below = anchor.y() > self.rect.bottom()

            left = anchor.x() <= self.rect.left()
            leftCenter = anchor.x() > self.rect.left() and anchor.x() <= self.rect.center().x()
            rightCenter = anchor.x() > self.rect.center().x() and anchor.x() <= self.rect.right()
            right = anchor.x() > self.rect.right()

            x = (right + rightCenter) * self.rect.width()
            y = (below + belowCenter) * self.rect.height()
            cornerCase = (above and left) or (above and right) or (below and left) or (below and right)
            vertical = abs(anchor.x() - x) > abs(anchor.y() - y)

            x1 = x + leftCenter * 10 - rightCenter * 20 + cornerCase * (not vertical) * (left * 10 - right * 20)
            y1 = y + aboveCenter * 10 - belowCenter * 20 + cornerCase * vertical * (above * 10 - below * 20)
            point1.setX(x1)
            point1.setY(y1)

            x2 = x + leftCenter * 20 - rightCenter * 10 + cornerCase * (not vertical) * (left * 20 - right * 10)
            y2 = y + aboveCenter * 20 - belowCenter * 10 + cornerCase * vertical * (above * 20 - below * 10)
            point2.setX(x2)
            point2.setY(y2)

            path.moveTo(point1)
            path.lineTo(anchor)
            path.lineTo(point2)
            path = path.simplified()

        painter.setBrush(QColor(255, 255, 255))
        painter.drawPath(path)
        painter.drawText(self.textRect, self.text)

    def setText(self, text):
        self.text = text
        metrics = QFontMetrics(self.font)
        self.textRect = QRectF(metrics.boundingRect(QRect(0, 0, 150, 150), Qt.AlignLeft, self.text))
        self.textRect.translate(5, 5)
        self.prepareGeometryChange()
        self.rect = self.textRect.adjusted(-5, -5, 5, 5)

    def setAnchor(self, point):
        self.anchor = point

    def updateGeometry(self):
        self.prepareGeometryChange()
        self.setPos(self.chart.mapToPosition(self.anchor) + QPointF(10, 50))




class CustomChartView(QChartView):
    def __init__(self, chart, parent=None):
        super().__init__(chart, parent)
        self.setMouseTracking(True)
        self.tooltip = QGraphicsTextItem()
        self.tooltip.setDefaultTextColor(Qt.red)
        self.tooltip.setZValue(1)
        self.scene().addItem(self.tooltip)
        self.tooltip.hide()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        value = self.chart().mapToValue(pos)
        x = QDateTime.fromMSecsSinceEpoch(int(value.x())).toString("yyyy-MM-dd HH:mm:ss")
        y = value.y()
        self.tooltip.setPlainText(f"({x}, {y:.2f})")
        self.tooltip.setPos(pos)
        self.tooltip.show()
        super().mouseMoveEvent(event)


# #在ui界面上绘制数据图像
# def plot_data(chart, x_data, y_data, title, x_label, y_label):
#     series = QLineSeries()
#     for x, y in zip(x_data, y_data):
#         series.append(x.timestamp() * 1000, y)  # 将时间转换为毫秒

#     chart.removeAllSeries()
#     chart.addSeries(series)
#     chart.createDefaultAxes()
#     chart.setTitle(title)

#     # 设置 x 轴为时间轴
#     axis_x = QDateTimeAxis()
#     axis_x.setFormat("yyyy-MM-dd HH:mm:ss")
#     axis_x.setTitleText(x_label)
#     chart.setAxisX(axis_x, series)

#     # 设置 y 轴
#     axis_y = QValueAxis()
#     axis_y.setTitleText(y_label)
#     chart.setAxisY(axis_y, series)

#使用qtchart绘制ACF图的图像
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.stattools import acf
# def plot_acf(chart, data, lags=40, title="ACF 自相关图"):
#     """
#     使用 QtChart 绘制 ACF（自相关函数）图

#     :param chart: QChart 实例
#     :param data: pandas.Series 形式的时间序列数据
#     :param lags: 计算 ACF 的最大滞后阶数
#     :param title: 图表标题
#     """
#     # 计算 ACF 值
#     acf_values = acf(data, nlags=lags, fft=True)

#     # 清空图表
#     chart.removeAllSeries()

#     # 创建 QLineSeries 并填充 ACF 数据
#     series = QLineSeries()
#     for i, value in enumerate(acf_values):
#         series.append(i, value)  # 横坐标是滞后阶数 i，纵坐标是 ACF 值

#     # 添加 ACF 线条到图表
#     chart.addSeries(series)

#     # 设置图表标题
#     chart.setTitle(title)

#     # 设置 X 轴
#     axis_x = QValueAxis()
#     axis_x.setTitleText("Lags")
#     axis_x.setLabelFormat("%d")
#     axis_x.setRange(0, lags)  # X 轴范围设为最大滞后阶数
#     chart.setAxisX(axis_x, series)

#     # 设置 Y 轴
#     axis_y = QValueAxis()
#     axis_y.setTitleText("ACF Value")
#     axis_y.setLabelFormat("%.2f")
#     axis_y.setRange(-1, 1)  # ACF 取值范围一般在 -1 到 1 之间
#     chart.setAxisY(axis_y, series)

#     # 关联坐标轴到系列
#     chart.createDefaultAxes()


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ACFFigureCanvas(FigureCanvas):
    def __init__(self, parent=None):
        """
        ACF 绘图的 Matplotlib 画布（嵌入 PyQt5 widget）
        :param parent: 目标 PyQt5 widget
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        self.fig, self.ax = plt.subplots(figsize=(6, 4))  # 创建 Matplotlib Figure
        super().__init__(self.fig)  # 继承 FigureCanvas
        self.setParent(parent)
    def plot_acf(self, data, lags=40, title="自相关函数（ACF）"):
        """
        在 PyQt5 界面中绘制 ACF 图
        :param data: 需要计算 ACF 的时间序列数据
        :param lags: 计算 ACF 的滞后阶数（默认 40）
        :param title: 图表标题
        """
        self.ax.clear()  # 清空旧图
        plot_acf(data, ax=self.ax, lags=lags, color="#4C72B0")  # 使用柔和深蓝绘制 ACF 线条
        self.ax.set_title(title, fontsize=14, color="#4C72B0", fontfamily="SimHei")  # 标题颜色与线条一致
        self.ax.grid(True, linestyle="--", alpha=0.6, color="#8172B3")  # 网格线使用灰紫色
        self.ax.set_facecolor("#F5F5F5")  # 设置浅灰色背景
        self.draw()  # 重新渲染图像

    def plot_pacf(self, data, lags=40, title="偏自相关函数（PACF）"):
        """
        在 PyQt5 界面中绘制 PACF 图
        :param data: 需要计算 PACF 的时间序列数据
        :param lags: 计算 PACF 的滞后阶数（默认 40）
        :param title: 图表标题
        """
        self.ax.clear()  # 清空旧图
        plot_pacf(data, ax=self.ax, lags=lags, color="#DD8452")  # 使用陶土橙绘制 PACF 线条
        self.ax.set_title(title, fontsize=14, color="#DD8452", fontfamily="SimHei")  # 标题颜色与线条一致
        self.ax.grid(True, linestyle="--", alpha=0.6, color="#8172B3")  # 网格线使用灰紫色
        self.ax.set_facecolor("#F5F5F5")  # 设置浅灰色背景
        self.draw()  # 重新渲染图像
    

    # def plot_acf(self, data, lags=40, title="自相关函数（ACF）"):
    #     """
    #     在 PyQt5 界面中绘制 ACF 图
    #     :param data: 需要计算 ACF 的时间序列数据
    #     :param lags: 计算 ACF 的滞后阶数（默认 40）
    #     :param title: 图表标题
    #     """
    #     self.ax.clear()  # 清空旧图
    #     plot_acf(data, ax=self.ax, lags=lags)  # 使用 statsmodels 绘制 ACF
    #     self.ax.set_title(title, fontsize=14, color='#00AAFF')  # 避免与背景冲突
    #     self.ax.grid(True, linestyle="--", alpha=0.6)  # 添加网格，提高可读性
    #     self.draw()  # 重新渲染图像

    # def plot_pacf(self, data, lags=40, title="偏自相关函数（PACF）"):
    #     """
    #     在 PyQt5 界面中绘制 PACF 图
    #     :param data: 需要计算 PACF 的时间序列数据
    #     :param lags: 计算 PACF 的滞后阶数（默认 40）
    #     :param title: 图表标题
    #     """
    #     self.ax.clear()  # 清空旧图
    #     plot_pacf(data, ax=self.ax, lags=lags)  # 使用 statsmodels 绘制 PACF
    #     self.ax.set_title(title, fontsize=14, color='#00AAFF')  # 避免与背景冲突
    #     self.ax.grid(True, linestyle="--", alpha=0.6)  # 添加网格，提高可读性
    #     self.draw()  # 重新渲染图像

import matplotlib.dates as mdates

class DecomFigureCanvas(FigureCanvas):
    def __init__(self, parent=None):
        """
        绘图的 Matplotlib 画布（嵌入 PyQt5 widget）
        :param parent: 目标 PyQt5 widget
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        self.fig, self.ax = plt.subplots(figsize=(6, 4))  # 创建 Matplotlib Figure
        super().__init__(self.fig)  # 继承 FigureCanvas
        self.setParent(parent)

        #绘制时间序列分解的子图
    def plot_decom(self, data, title, color="blue"):

        self.ax.clear()

        # 如果 data 的索引是时间戳，转换为 Datetime 格式
        if isinstance(data.index, pd.DatetimeIndex):
            self.ax.plot(data.index, data, color=color)
            
            # 1. **设置日期格式**
            self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # 自动调整刻度
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # 显示日期

            # 2. **防止日期标签重叠**
            plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45, ha="right")  # 旋转标签

        else:
            self.ax.plot(data, color=color)

        self.ax.plot(data, color=color)
        self.ax.set_title(title, fontsize=14, color='#00AAFF')  # 避免与背景冲突
        self.ax.grid(True, linestyle="--", alpha=0.6,color="#8172B3")
        self.ax.set_facecolor("#F5F5F5")  # 设置浅灰色背景
        self.draw()


    def plot_scatter(self, data, title, color="purple"):
        """专门用于绘制散点图（残差）"""
        self.ax.clear()
        
        # 处理时间轴格式
        if isinstance(data.index, pd.DatetimeIndex):
            self.ax.scatter(data.index, data, color=color, alpha=0.6)  # 残差用散点图
            
            # 优化 x 轴时间格式
            self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45, ha="right")  

        else:
            self.ax.scatter(range(len(data)), data, color=color, alpha=0.6)

        self.ax.set_title(title, fontsize=14, color='#00AAFF')
        self.ax.grid(True, linestyle="--", alpha=0.6,color="#8172B3")
        self.ax.set_facecolor("#F5F5F5")  # 设置浅灰色背景
        self.draw()


def plot_data(chart, x_data, y_data, title, x_label, y_label):

    #设置图表的主题
    chart.setTheme(QChart.ChartThemeBlueIcy)
    chart.setAnimationOptions(QChart.SeriesAnimations)


    # 创建 QLineSeries
    series = QLineSeries()
    for x, y in zip(x_data, y_data):
        series.append(x.timestamp() * 1000, y)  # 将时间转换为毫秒

    # 清空图表
    chart.removeAllSeries()
    chart.addSeries(series)
    chart.createDefaultAxes()


    # 设置标题字体为黑体
    font = QFont("黑体", 16)  # 设置字体为黑体，大小为12
    chart.setTitleFont(font)
    chart.setTitle(title)

    # 设置图例字体为黑体
    legend_font = QFont("黑体", 12)  # 设置字体为黑体，大小为12
    chart.legend().setFont(legend_font)

    # 设置 x 轴为时间轴
    axis_x = QDateTimeAxis()
    axis_x.setFormat("yyyy-MM-dd HH:mm:ss")
    axis_x.setTitleText(x_label)
    chart.setAxisX(axis_x, series)

    # 设置 y 轴
    axis_y = QValueAxis()
    axis_y.setTitleText(y_label)
    chart.setAxisY(axis_y, series)

    # 添加标注功能
    tooltip = None
    callouts = []

    def keepCallout():
        nonlocal tooltip
        if tooltip:
            callouts.append(tooltip)
            tooltip = Callout(chart)

    def showTooltip(point, state):
        nonlocal tooltip
        if tooltip is None:
            tooltip = Callout(chart)
        if state:
            tooltip.setText(f'X: {QDateTime.fromMSecsSinceEpoch(int(point.x())).toString("yyyy-MM-dd HH:mm:ss")}\nY: {point.y():.2f}')
            tooltip.setAnchor(point)
            tooltip.setZValue(11)
            tooltip.updateGeometry()
            tooltip.show()
        else:
            tooltip.hide()

    # 绑定事件
    series.hovered.connect(showTooltip)
    series.clicked.connect(keepCallout)



def plot_muldata(chart, x_data, y_data_dict, title, x_label, y_label):
    """
    绘制多变量数据的函数

    :param chart: QChart 对象，用于绘制图表
    :param x_data: x 轴数据，通常是时间序列
    :param y_data_dict: 字典，键为变量名称，值为对应的 y 轴数据
    :param title: 图表标题
    :param x_label: x 轴标签
    :param y_label: y 轴标签
    """
    # 设置图表的主题
    chart.setTheme(QChart.ChartThemeBlueIcy)
    chart.setAnimationOptions(QChart.SeriesAnimations)

    # 清空图表
    chart.removeAllSeries()

    # 定义颜色列表（现代配色方案）
    colors = [
        QColor(76, 114, 176),  # 柔和深蓝
        QColor(221, 132, 82),  # 陶土橙
        QColor(85, 168, 104),  # 森林绿
        QColor(129, 114, 179), # 灰紫色
        QColor(210, 77, 87),   # 深红色
        QColor(100, 181, 205), # 浅蓝色
    ]
    background_color = QColor(245, 245, 245)  # 浅灰色背景

    # 设置背景颜色
    chart.setBackgroundBrush(QBrush(background_color))

    # 创建并添加多个 QLineSeries
    for idx, (var_name, y_data) in enumerate(y_data_dict.items()):
        series = QLineSeries()
        series.setName(var_name)  # 设置系列名称

        # 为每个系列分配不同的颜色
        color = colors[idx % len(colors)]  # 循环使用颜色列表
        series.setColor(color)  # 设置线条颜色
        series.setPen(QPen(color, 3))  # 设置线条粗细为 3，更粗更圆润

        # 添加数据到 series
        for x, y in zip(x_data, y_data):
            if y is not None:  # 如果 y 不为空
                series.append(x.timestamp() * 1000, y)  # 将时间转换为毫秒

        # 添加 series 到图表
        chart.addSeries(series)

    # 创建默认的坐标轴
    chart.createDefaultAxes()

    # 设置标题字体为黑体
    font = QFont("黑体", 16)  # 设置字体为黑体，大小为12
    chart.setTitleFont(font)
    chart.setTitle(title)

    # 设置图例字体为黑体
    legend_font = QFont("黑体", 12)  # 设置字体为黑体，大小为12
    chart.legend().setFont(legend_font)

    # 设置 x 轴为时间轴
    axis_x = QDateTimeAxis()
    axis_x.setFormat("yyyy-MM-dd HH:mm:ss")
    axis_x.setTitleText(x_label)
    axis_x.setGridLineVisible(True)  # 显示网格线
    axis_x_font = QFont("黑体", 12)  # 设置字体为黑体，大小为12
    axis_x.setTitleFont(axis_x_font)  # 设置 x 轴标题字体
    chart.setAxisX(axis_x)

    # 设置 y 轴
    axis_y = QValueAxis()
    axis_y.setTitleText(y_label)
    axis_y.setGridLineVisible(True)  # 显示网格线
    axis_y_font = QFont("黑体", 12)  # 设置字体为黑体，大小为12
    axis_y.setTitleFont(axis_y_font)  # 设置 y 轴标题字体
    chart.setAxisY(axis_y)

    # 确保每个系列都与轴正确关联
    for series in chart.series():
        chart.setAxisX(axis_x, series)
        chart.setAxisY(axis_y, series)

    # 设置图例
    chart.legend().setVisible(True)
    chart.legend().setAlignment(Qt.AlignBottom)

    # 添加标注功能
    tooltip = None
    callouts = []

    def keepCallout():
        nonlocal tooltip
        if tooltip:
            callouts.append(tooltip)
            tooltip = Callout(chart)

    def showTooltip(point, state,series_name):
        nonlocal tooltip
        if tooltip is None:
            tooltip = Callout(chart)
        if state:
            tooltip.setText(f'Series: {series_name}\nX: {QDateTime.fromMSecsSinceEpoch(int(point.x())).toString("yyyy-MM-dd HH:mm:ss")}\nY: {point.y():.2f}')
            
            tooltip.setAnchor(point)
            tooltip.setZValue(11)
            tooltip.updateGeometry()
            tooltip.show()
        else:
            tooltip.hide()

    # 绑定事件
    for series in chart.series():
        series.hovered.connect(lambda point, state, series=series: showTooltip(point, state, series.name()))
        
        series.clicked.connect(keepCallout)

def plot_predata(chart, x_data, y_data, title, x_label, y_label, pred_length):
    """
    绘制多变量数据的函数，并根据预测长度将曲线的最后一部分数据点用另一种颜色标识

    :param chart: QChart 对象，用于绘制图表
    :param x_data: x 轴数据，通常是时间序列
    :param y_data: y 轴数据，单列数据
    :param title: 图表标题
    :param x_label: x 轴标签
    :param y_label: y 轴标签
    :param pred_length: 预测长度，用于标识最后一部分数据点
    """
    # 设置图表的主题
    chart.setTheme(QChart.ChartThemeBlueIcy)
    chart.setAnimationOptions(QChart.SeriesAnimations)

    # 清空图表
    chart.removeAllSeries()

    # 定义颜色
    original_color = QColor(76, 144, 176)  # 蓝色表示原始数据
    predicted_color = QColor(221, 132, 82)  # 红色表示预测数据

    background_color = QColor(245, 245, 245)  # 浅灰色背景

    # 设置背景颜色
    chart.setBackgroundBrush(QBrush(background_color))


    # 创建原始数据系列
    original_series = QLineSeries()
    original_series.setName(f"Original {y_label}")
    original_series.setColor(original_color)
    original_series.setPen(QPen(original_color, 3))  # 设置线条粗细为 3，更粗更圆润

    # 创建预测数据系列
    predicted_series = QLineSeries()
    predicted_series.setName(f"Predicted {y_label}")
    predicted_series.setColor(predicted_color)
    predicted_series.setPen(QPen(predicted_color, 3))  # 设置线条粗细为 3，更粗更圆润

    # # 分割数据
    # split_index = len(y_data) - pred_length  # 分割点
    # for i, (x, y) in enumerate(zip(x_data, y_data)):
    #     if y is not None:  # 如果 y 不为空
    #         if i < split_index:
    #             original_series.append(x.timestamp() * 1000, y)  # 原始数据
    #         else:
    #             predicted_series.append(x.timestamp() * 1000, y)  # 预测数据

    #打印传入的数据长度
    # print(f"y_data的长度: {len(y_data)}")
    # print(f"y_data的类型: {type(y_data)}")


    # 分割数据
    split_index = len(y_data) - pred_length  # 分割点

    #打印分割数据长度
    print(f"split_index: {split_index}")

    for i, (x, y) in enumerate(zip(x_data, y_data)):
        #打印输入数据
        #print(f"i: {i}, x: {x}, y: {y}")

        if y is not None and not isinstance(y,str):  # 如果 y 不为空
            try:
                y_float = float(y)  # 将 y 转换为浮点数
                if i < split_index:
                    original_series.append(x.timestamp() * 1000, y_float)  # 原始数据
                else:
                    predicted_series.append(x.timestamp() * 1000, y_float)  # 预测数据
            except ValueError:
                print(f"Warning: Could not convert y value '{y}' to float. Skipping this data point.")

    #打印分割后两个数据的长度
    print(f"Original series length: {len(original_series)}")
    print(f"Predicted series length: {len(predicted_series)}")


    # 添加 series 到图表
    chart.addSeries(original_series)
    chart.addSeries(predicted_series)

    # 创建默认的坐标轴
    chart.createDefaultAxes()

    # 设置标题字体为黑体
    font = QFont("黑体", 16)  # 设置字体为黑体，大小为12
    chart.setTitleFont(font)
    chart.setTitle(title)

    # 设置图例字体为黑体
    legend_font = QFont("黑体", 12)  # 设置字体为黑体，大小为12
    chart.legend().setFont(legend_font)

    # 设置 x 轴为时间轴
    axis_x = QDateTimeAxis()
    axis_x.setFormat("yyyy-MM-dd HH:mm:ss")
    axis_x.setTitleText(x_label)
    axis_x_font = QFont("黑体", 12)  # 设置字体为黑体，大小为12
    axis_x.setTitleFont(axis_x_font)  # 设置 x 轴标题字体
    axis_x.setGridLineVisible(True)  # 显示网格线
    chart.setAxisX(axis_x)

    #打印x轴的数据范围
    print(f"x_data的最小值: {min(x_data)}")
    print(f"x_data的最大值: {max(x_data)}")

    # 设置 y 轴
    axis_y = QValueAxis()
    axis_y.setTitleText(y_label)
    axis_y_font = QFont("黑体", 12)  # 设置字体为黑体，大小为12
    axis_y.setTitleFont(axis_y_font)  # 设置 y 轴标题字体
    axis_y.setGridLineVisible(True)  # 显示网格线
    chart.setAxisY(axis_y)

    # 确保每个系列都与轴正确关联
    for series in chart.series():
        chart.setAxisX(axis_x, series)
        chart.setAxisY(axis_y, series)

    # 设置图例
    chart.legend().setVisible(True)
    chart.legend().setAlignment(Qt.AlignBottom)

    # 添加标注功能
    tooltip = None
    callouts = []

    def keepCallout():
        nonlocal tooltip
        if tooltip:
            callouts.append(tooltip)
            tooltip = Callout(chart)

    def showTooltip(point, state, series_name):
        nonlocal tooltip
        if tooltip is None:
            tooltip = Callout(chart)
        if state:
            tooltip.setText(f'Series: {series_name}\nX: {QDateTime.fromMSecsSinceEpoch(int(point.x())).toString("yyyy-MM-dd HH:mm:ss")}\nY: {point.y():.2f}')
            tooltip.setAnchor(point)
            tooltip.setZValue(11)
            tooltip.updateGeometry()
            tooltip.show()
        else:
            tooltip.hide()

    # 绑定事件
    for series in chart.series():
        series.hovered.connect(lambda point, state, series=series: showTooltip(point, state, series.name()))
        series.clicked.connect(keepCallout)
