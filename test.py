import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout

class SlowDownApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Slow Down System')

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel('发现《Slow Down》 >')
        title_label.setStyleSheet('font-size: 20px; font-weight: bold;')
        main_layout.addWidget(title_label)

        # Realtime Message Data
        realtime_label = QLabel('Realtime Message Data')
        realtime_label.setStyleSheet('font-size: 16px; font-weight: bold;')
        main_layout.addWidget(realtime_label)

        # Dates and times
        date_label = QLabel('March 27th 2019\n0:26:08 +02:05 - 09:59')
        main_layout.addWidget(date_label)

        # In 31.28
        in_label = QLabel('In 31.28\n0:22:04')
        main_layout.addWidget(in_label)

        # Last 100 seconds
        last_label = QLabel('Last 100 seconds\n- Last 20 minutes\n- Last 24 hours')
        main_layout.addWidget(last_label)

        # Upload your data
        upload_label = QLabel('Upload your data\n- 400\n  - 200\n  - 100')
        main_layout.addWidget(upload_label)

        # Latest 100 minutes
        latest_label = QLabel('Latest 100 minutes\n- Latest 20 minutes\n- Latest 24 hours')
        main_layout.addWidget(latest_label)

        # Latest 50 minutes
        latest50_label = QLabel('Latest 50 minutes\n- Latest 25 minutes\n- Latest 25 hours')
        main_layout.addWidget(latest50_label)

        # Latest 100 minutes
        latest100_label = QLabel('Latest 100 minutes\n- Latest 20 minutes\n- Latest 24 hours')
        main_layout.addWidget(latest100_label)

        # Buttons layout
        buttons_layout = QHBoxLayout()

        # Upload button
        upload_button = QPushButton('Upload Data')
        buttons_layout.addWidget(upload_button)

        # Refresh button
        refresh_button = QPushButton('Refresh')
        buttons_layout.addWidget(refresh_button)

        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SlowDownApp()
    ex.show()
    sys.exit(app.exec_())