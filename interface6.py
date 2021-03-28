import datetime
import sys
import pickle

from PyQt5 import QtCore, uic, QtGui
from PyQt5.QtCore import pyqtSlot, QTimer, QObject, Qt, QPoint, QRect, QPointF, QLineF, QRectF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
import pyqtgraph as pg

import numpy as np
import os

qtcreator_file = "interface.ui"  # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


# 曲线图的自定义时间横轴
class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLabel(text='Time', units=None)
        self.enableAutoSIPrefix(False)

    # 此处values为原始默认的横轴刻度，即1，2，3等等；
    # 我们使用的时间轴数据为时间戳（浮点类型），所以用fromtimestamp转换成datetime类型，然后再用strftime转换成字符串类型；strftime括号内
    # 是可更改的时间显示格式
    def tickStrings(self, values, scale, spacing):
        return [datetime.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S") for value in values]


# 像素图的x、y坐标轴
class ScaledAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 每个像素点被放大了50倍，此处除以50
    def tickStrings(self, values, scale, spacing):
        values = [int(value / 50) for value in values]
        return [value for value in values]


# 后台处理数据的线程
class Worker(QObject):
    # 被传送到GUI的信号
    frame_processed = QtCore.pyqtSignal(tuple)  # 第一帧已处理
    curves_added = QtCore.pyqtSignal(tuple)  # 曲线图信息已添加
    param_acquired = QtCore.pyqtSignal(tuple)  # 参数以获取
    all_done = QtCore.pyqtSignal()  # 进程已完毕

    # 通过数据处理脚本的函数yield结果传送数据；仅用于实时处理&监控
    def run(self, path):
        import intelligent_ground_baseline122 as dp
        dp.main(path)
        frames = dp.frames_gen
        for frame in frames:
            if len(frame) == 2:
                self.frame_processed.emit(frame)
            elif len(frame) == 5:
                height = int(frame[0])
                width = int(frame[1])
                down_thresh = frame[2]
                b_thickness = frame[3]
                pixelsize = frame[4]
                self.param_acquired.emit((height, width, down_thresh, b_thickness, pixelsize))
            elif len(frame) == 4:
                time_stamps = np.array([x.timestamp() for x in frame[0]])
                img_data_list = np.array(frame[1])
                baseline = np.array(frame[2])
                mirror_baseline = np.array(frame[3])
                self.curves_added.emit((time_stamps, img_data_list, baseline, mirror_baseline))
        print('done')
        self.all_done.emit()


class Monitor(QMainWindow, Ui_MainWindow):
    # 信号
    process_requested = QtCore.pyqtSignal(str)  # 已请求处理数据，传送到worker
    frame_generated = QtCore.pyqtSignal()  # 第一帧已处理，GUI函数用

    def __init__(self, id_track={}, frame_names=[], time_range=[], raw_data=[], baseline=[], mirror_baseline=[], height=8, width=32, is_started_from_outside=False):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)

        self.setupUi(self)
        self.setWindowTitle('Data Processor')
        self.logFilePath = None
        self.p_logFilePath = None
        self.pixelImgPath = None
        self.curveImgPath = None
        self.actionOpen.triggered.connect(self.open_file)
        self.actionOpen_processed_data.triggered.connect(self.open_processed_data)
        self.fileNameDisplay.adjustSize()

        # 像素图
        self.height = height
        self.width = width
        self.down_thresh = 100
        self.b_thickness = 1
        self.pixelsize = 50

        self.nextFrame.clicked.connect(self.next_frame)
        self.prevFrame.clicked.connect(self.prev_frame)
        self.frames = frame_names
        self.id_track = id_track
        self.frame_counter = 0

        self.setPViewAxis()

        # 曲线图
        self.curve_row_num = self.YCoordBox.value()
        self.curve_col_num = self.XCoordBox.value()
        self.img_data_list = raw_data
        self.baseline = baseline
        self.mirror_baseline = mirror_baseline
        self.time_range = time_range

        self.time_range_shown = []
        self.img_data_shown = []
        self.baseline_shown = []
        self.m_baseline_shown = []

        self.YCoordBox.setPrefix("Y Coordinate: ")
        self.XCoordBox.setPrefix("X Coordinate: ")
        self.YCoordBox.setRange(0, self.height-1)
        self.YCoordBox.setWrapping(True)
        self.XCoordBox.setRange(0, self.width-1)
        self.XCoordBox.setWrapping(True)
        self.YCoordBox.valueChanged.connect(self.curveImg.enableAutoRange)
        self.XCoordBox.valueChanged.connect(self.curveImg.enableAutoRange)
        self.YCoordBox.valueChanged.connect(self.post_update_curve_label)
        self.XCoordBox.valueChanged.connect(self.post_update_curve_label)

        self.img_data_list_graph = self.curveImg.plot(self.img_data_list, pen='r',
                                                      symbol='o',
                                                      symbolPen='r', symbolBrush=0.5,
                                                      name='{},{}'.format(0, 0), linewidth=1)
        self.baseline_graph = self.curveImg.plot(self.baseline, pen='g',
                                                 name='{}'.format('filter'), linewidth=1)
        self.mirror_baseline_graph = self.curveImg.plot(self.mirror_baseline,
                                                        pen='b', name='{}'.format('filter'),
                                                        linewidth=1)
        self.curveImg.setAxisItems(axisItems={'bottom': TimeAxisItem(orientation='bottom')})

        self.curve_counter = 0

        # 子线程
        self.worker = Worker()
        self.worker_thread = QtCore.QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()
        self.worker.frame_processed.connect(self.add_to_frame_list)
        self.worker.curves_added.connect(self.update_curve_data)
        self.worker.param_acquired.connect(self.update_params)
        self.worker.all_done.connect(self.post_update_curve_label)
        self.worker.all_done.connect(self.update_finished)
        self.worker.all_done.connect(self.save_pixel_map_data)
        self.worker.all_done.connect(self.save_curve_data)
        self.process_requested.connect(self.worker.run)

        # 其他标签、信号连接
        self.frame_generated.connect(self.show_frame)
        self.started_processing = False
        self.startProcessing.clicked.connect(self.start_processing)
        self.all_done = False
        self.open_processed = False

        self.started_from_outside = is_started_from_outside
        #
        # if self.started_from_outside:
        #     self.open_processed_data()

    @pyqtSlot()
    def open_file(self):
        self.open_processed = False
        filename, fil = QFileDialog.getOpenFileName(self, 'Open log file', filter="Log files (*.log)")
        if filename:
            self.logFilePath = filename
            self.fileNameDisplay.setText("Currently loaded filename: " + self.logFilePath)
            self.fileNameDisplay.adjustSize()
            self.started_processing = False

    @pyqtSlot()
    def open_processed_data(self):
        if not self.started_from_outside:
            self.open_processed = True
            filename, fil = QFileDialog.getOpenFileName(self, 'Open processed log file', filter="Log files (*.log)")
            if filename:
                if os.path.isdir(filename[:-4]):
                    self.p_logFilePath = filename[:-4]
                    self.fileNameDisplay.setText(
                        "Currently loaded processed filename: " + self.p_logFilePath + " (already "
                                                                                       "processed)")
                    self.curveImgPath = os.path.join(self.p_logFilePath, 'curve_img')
                    self.pixelImgPath = os.path.join(self.p_logFilePath, 'diff_result_img')
                    print(self.p_logFilePath, self.curveImgPath)
                    info = os.path.join(self.curveImgPath, 'info.csv')
                    with open(info) as i:
                        self.height = int(i.readline())
                        self.width = int(i.readline())
                    self.update_curve_img_from_csv()
                    self.update_pixel_map_from_pickle()
                else:
                    self.fileNameDisplay.setText("This log file has not been processed")
        else:
            self.open_processed = True
            self.update_curve_img_from_csv()
            self.update_pixel_map_from_csv()

    # 开始后台处理数据
    def start_processing(self):
        if self.started_processing:
            self.started_processing = False
            self.all_done = False
            self.frames = []
            self.frame_counter = 0
            self.curve_counter = 0
            self.time_range = list(range(0))
            self.img_data_list = []
            self.baseline = []
            self.mirror_baseline = []
            self.time_range_shown = []
            self.img_data_shown = []
            self.baseline_shown = []
            self.m_baseline_shown = []
            self.curveImg.clear()
        if not self.logFilePath:
            self.fileNameDisplay.setText("No file for processing!")
            self.fileNameDisplay.adjustSize()
        else:
            self.started_processing = True
            """ 
            CODE BELOW IMPLEMENTED IN DP TO SPEED UP UI.
            """
            filename = os.path.basename(self.logFilePath)
            fn_no_extension = os.path.splitext(filename)[0]
            dirname = os.path.dirname(self.logFilePath)
            self.pixelImgPath = os.path.normpath(os.path.join(dirname, fn_no_extension, 'diff_result_img'))
            self.curveImgPath = os.path.join(os.path.join(dirname, fn_no_extension, 'curve_img'))
            self.process_requested.emit(self.logFilePath)

    # 设置像素图轴
    def setPViewAxis(self):
        self.vb = self.pView.getViewBox()
        self.vb.invertY()
        self.pView.setAxisItems(axisItems={'top': ScaledAxisItem(orientation='top')})
        self.pView.showAxis('top')
        self.pView.hideAxis('bottom')
        self.pView.setAxisItems(axisItems={'left': ScaledAxisItem(orientation='left')})

        ay = self.pView.getAxis('left')
        ax = self.pView.getAxis('top')
        ay.setTickSpacing(50, 50)
        ax.setTickSpacing(50, 50)

    # 在像素图上生成网格画布
    def draw_canvas(self):
        self.pView.resize(self.width * self.pixelsize, self.height * self.pixelsize)
        pen = QtGui.QPen()
        pen.setColor(QColor(255, 255, 255))
        for i in range(1, self.width):
            pt1 = QPointF(self.pixelsize * (i - 0.5), - 0.5 * self.pixelsize)
            pt2 = QPointF(self.pixelsize * (i - 0.5), (self.height - 0.5) * self.pixelsize)
            line = QLineF(pt1, pt2)
            l = pg.QtGui.QGraphicsLineItem(line)
            l.setPen(pen)
            self.vb.addItem(l)
        for j in range(1, self.height):
            pt1 = QPointF(-0.5 * self.pixelsize, self.pixelsize * (j - 0.5))
            pt2 = QPointF((self.width - 0.5) * self.pixelsize, self.pixelsize * (j - 0.5))
            line = QLineF(pt1, pt2)
            l = pg.QtGui.QGraphicsLineItem(line)
            l.setPen(pen)
            self.vb.addItem(l)

    # 绘制某一帧的像素图
    def draw_pixel_map(self, frame_name):
        """

        :param frame_name: 帧的名称，字符串，”20201012090724047“
        """
        self.vb.clear()
        self.draw_canvas()

        pixel_brush = QtGui.QBrush()
        pixel_brush.setStyle(Qt.SolidPattern)

        for id_index in self.id_track.keys():
            if frame_name in self.id_track[id_index].keys():
                for point_index in range(len(self.id_track[id_index][frame_name])):
                    y, x = self.id_track[id_index][frame_name][point_index][0], \
                           self.id_track[id_index][frame_name][point_index][1]
                    pt1 = QPointF(self.pixelsize * (x - 0.5), self.pixelsize * (y - 0.5))
                    pt2 = QPointF(self.pixelsize * (x + 0.5), self.pixelsize * (y + 0.5))
                    r = pg.QtGui.QGraphicsRectItem(QRectF(pt1, pt2))
                    if id_index % 9 == 0:
                        pixel_brush.setColor(QColor(125 - id_index, id_index, 255 - id_index))
                    elif id_index % 8 == 0:
                        pixel_brush.setColor(QColor(255 - id_index, 125 - id_index, id_index))
                    elif id_index % 7 == 0:
                        pixel_brush.setColor(QColor(id_index, 255 - id_index, 125 - id_index))
                    elif id_index % 6 == 0:
                        pixel_brush.setColor(QColor(125 + id_index, 255 - id_index, id_index))
                    elif id_index % 5 == 0:
                        pixel_brush.setColor(QColor(id_index, 125 + id_index, 255 - id_index))
                    elif id_index % 4 == 0:
                        pixel_brush.setColor(QColor(255 - id_index, id_index, 125 + id_index))
                    elif id_index % 3 == 0:
                        pixel_brush.setColor(QColor(135 - id_index, 120 + id_index, id_index))
                    elif id_index % 2 == 0:
                        pixel_brush.setColor(QColor(id_index, 135 - id_index, 120 + id_index))
                    else:
                        pixel_brush.setColor(QColor(120 + id_index, id_index, 135 - id_index))
                    r.setBrush(pixel_brush)
                    self.vb.addItem(r)

    # 通过后台数据处理得到本次数据处理的基本参数信息，例如宽、高等
    def update_params(self, params):
        """

        :param params: 来自worker线程，当len(frame) == 5，包含了高，宽，down_thresh，像素边框粗细，像素大小。
        """
        self.height = params[0]
        self.width = params[1]
        self.down_thresh = params[2]
        self.YCoordBox.setRange(0, self.height - 1)
        self.XCoordBox.setRange(0, self.width - 1)
        self.b_thickness = params[3]
        self.pixelsize = params[4]

    # 更新帧名列表，以及跟踪字典
    def add_to_frame_list(self, frame):
        """

        :param frame: 来自worker线程，当len(frame) == 2，包含了帧名以及跟踪字典。
        """
        frame_name, self.id_track = frame[0], {**self.id_track, **frame[1]}
        self.frames.append(frame_name)
        if len(self.frames) == 1:
            self.frame_generated.emit()
            self.draw_canvas()

        self.counter_display()

    # 更新曲线图信息
    def update_curve_data(self, curve):
        """

        :param curve: 来自worker线程，当len(frame) == 4，包含了时间，原始数据，基线以及镜像基线。
        """
        self.time_range = curve[0]
        self.img_data_list = curve[1]
        self.baseline = curve[2]
        self.mirror_baseline = curve[3]

        self.update_curve_img()
        self.curve_counter += 1

    # 显示当前帧的像素图，默认为最初帧
    def show_frame(self, frame_count=0):
        if len(self.frames):
            frame_name = self.frames[frame_count]
            self.draw_pixel_map(frame_name)
            self.counter_display()

    def next_frame(self):
        if self.started_processing or self.open_processed:
            if self.frame_counter == len(self.frames) - 1:
                self.fileNameDisplay.setText("You're on the last frame! There is no next frame.")
                return
            self.frame_counter += 1
            self.show_frame(self.frame_counter)
        else:
            self.fileNameDisplay.setText("Please start processing first.")
            self.fileNameDisplay.adjustSize()

    def prev_frame(self):
        if self.started_processing or self.open_processed:
            if self.frame_counter == 0:
                self.fileNameDisplay.setText("You're on the first frame! There is no previous frame.")
                return
            self.frame_counter -= 1
            self.show_frame(self.frame_counter)
        else:
            self.fileNameDisplay.setText("Please start processing first.")
            self.fileNameDisplay.adjustSize()

    # 显示当前正在观看帧的信息
    def counter_display(self):
        self.frameCounter.setText(
            'Number of frames processed: ' + str(len(self.frames)) + "; you're looking at frame " + str(
                self.frame_counter) + ", which happened on " + datetime.datetime.strptime(
                self.frames[self.frame_counter], "%Y%m%d%H%M%S%f").strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        self.frameCounter.adjustSize()

    # 实时处理中更新曲线图数据的函数
    def update_curve_img(self):
        self.curve_row_num = self.YCoordBox.value()
        self.curve_col_num = self.XCoordBox.value()
        if len(self.time_range) <= 100:
            self.time_range_shown = self.time_range
            self.img_data_shown = self.img_data_list[:, self.curve_row_num, self.curve_col_num]
            self.baseline_shown = self.baseline[:, self.curve_row_num, self.curve_col_num]
            self.m_baseline_shown = self.mirror_baseline[:, self.curve_row_num, self.curve_col_num] - self.down_thresh
        else:
            self.time_range_shown = self.time_range[-100:]
            self.img_data_shown = self.img_data_list[:, self.curve_row_num, self.curve_col_num][-100:]
            self.baseline_shown = self.baseline[:, self.curve_row_num, self.curve_col_num][-100:]
            self.m_baseline_shown = self.mirror_baseline[:, self.curve_row_num, self.curve_col_num][
                                    -100:] - self.down_thresh
        self.update_curve_label()

    # 前一个函数自动执行此函数以更新曲线图象
    def update_curve_label(self):
        # print(self.time_range[-50], self.time_range[-1])
        if len(self.time_range) <= 50:
            self.curveImg.setXRange(min=self.time_range[0], max=self.time_range[-1], padding=0)
        else:
            self.curveImg.setXRange(min=self.time_range[-50], max=self.time_range[-1], padding=0)
        self.img_data_list_graph.setData(self.time_range_shown, self.img_data_shown)
        self.baseline_graph.setData(self.time_range_shown, self.baseline_shown)
        self.mirror_baseline_graph.setData(self.time_range_shown, self.m_baseline_shown)

    # 显示曲线图，用于打开已处理过的log文件
    def post_update_curve_label(self):
        if self.all_done and not self.open_processed:
            self.curve_row_num = self.YCoordBox.value()
            self.curve_col_num = self.XCoordBox.value()
            self.img_data_list_graph.setData(self.time_range,
                                             self.img_data_list[:, self.curve_row_num, self.curve_col_num])
            self.baseline_graph.setData(self.time_range, self.baseline[:, self.curve_row_num, self.curve_col_num])
            self.mirror_baseline_graph.setData(self.time_range,
                                               self.mirror_baseline[:, self.curve_row_num, self.curve_col_num])
        if self.open_processed:
            self.curve_row_num = self.YCoordBox.value()
            self.curve_col_num = self.XCoordBox.value()
            self.img_data_list_graph.setData(self.time_range,
                                             self.img_data_list[self.curve_row_num][self.curve_col_num])
            self.baseline_graph.setData(self.time_range, self.baseline[self.curve_row_num][self.curve_col_num])
            self.mirror_baseline_graph.setData(self.time_range,
                                               self.mirror_baseline[self.curve_row_num][self.curve_col_num])

    # 实时处理结束后更新已处理完毕的标签
    def update_finished(self):
        self.all_done = True

    # 通过生成的csv绘制完整的曲线图
    def update_curve_img_from_csv(self):
        if not self.started_from_outside:
            self.img_data_list = {}
            self.baseline = {}
            self.mirror_baseline = {}

            time = os.path.join(self.curveImgPath, 'time.csv')

            with open(time) as t:
                self.time_range = [float(x) for x in t.readlines()]

            for row_number in range(self.height):
                self.img_data_list[row_number] = {}
                self.baseline[row_number] = {}
                self.mirror_baseline[row_number] = {}
                for col_number in range(self.width):
                    self.img_data_list[row_number][col_number] = {}
                    self.baseline[row_number][col_number] = {}
                    self.mirror_baseline[row_number][col_number] = {}

            for row_number in range(self.height):
                for col_number in range(self.width):
                    file = os.path.join(self.curveImgPath, 'filename{}_{}.csv'.format(row_number, col_number))
                    with open(file) as f:
                        self.img_data_list[row_number][col_number] = [int(x.split('.')[0]) for x in f.readline().split(',')]
                        self.baseline[row_number][col_number] = [int(x.split('.')[0]) for x in f.readline().split(',')]
                        self.mirror_baseline[row_number][col_number] = [int(x.split('.')[0]) for x in f.readline().split(',')]

        self.post_update_curve_label()

    # 通过生成的pickle文件绘制像素图
    def update_pixel_map_from_pickle(self):
        if not self.started_from_outside:
            with open(os.path.join(self.pixelImgPath, 'id_track.p'), 'rb') as handle:
                self.id_track = pickle.load(handle)
            with open(os.path.join(self.pixelImgPath, 'frame_names.p'), 'rb') as handle:
                self.frames = pickle.load(handle)

        self.show_frame()

    # 用pickle保存像素图数据，及跟踪字典以及帧名
    def save_pixel_map_data(self):
        with open(os.path.join(self.pixelImgPath, 'id_track.p'), 'wb') as handle:
            pickle.dump(self.id_track, handle)
        with open(os.path.join(self.pixelImgPath, 'frame_names.p'), 'wb') as handle:
            pickle.dump(self.frames, handle)

    # 保存曲线图csv数据
    def save_curve_data(self):
        for row_j in range(self.height):
            for col_i in range(self.width):
                file = np.array([np.array(self.img_data_list)[:, row_j, col_i],
                                 np.array(self.baseline)[:, row_j, col_i],
                                 np.array(self.mirror_baseline)[:, row_j, col_i] - self.down_thresh])
                np.savetxt(os.path.join(self.curveImgPath, 'filename{}_{}.csv'.format(row_j, col_i)), file, fmt="%s",
                           delimiter=",")
        info_file = np.array([np.array(self.height), np.array(self.width)])
        np.savetxt(os.path.join(self.curveImgPath, 'info.csv'), info_file, fmt="%s", delimiter=",")
        time_file = self.time_range
        np.savetxt(os.path.join(self.curveImgPath, 'time.csv'), time_file, fmt="%s", delimiter=",")

# def show(id_track, frame_names, time_range, raw_data, baseline, mirror_baseline, height=8, width=32):
#     '''
#     input说明：
#     id_track: DICTIONARY;
#     frame_names: LIST of frame names; ex: ['202011080907138', ...];
#     time_range: LIST，包含timestamps: '%Y%m%d%H%M%S%f'.timestamp();
#     raw_data, baseline, mirror_baseline: 均为2D字典；ex: raw_data[row_number][column_number] = -16890（数据）;
#     '''
#     app = QApplication(sys.argv)
#     window = Monitor(id_track, frame_names, time_range, raw_data, baseline, mirror_baseline, height=8, width=32)
#     window.show()
#     sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Monitor()
    window.show()
    sys.exit(app.exec_())
