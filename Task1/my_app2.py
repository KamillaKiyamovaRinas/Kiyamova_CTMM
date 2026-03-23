import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QFileDialog, QGroupBox, QRadioButton, QButtonGroup,
                             QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget,
                             QSplitter, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Импортируем ваши модули
from ode_solver import ThermalODESolver
from obj_loader import load_obj_geometry

class MplCanvas(FigureCanvas):
    """Виджет для отображения графиков matplotlib"""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.solver = ThermalODESolver()

        # Пути к файлам
        self.obj_path = None
        self.json_path = None

        # Данные геометрии
        self.obj_data = None
        self.config_data = None
        
        # Результаты расчёта
        self.results = None
        self.infinite_solver = None

        self.initUI()
        
    def initUI(self):
        # Настройки главного окна
        self.setWindowTitle("Тепловой расчет КА")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Сплиттер для разделения левой панели и графика
        splitter = QSplitter()
        main_layout.addWidget(splitter)
        
        # Левая панель
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)

        # Группа загрузки файлов
        files_group = QGroupBox("Загрузка файлов")
        files_layout = QVBoxLayout()
        
        # Загрузка OBJ файла
        obj_layout = QHBoxLayout()
        self.obj_path_label = QLabel("Файл модели не выбран")
        self.obj_path_label.setWordWrap(True)
        obj_btn = QPushButton("Загрузить .obj")
        obj_btn.clicked.connect(self.load_obj_file)
        obj_layout.addWidget(obj_btn)
        obj_layout.addWidget(self.obj_path_label)
        
        # Загрузка JSON конфигурации
        json_layout = QHBoxLayout()
        self.json_path_label = QLabel("Файл конфигурации не выбран")
        self.json_path_label.setWordWrap(True)
        json_btn = QPushButton("Загрузить .json")
        json_btn.clicked.connect(self.load_json_file)
        json_layout.addWidget(json_btn)
        json_layout.addWidget(self.json_path_label)
        
        files_layout.addLayout(obj_layout)
        files_layout.addLayout(json_layout)
        files_group.setLayout(files_layout)
        
        # Группа параметров расчета времени
        time_group = QGroupBox("Параметры времени расчета")
        time_layout = QVBoxLayout()
        
        # Выбор типа расчета
        calc_type_layout = QHBoxLayout()
        calc_type_label = QLabel("Тип расчета:")
        self.calc_type_group = QButtonGroup()
        
        self.finite_time_radio = QRadioButton("Конечное время")
        self.infinite_time_radio = QRadioButton("Бесконечное (движущееся окно)")
        self.finite_time_radio.setChecked(True)
        
        self.calc_type_group.addButton(self.finite_time_radio)
        self.calc_type_group.addButton(self.infinite_time_radio)
        
        calc_type_layout.addWidget(calc_type_label)
        calc_type_layout.addWidget(self.finite_time_radio)
        calc_type_layout.addWidget(self.infinite_time_radio)
        
        # Параметры конечного времени
        finite_time_layout = QHBoxLayout()
        finite_time_label = QLabel("Время расчета (с):")
        self.finite_time_spin = QDoubleSpinBox()
        self.finite_time_spin.setRange(0, 1e6)
        self.finite_time_spin.setValue(3600)
        self.finite_time_spin.setSingleStep(100)
        
        finite_time_layout.addWidget(finite_time_label)
        finite_time_layout.addWidget(self.finite_time_spin)
        
        # Параметры бесконечного времени
        infinite_layout = QHBoxLayout()
        self.speed_label = QLabel("Шаг по времени (с):")
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 100)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setEnabled(False)
        
        infinite_layout.addWidget(self.speed_label)
        infinite_layout.addWidget(self.speed_spin)
        
        # Параметры окна для бесконечного времени
        window_layout = QHBoxLayout()
        self.window_label = QLabel("Размер окна (с):")
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(1, 1000)
        self.window_spin.setValue(100.0)
        self.window_spin.setSingleStep(10)
        self.window_spin.setEnabled(False)
        
        window_layout.addWidget(self.window_label)
        window_layout.addWidget(self.window_spin)
        
        self.finite_time_radio.toggled.connect(self.toggle_calc_type)
        
        time_layout.addLayout(calc_type_layout)
        time_layout.addLayout(finite_time_layout)
        time_layout.addLayout(infinite_layout)
        time_layout.addLayout(window_layout)
        time_group.setLayout(time_layout)
        
        # Группа начальных условий
        init_group = QGroupBox("Начальные условия")
        init_layout = QVBoxLayout()
        
        self.init_from_file_radio = QRadioButton("Из файла конфигурации")
        self.init_stationary_radio = QRadioButton("Стационарное решение")
        self.init_from_file_radio.setChecked(True)
        
        init_layout.addWidget(self.init_from_file_radio)
        init_layout.addWidget(self.init_stationary_radio)
        init_group.setLayout(init_layout)
        
        # Группа управления расчетом
        control_group = QGroupBox("Управление расчетом")
        control_layout = QVBoxLayout()
        
        buttons_layout = QHBoxLayout()
        self.calc_btn = QPushButton("Начать расчет")
        self.calc_btn.clicked.connect(self.start_calculation)
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.clicked.connect(self.stop_calculation)
        self.stop_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.calc_btn)
        buttons_layout.addWidget(self.stop_btn)
        
        # Кнопка сохранения результатов
        self.save_btn = QPushButton("Сохранить результаты в CSV")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        
        control_layout.addLayout(buttons_layout)
        control_layout.addWidget(self.save_btn)
        control_group.setLayout(control_layout)
        
        # Группа информации о модели
        info_group = QGroupBox("Информация о модели")
        info_layout = QVBoxLayout()
        self.info_text = QLabel("Модель не загружена")
        self.info_text.setWordWrap(True)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        
        # Добавляем все группы на левую панель
        left_layout.addWidget(files_group)
        left_layout.addWidget(time_group)
        left_layout.addWidget(init_group)
        left_layout.addWidget(control_group)
        left_layout.addWidget(info_group)
        left_layout.addStretch()
        
        # Правая панель с графиком
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Создаем холст для графика
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.canvas.axes.set_xlabel("Время (с)")
        self.canvas.axes.set_ylabel("Температура (K)")
        self.canvas.axes.set_title("Динамика изменения температур КЭ")
        self.canvas.axes.grid(True)
        self.canvas.fig.tight_layout()
        
        right_layout.addWidget(self.canvas)
        
        # Добавляем панели в сплиттер
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Устанавливаем начальные размеры панелей (30% - левая, 70% - правая)
        splitter.setSizes([300, 700])
        
        # Таймер для обновления графика (для бесконечного расчета)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_infinite_plot)
        
    def toggle_calc_type(self):
        """Переключение между конечным и бесконечным временем расчета"""
        is_finite = self.finite_time_radio.isChecked()
        self.finite_time_spin.setEnabled(is_finite)
        self.speed_spin.setEnabled(not is_finite)
        self.window_spin.setEnabled(not is_finite)
        
    def load_obj_file(self):
        """Загрузка .obj файла космического аппарата"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл модели",
            "",
            "OBJ Files (*.obj)"
        )

        if file_path:
            try:
                self.obj_path = file_path
                self.obj_path_label.setText(file_path)

                self.obj_data = load_obj_geometry(file_path)

                self.update_model_info()

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
            
    def load_json_file(self):
        """Загрузка .json файла конфигурации"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите файл конфигурации", 
            "", 
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.json_path = file_path
            self.json_path_label.setText(file_path)
            
    def update_model_info(self):
        """Обновление информации о модели"""
        if self.obj_data is None:
            return

        self.info_text.setText(
            "Модель загружена\n"
            f"Количество элементов: {self.obj_data['num_elements']}\n"
            f"Вершин: {self.obj_data['vertices_count']}\n"
            f"Граней: {self.obj_data['faces_count']}"
        )
        
    def start_calculation(self):
        """Начать расчета"""
        if self.obj_path is None:
            QMessageBox.warning(self, "Ошибка", "Загрузите OBJ файл")
            return

        if self.json_path is None:
            QMessageBox.warning(self, "Ошибка", "Загрузите JSON файл")
            return

        try:

            self.solver.load_from_json(self.json_path)
            self.solver.update_from_obj(self.obj_data)

            if self.init_stationary_radio.isChecked():
                T0 = self.solver.find_stationary_solution()
            else:
                T0 = self.solver.T0

            # -------- конечное время --------
            if self.finite_time_radio.isChecked():

                t_end = self.finite_time_spin.value()

                self.results = self.solver.solve_finite_time(
                    t_span=(0, t_end),
                    T0=T0
                )

                self.plot_results(self.results)

            # -------- бесконечное время --------
            else:

                self.results = {
                    "t": [],
                    "y": []
                }

                self.infinite_solver = self.solver.solve_infinite_time(
                    initial_T=T0,
                    time_window=self.window_spin.value(),
                    step=self.speed_spin.value()
                )

                self.timer.start(200)

            self.calc_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
        
    def stop_calculation(self):
        """Остановить расчёт"""
        self.timer.stop()

        self.calc_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        
    def update_infinite_plot(self):
        """Обновить график"""
        try:

            data = next(self.infinite_solver)

            t = data["t"]
            y = data["y"]

            # сохраняем историю
            if len(self.results["t"]) == 0:
                self.results["t"] = list(t)
                self.results["y"] = y
            else:
                self.results["t"].extend(t)
                self.results["y"] = np.hstack((self.results["y"], y))

            self.canvas.axes.clear()

            N = self.results["y"].shape[0]

            #print(self.results["t"])

            #t_plot = self.results["t"][-self.window_spin.value():]
            #y_plot = self.results["y"][:, -self.window_spin.value():]

            for i in range(N):
                self.canvas.axes.plot(t, y[i], label=f"T{i+1}")

            self.canvas.axes.set_xlabel("Время (с)")
            self.canvas.axes.set_ylabel("Температура (K)")
            self.canvas.axes.set_title("Температура элементов (реальное время)")
            self.canvas.axes.legend()
            self.canvas.axes.grid(True)

            self.canvas.draw()

        except StopIteration:
            self.stop_calculation()
        
    def plot_results(self, results):
        """Отрисовка результатов расчета"""
        self.canvas.axes.clear()
        
        t = results['t']
        y = results['y']
        
        # Отрисовываем все элементы
        for i in range(y.shape[0]):
            self.canvas.axes.plot(t, y[i, :], label=f'Элемент {i+1}')
        
        self.canvas.axes.set_xlabel("Время (с)")
        self.canvas.axes.set_ylabel("Температура (K)")
        self.canvas.axes.set_title("Динамика изменения температур")
        self.canvas.axes.grid(True)
        self.canvas.axes.legend(loc='upper right', fontsize='small')
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
    def save_results(self):
        """Сохранить в .csv"""
        if self.results is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результаты",
            "",
            "CSV Files (*.csv)"
        )

        if file_path:

            t = np.array(self.results["t"])
            y = np.array(self.results["y"])

            self.solver.save_results_to_csv(t, y, file_path)

            QMessageBox.information(self, "Информация", "Файл сохранён")

def main():
    app = QApplication(sys.argv)
    
    # Устанавливаем стиль приложения
    app.setStyle('Fusion')
    
    # Создаем и показываем главное окно
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()