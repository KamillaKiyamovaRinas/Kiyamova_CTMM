import numpy as np
import os
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from typing import Callable, Dict, List, Tuple, Optional
import json

from obj_loader import load_obj_geometry

class ThermalODESolver:
    """
    Класс для решения системы ОДУ теплового баланса конечных элементов
    """

    def __init__(self):
        """Инициализация решателя"""
        self.N = 0                      # Количество элементов
        self.c = []                      # Теплоемкости [c1, c2, ..., cN]
        self.epsilon = []                 # Коэффициенты излучения
        self.lambda_matrix = None         # Матрица теплопроводностей λ_ij
        self.S_matrix = None              # Матрица площадей сечений S_ij
        self.S_surface = []                # Площади поверхностей S_i
        self.Q_R = []                      # Внутренние источники тепла
        self.C0 = 5.67                     # Постоянная Стефана-Больцмана
        
        # Параметры для временной зависимости (для Q5_R)
        self.A = 1.0                       # Параметр A для функции sin

    def load_from_json(self, json_path: str, obj_data: Dict = None):
        """
        Загрузка данных из JSON файла
        
        Args:
            json_path: путь к JSON файлу конфигурации
            obj_data: данные из OBJ файла (геометрия)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Определяем количество элементов
        if 'metadata' in config and 'num_elements' in config['metadata']:
            self.N = config['metadata']['num_elements']
        else:
            # Пытаемся определить из других полей
            if 'thermal_characteristics' in config:
                if 'heat_capacity' in config['thermal_characteristics']:
                    self.N = len(config['thermal_characteristics']['heat_capacity'])
        
        print(f"Загружено {self.N} элементов")
        # Загрузка тепловых связей
        self._init_thermal_matrices()
        
        # Загрузка теплоемкостей
        if 'thermal_characteristics' in config:
            tc = config['thermal_characteristics']
            # Теплоемкости
            if 'heat_capacity' in tc:
                if isinstance(tc['heat_capacity'], dict):
                    # Формат {"c1": 900, "c2": 900, ...}
                    self.c = [tc['heat_capacity'].get(f'c{i+1}', 0) for i in range(self.N)]
                elif isinstance(tc['heat_capacity'], list):
                    self.c = tc['heat_capacity']
            
            # Коэффициенты излучения
            if 'emissivity' in tc:
                if isinstance(tc['emissivity'], dict):
                    self.epsilon = [tc['emissivity'].get(f'epsilon{i+1}', 0) for i in range(self.N)]
                elif isinstance(tc['emissivity'], list):
                    self.epsilon = tc['emissivity']
        
        
        if 'thermal_connections' in config:
            connections = config['thermal_connections']
            
            # Инициализируем матрицу теплопроводностей
            self.lambda_matrix = np.zeros((self.N, self.N))
            
            # Заполняем связи
            for key, value in connections.items():
                if value is None or value == 0:
                    continue
                    
                # Парсим ключ вида "lambda_12" или "12"
                if 'lambda_' in key:
                    nums = key.replace('lambda_', '').split('_')
                else:
                    nums = key.split('_')
                
                if len(nums) == 2:
                    try:
                        i, j = int(nums[0]) - 1, int(nums[1]) - 1  # к 0-индексации
                        self.lambda_matrix[i, j] = value
                        self.lambda_matrix[j, i] = value  # симметричная матрица
                    except (ValueError, IndexError):
                        print(f"Ошибка в формате связи: {key}")
        
        # Загрузка источников тепла
        if 'heat_sources' in config:
            heat_sources = config['heat_sources']
            self.Q_R = [0] * self.N
            
            for key, value in heat_sources.items():
                # Парсим ключи вида "Q1_R", "Q2_R" или "Q1", "Q2"
                nums = ''.join([c for c in key if c.isdigit()])
                if nums:
                    try:
                        idx = int(nums) - 1
                        if isinstance(value, (int, float)):
                            self.Q_R[idx] = value
                        elif isinstance(value, str):
                            self.Q_R[idx] = value
                            # if 'A' in value and idx == 4:
                            #     self.A = 1.0
                    except (ValueError, IndexError):
                        print(f"Ошибка в источнике тепла: {key}")
        
        # Загрузка начальных температур
        if 'initial_temperatures' in config:
            init_temps = config['initial_temperatures']
            if isinstance(init_temps, list):
                self.T0 = np.array(init_temps)
            elif isinstance(init_temps, dict):
                if 'values' in init_temps:
                    values = init_temps['values']
                    self.T0 = np.array([values.get(f'T{i+1}', 300) for i in range(self.N)])
                else:
                    self.T0 = np.array([init_temps.get(f'T{i+1}', 300) for i in range(self.N)])
        else:
            self.T0 = np.ones(self.N) * 300  # По умолчанию 300K

    
    def _init_thermal_matrices(self):
        """Инициализация матриц тепловых параметров"""
        self.lambda_matrix = np.zeros((self.N, self.N))
        self.S_matrix = np.zeros((self.N, self.N))
        self.S_surface = np.zeros(self.N)

    def update_from_obj(self, obj_data: Dict):
        """
        Обновление геометрических параметров из OBJ файла
        
        Args:
            obj_data: словарь с данными из OBJ файла
            Должен содержать:
            - 'S_surface': площади поверхностей для каждого элемента
            - 'S_connections': матрица площадей сечений между элементами
        """
        if 'S_surface' in obj_data:
            self.S_surface = np.array(obj_data['S_surface'])
        
        if 'S_connections' in obj_data:
            self.S_matrix = np.array(obj_data['S_connections'])
    
    
    def heat_source_func(self, t: float, element_idx: int) -> float:
        """
        Вычисление источника тепла Q_i^R(t) для элемента
        
        Args:
            t: время
            element_idx: индекс элемента (0-based)
        
        Returns:
            значение Q_i^R в момент t
        """
        Q = self.Q_R[element_idx]
        
        # Если это число, возвращаем как есть
        if isinstance(Q, (int, float)):
            return Q
        
        # Если это строка с функцией, вычисляем
        if isinstance(Q, str):
            # Для элемента 5: Q5_R = A*(20 + 3*sin(t/4))
            try:
                # Заменяем переменные
                expr = Q.replace('A', str(self.A))
                expr = expr.replace('t', str(t))
                expr = expr.replace('sin', 'np.sin')
                return eval(expr)
                # self.A * (20 + 30*np.sin(t/4))
            except:
                return 0.0
        
        return 0.0
    
    def ode_system(self, t: float, T: np.ndarray) -> np.ndarray:
        """
        Правая часть системы ОДУ
        dT_i/dt = (1/c_i) * [sum(Q_ij^TC) + Q_i^E + Q_i^R(t)]
        
        Args:
            t: время
            T: вектор температур [T1, T2, ..., TN]
        
        Returns:
            вектор производных dT/dt
        """
        dTdt = np.zeros(self.N)
        
        for i in range(self.N):
            # Теплопроводность:
            Q_conduct = 0.0
            for j in range(self.N):
                if i != j and self.lambda_matrix[i, j] > 0:
                    # коэффициент тепловой связи
                    k_ij = self.lambda_matrix[i, j] * self.S_matrix[i, j]
                    Q_conduct += -k_ij * (T[j] - T[i])
            
            # Излучение
            Q_rad = -self.epsilon[i] * self.S_surface[i] * self.C0 * (T[i] / 100) ** 4
            
            # Внутренний источник
            Q_source = self.heat_source_func(t, i)
            #print(f"Источник: {Q_source}")
            
            # Суммарный поток и производная
            total_flow = Q_conduct + Q_rad + Q_source
            dTdt[i] = total_flow / self.c[i] if self.c[i] > 0 else 0
        return dTdt
    
    def solve_finite_time(self, t_span: Tuple[float, float], 
                         T0: Optional[np.ndarray] = None,
                         method: str = 'RK45',
                         rtol: float = 1e-6,
                         atol: float = 1e-9) -> Dict:
        """
        Решение для конечного временного интервала
        
        Args:
            t_span: (t_start, t_end)
            T0: начальные температуры (если None, используются из конфига)
            method: метод решения ('RK45', 'Radau', 'BDF', etc.)
            rtol: относительная точность
            atol: абсолютная точность
        
        Returns:
            словарь с результатами {'t': times, 'y': temperatures}
        """
        if T0 is None:
            T0 = self.T0
        
        # Решаем ОДУ
        solution = solve_ivp(
            self.ode_system,
            t_span,
            T0,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=True  # для интерполяции
        )
        
        return {
            't': solution.t,
            'y': solution.y,
            'success': solution.success,
            'message': solution.message
        }
    
    def find_stationary_solution(self, T_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Поиск стационарного решения (когда dT/dt = 0)
        
        Args:
            T_guess: начальное приближение
        
        Returns:
            стационарные температуры
        """
        if T_guess is None:
            T_guess = self.T0
        
        # Определяем функцию для fsolve (правая часть = 0)
        def stationary_func(T):
            return self.ode_system(0, T)
        
        # Ищем решение
        T_stat = fsolve(stationary_func, T_guess)
        
        return T_stat
    
    def solve_infinite_time(self, initial_T: np.ndarray, 
                           time_window: float = 100.0,
                           step: float = 1.0,
                           callback: Optional[Callable] = None):
        """
        Генератор для бесконечного времени (движущееся окно)
        
        Args:
            initial_T: начальные температуры
            time_window: размер временного окна
            step: шаг по времени
            callback: функция обратного вызова для каждого шага
        
        Yields:
            результаты для каждого временного окна
        """
        t_current = 0.0
        T_current = initial_T.copy()
        
        while True:
            # Решаем на небольшом интервале
            t_span = (t_current, t_current + time_window)
            result = self.solve_finite_time(t_span, T_current)
            
            # Обновляем текущее время и температуру
            t_current += step
            T_current = result['y'][:, -1]  # последняя температура
            
            if callback:
                callback(t_current, T_current, result)
            
            yield {
                't': result['t'],
                'y': result['y'],
                'current_time': t_current,
                'current_temp': T_current
            }
    
    def save_results_to_csv(self, t: np.ndarray, y: np.ndarray, filename: str):
        """
        Сохранение результатов в CSV файл
        
        Args:
            t: массив времен
            y: матрица температур (строки - время, столбцы - элементы)
            filename: имя файла
        """
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            header = ['Time'] + [f'T{i+1}' for i in range(self.N)]
            writer.writerow(header)

            for i, time in enumerate(t):
                row = [time] + list(y[:, i])
                writer.writerow(row)
        
        print(f"Результаты сохранены в {filename}")