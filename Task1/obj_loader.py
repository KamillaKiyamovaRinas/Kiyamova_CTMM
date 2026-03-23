import numpy as np
from collections import defaultdict
import traceback

def parse_obj(filename):
    """
    Парсит .obj файл и возвращает словарь с информацией о КЭ
    """
    vertices = []
    elements = {}
    
    current_element = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            if parts[0] == 'v':
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
                
            elif parts[0] == 'g':
                current_element = parts[1] if len(parts) > 1 else f"element_{len(elements)}"
                elements[current_element] = []
                
            elif parts[0] == 'f' and current_element is not None:
                # Грань: f v1 v2 v3 ...
                face_vertices = []
                for p in parts[1:]:
                    vertex_idx = int(p.split('/')[0]) - 1
                    face_vertices.append(vertex_idx)

                for i in range(1, len(face_vertices) - 1):
                    triangle = [face_vertices[0], face_vertices[i], face_vertices[i+1]]
                    elements[current_element].append(triangle)
    
    return vertices, elements

def triangle_area(v1, v2, v3):
    a = np.array(v1)
    b = np.array(v2)
    c = np.array(v3)

    ab = b - a
    ac = c - a

    cross_product = np.cross(ab, ac)
    area = 0.5 * np.linalg.norm(cross_product)
    return area

def calculate_surface_areas(vertices, elements):
    S_i = {}
    
    for element_name, faces in elements.items():
        total_area = 0.0
        for face in faces:
            v1, v2, v3 = face
            area = triangle_area(
                vertices[v1],
                vertices[v2], 
                vertices[v3]
            )
            total_area += area
        S_i[element_name] = total_area
    
    return S_i


# Для рассчёта calculate S_ij нужно всё сделать по-другому
# 1. Найти, есть ли у КЭ_i и КЭ_j грани, которые лежат в одной плоскости по x, y, z
# 2. Для множеств таких "граней" из КЭ_i и КЭ_j найти площадь их пересечения
# 3. Этот способ учитывает случаи, когда у конечных элементов нет совпадающих вершин. Поскольку это
#    ещё не значит, что S_ij = 0
def calculate_Sij(vertices, elements):
    """
    Вычисляет матрицу площадей сечения S_ij между конечными элементами
    
    Args:
        vertices: список вершин [(x,y,z), ...]
        elements: словарь {имя_элемента: список треугольников}
    
    Returns:
        словарь {(i,j): площадь_сечения} для всех пар элементов
    """
    # Преобразуем элементы в список для индексации
    element_names = list(elements.keys())
    n_elements = len(element_names)
    
    # Словарь для хранения граней каждого элемента с их плоскостями
    # Для каждой грани храним: тип плоскости (x/y/z), значение константы, и ограничивающий прямоугольник
    element_faces = {}
    
    for elem_name in element_names:
        triangles = elements[elem_name]
        faces = []
        
        for tri in triangles:
            # Получаем координаты вершин треугольника
            v0, v1, v2 = [vertices[idx] for idx in tri]
            
            # Проверяем, параллельна ли грань координатной плоскости
            # Для этого все вершины должны иметь одинаковую координату по одной оси
            
            # Проверка X = const
            if abs(v0[0] - v1[0]) < 1e-10 and abs(v0[0] - v2[0]) < 1e-10:
                # Грань параллельна YZ плоскости
                x_const = v0[0]
                # Находим bounding box по Y и Z
                y_min = min(v0[1], v1[1], v2[1])
                y_max = max(v0[1], v1[1], v2[1])
                z_min = min(v0[2], v1[2], v2[2])
                z_max = max(v0[2], v1[2], v2[2])
                faces.append(('x', x_const, (y_min, y_max, z_min, z_max)))
                
            # Проверка Y = const
            elif abs(v0[1] - v1[1]) < 1e-10 and abs(v0[1] - v2[1]) < 1e-10:
                # Грань параллельна XZ плоскости
                y_const = v0[1]
                x_min = min(v0[0], v1[0], v2[0])
                x_max = max(v0[0], v1[0], v2[0])
                z_min = min(v0[2], v1[2], v2[2])
                z_max = max(v0[2], v1[2], v2[2])
                faces.append(('y', y_const, (x_min, x_max, z_min, z_max)))
                
            # Проверка Z = const
            elif abs(v0[2] - v1[2]) < 1e-10 and abs(v0[2] - v2[2]) < 1e-10:
                # Грань параллельна XY плоскости
                z_const = v0[2]
                x_min = min(v0[0], v1[0], v2[0])
                x_max = max(v0[0], v1[0], v2[0])
                y_min = min(v0[1], v1[1], v2[1])
                y_max = max(v0[1], v1[1], v2[1])
                faces.append(('z', z_const, (x_min, x_max, y_min, y_max)))
        
        element_faces[elem_name] = faces
    
    # Вычисляем площади пересечений для всех пар элементов
    S_ij = {}
    #S_ij = np.zeros((len(element_names), len(element_names)))
    
    for i in range(n_elements):
        for j in range(i+1, n_elements):
            elem_i = element_names[i]
            elem_j = element_names[j]
            
            total_area = 0
            
            # Проверяем все пары граней
            for face_i in element_faces[elem_i]:
                for face_j in element_faces[elem_j]:
                    # Грани должны быть параллельны одной и той же плоскости
                    if face_i[0] != face_j[0]:
                        continue
                    
                    # И находиться на одной координате
                    if abs(face_i[1] - face_j[1]) > 1e-10:
                        continue
                    
                    # Находим площадь пересечения bounding box'ов
                    if face_i[0] == 'x':
                        # Для плоскости X=const пересекаем по Y и Z
                        y_min = max(face_i[2][0], face_j[2][0])
                        y_max = min(face_i[2][1], face_j[2][1])
                        z_min = max(face_i[2][2], face_j[2][2])
                        z_max = min(face_i[2][3], face_j[2][3])
                        
                        if y_min < y_max - 1e-10 and z_min < z_max - 1e-10:
                            area = (y_max - y_min) * (z_max - z_min)
                            total_area += area
                            
                    elif face_i[0] == 'y':
                        # Для плоскости Y=const пересекаем по X и Z
                        x_min = max(face_i[2][0], face_j[2][0])
                        x_max = min(face_i[2][1], face_j[2][1])
                        z_min = max(face_i[2][2], face_j[2][2])
                        z_max = min(face_i[2][3], face_j[2][3])
                        
                        if x_min < x_max - 1e-10 and z_min < z_max - 1e-10:
                            area = (x_max - x_min) * (z_max - z_min)
                            total_area += area
                            
                    elif face_i[0] == 'z':
                        # Для плоскости Z=const пересекаем по X и Y
                        x_min = max(face_i[2][0], face_j[2][0])
                        x_max = min(face_i[2][1], face_j[2][1])
                        y_min = max(face_i[2][2], face_j[2][2])
                        y_max = min(face_i[2][3], face_j[2][3])
                        
                        if x_min < x_max - 1e-10 and y_min < y_max - 1e-10:
                            area = (x_max - x_min) * (y_max - y_min)
                            total_area += area
            
            if total_area > 1e-10:  # Игнорируем очень маленькие площади
                S_ij[(elem_i, elem_j)] = total_area
                # S_ij[i][j] = total_area
                # S_ij[j][i] = total_area
    
    return S_ij


# Пример использования:
def print_Sij_matrix(S_ij, element_names):
    """
    Печатает матрицу S_ij в удобном виде
    """
    n = len(element_names)
    print("\nМатрица площадей сечения S_ij:")
    print("   ", end="")
    for j in range(n):
        print(f"{j:8d}", end="")
    print()
    
    for i in range(n):
        print(f"{i:2d} ", end="")
        for j in range(n):
            if i == j:
                print(f"{0:8.3f}", end="")
            else:
                print(f"{S_ij[i][j]:8.3f}", end="")
        print()


def load_obj_geometry(obj_path: str) -> dict:
    """
    Загрузка геометрии из OBJ файла и вычисление тепловых параметров
    
    Args:
        obj_path: путь к OBJ файлу
        
    Returns:
        словарь с геометрическими параметрами для решателя:
        {
            'S_surface': list,  # площади поверхностей в порядке элементов
            'S_connections': np.ndarray,  # матрица площадей сечений
            'element_names': list,  # имена элементов для отладки
            'num_elements': int  # количество элементов
        }
    """
    print(f"Загрузка OBJ файла: {obj_path}")
    
    try:
        vertices, elements = parse_obj(obj_path)
        
        if not elements:
            raise ValueError("В OBJ файле не найдено ни одного элемента (объекта)")
        
        print(f"Найдено элементов: {len(elements)}")
        print(f"Найдено вершин: {len(vertices)}")
        
        element_names = sorted(elements.keys())
        N = len(element_names)
        
        name_to_idx = {name: i for i, name in enumerate(element_names)}
        
        S_i_dict = calculate_surface_areas(vertices, elements)
        
        S_surface = [S_i_dict[name] for name in element_names]
        
        S_ij_dict = calculate_Sij(vertices, elements)
        
        S_connections = np.zeros((N, N))
        
        for (name_i, name_j), area in S_ij_dict.items():
            i = name_to_idx[name_i]
            j = name_to_idx[name_j]
            S_connections[i, j] = area
            S_connections[j, i] = area
        
        print("\n=== Результаты загрузки геометрии ===")
        print(f"Количество КЭ: {N}")
        print("\nПлощади поверхностей S_i:")
        for i, name in enumerate(element_names):
            print(f"  {name}: {S_surface[i]:.6f}")
        
        print("\nПлощади общих граней S_ij:")
        for (name_i, name_j), area in S_ij_dict.items():
            print(f"  {name_i} - {name_j}: {area:.6f}")
        
        return {
            'S_surface': S_surface,
            'S_connections': S_connections,
            'element_names': element_names,
            'num_elements': N,
            'vertices_count': len(vertices),
            'faces_count': sum(len(faces) for faces in elements.values())
        }
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {obj_path}")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Ошибка при обработке OBJ файла: {repr(e)}")
    
# Демонстрация работы
if __name__ == "__main__":
    filename = "C:/Users/kkiya/Projects/ctmm/Task1/model2.obj"
    vertices, elements = parse_obj(filename)
    S_ij = calculate_Sij(vertices, elements)
    
    element_names = list(elements.keys())
    # print_Sij_matrix(S_ij, element_names)
    load_obj_geometry(filename)
