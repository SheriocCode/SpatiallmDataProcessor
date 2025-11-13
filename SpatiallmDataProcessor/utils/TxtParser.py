from shapely.geometry import Polygon
from utils.entity import Wall, Door, Window, Bbox

def parse_scene_file(filename):
    """解析txt文件，提取墙体和边界框数据"""
    walls = []
    bboxes = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('wall_'):
                parts = line.split('=')
                wall_id = int(parts[0].split('_')[1])
                wall_data = parts[1].replace('Wall(', '').rstrip(')').split(',')
                wall_data = [float(x.strip()) for x in wall_data]
                
                wall = Wall(
                    id=wall_id,
                    ax=wall_data[0],
                    ay=wall_data[1],
                    az=wall_data[2],
                    bx=wall_data[3],
                    by=wall_data[4],
                    bz=wall_data[5],
                    height=wall_data[6],
                    thickness=wall_data[7]
                )
                walls.append(wall)
            
            elif line.startswith('bbox_'):
                parts = line.split('=')
                bbox_id = int(parts[0].split('_')[1])
                bbox_data = parts[1].replace('Bbox(', '').rstrip(')').split(',')
                
                class_name = bbox_data[0].strip()
                numeric_data = [float(x.strip()) for x in bbox_data[1:]]
                
                bbox = Bbox(
                    id=bbox_id,
                    class_name=class_name,
                    position_x=numeric_data[0],
                    position_y=numeric_data[1],
                    position_z=numeric_data[2],
                    angle_z=numeric_data[3],
                    scale_x=numeric_data[4],
                    scale_y=numeric_data[5],
                    scale_z=numeric_data[6]
                )
                bboxes.append(bbox)
    
    walls.sort(key=lambda x: x.id)
    bboxes.sort(key=lambda x: x.id)
    return walls, bboxes

def parse_scene_file_wall_door_window(filename):
    """解析txt文件，提取墙体、门、窗数据"""
    walls = []
    doors = []
    windows = []
    wall_id_mapping = {}  # 用于存储 wall_id 的映射，例如 {"wall_0": 0}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('wall_'):
                # 解析墙体
                parts = line.split('=')
                wall_part = parts[0]  # 如 'wall_0'
                wall_id = int(wall_part.split('_')[1])  # 提取数字，如 0
                wall_data_str = parts[1].replace('Wall(', '').rstrip(')')
                wall_data = [float(x.strip()) for x in wall_data_str.split(',')]

                wall = Wall(
                    id=wall_id,
                    ax=wall_data[0],
                    ay=wall_data[1],
                    az=wall_data[2],
                    bx=wall_data[3],
                    by=wall_data[4],
                    bz=wall_data[5],
                    height=wall_data[6],
                    thickness=wall_data[7]
                )
                walls.append(wall)
                wall_id_mapping[wall_part] = wall_id  # 记录 wall_0 -> 0

            elif line.startswith('door_'):
                # 解析门
                parts = line.split('=')
                door_part = parts[0]  # 如 'door_0'
                door_id = int(door_part.split('_')[1])  # 提取数字，如 0
                door_data_str = parts[1].replace('Door(', '').rstrip(')')
                door_data = [x.strip() for x in door_data_str.split(',')]

                # 第一个参数是 wall 的名字，如 'wall_0'，我们要找到对应的 wall_id 数字
                wall_ref = door_data[0]  # 如 'wall_0'
                wall_id = wall_id_mapping.get(wall_ref)
                if wall_id is None:
                    raise ValueError(f"Door references undefined wall: {wall_ref}")

                position_x = float(door_data[1])
                position_y = float(door_data[2])
                position_z = float(door_data[3])
                width = float(door_data[4])
                height = float(door_data[5])

                door = Door(
                    id=door_id,
                    wall_id=wall_id,
                    position_x=position_x,
                    position_y=position_y,
                    position_z=position_z,
                    width=width,
                    height=height
                )
                doors.append(door)

            elif line.startswith('window_'):
                # 解析窗
                parts = line.split('=')
                window_part = parts[0]  # 如 'window_0'
                window_id = int(window_part.split('_')[1])  # 提取数字，如 0
                window_data_str = parts[1].replace('Window(', '').rstrip(')')
                window_data = [x.strip() for x in window_data_str.split(',')]

                # 第一个参数是 wall 的名字，如 'wall_2'
                wall_ref = window_data[0]  # 如 'wall_2'
                wall_id = wall_id_mapping.get(wall_ref)
                if wall_id is None:
                    raise ValueError(f"Window references undefined wall: {wall_ref}")

                position_x = float(window_data[1])
                position_y = float(window_data[2])
                position_z = float(window_data[3])
                width = float(window_data[4])
                height = float(window_data[5]) if len(window_data) > 5 else 1.4  # 默认值，或根据需求调整

                # Window 继承自 Door，构造函数参数一致，只是 entity_label 默认为 window
                window = Window(
                    id=window_id,
                    wall_id=wall_id,
                    position_x=position_x,
                    position_y=position_y,
                    position_z=position_z,
                    width=width,
                    height=height
                )
                windows.append(window)
    
    walls.sort(key=lambda x: x.id)
    doors.sort(key=lambda x: x.id)
    windows.sort(key=lambda x: x.id)
    
    return walls, doors, windows

def create_wall_polygon(walls):
    """从墙体数据创建多边形"""
    if not walls:
        return None
        
    polygon_points = [(wall.ax, wall.ay) for wall in walls]
    
    # 确保多边形闭合
    if polygon_points and polygon_points[0] != polygon_points[-1]:
        polygon_points.append(polygon_points[0])
        
    return Polygon(polygon_points)

