'''
processed.json to TimeGeo train.data
'''

import pickle
import pdb
import json
from config import *
import copy
import numpy as np
import datetime

max_grid = args.max_grid
GRID_SIZE = args.GRID_SIZE
lon_l, lon_r, lat_b, lat_u = 115.43, 117.52, 39.44, 41.05 # Beijing
earth_radius = 6378137.0
pi = 3.1415926535897932384626
meter_per_degree = earth_radius * pi / 180.0
lat_step = GRID_SIZE * (1.0 / meter_per_degree)
ratio = np.cos((lat_b + lat_u) * np.pi / 360)
lon_step = lat_step / ratio

def map_id_to_tokens_py(grid_id):
    x = grid_id // max_grid
    y = grid_id - x*max_grid - 1
    lon = (x+0.5)*lon_step + lon_l
    lat = (y+0.5)*lat_step + lat_b
    return lon, lat

def get_id_dict(data_dir):
    with open(data_dir,'r',encoding='utf-8') as f:
        dt_info = json.load(f)
    dt_id = {}
    for item in dt_info:
        dt_id[item["user_id"]] = item
    return dt_id

def getday(y=2017,m=8,d=15,n=0):
    the_date = datetime.datetime(y,m,d)
    result_date = the_date + datetime.timedelta(days=n)
    d = result_date.strftime('%Y%m%d')
    return d

def merge_day(traj,all_days,start):
    new_traj = [[[] for i in range(24)] for j in range(all_days)]#(天数,48)
    last = None
    for day in traj:
        for point in day:
            if len(point) != 0:
                last = point[0]
                break
        break
    for i in range(all_days):
        for j in range(24):
            if len(traj[i][j]) != 0:
                last = traj[i][j][-1]    
            new_traj[i][j] = copy.deepcopy(last)
            day_info = getday(2017,int(start[0:2]),int(start[2:]),i) \
                        + str('%02d' % j) + "0000"
            new_traj[i][j].append(day_info)
    return new_traj

data_dir = "/data2/songyiwen/human_traj_/dataset/processed_info_example.json"
out_dir = "/data2/songyiwen/human_traj_/dataset/timegeo_train_example.data"

dt_info = get_id_dict(data_dir)
user_traj = []
traj = []

for user_id, info in dt_info.items():#每个用户
    start = info["user_traces"][0][0][0:4]
    end = info["user_traces"][-1][0][0:4]

    all_days = (int(end[1])-int(start[1]))*31 + int(end[2:4])-int(start[2:4]) + 1 #天数
    overall_traj = [[[] for i in range(24)] for j in range(all_days)]#(天数,24)

    for i in range(len(info["user_traces"])):#每个轨迹点
        time = info["user_traces"][i][0]
        # week = datetime.strptime("2017" + time[0:4],"%Y%m%d").weekday()
        day_id = (int(time[0:4][1])-int(start[1]))*31 + int(time[0:4][2:4])-int(start[2:4])
        # hour_id = 2*int(time[4:6]) if int(time[6:])<30 else 2*int(time[4:6])+1
        hour_id = int(time[4:6])

        overall_traj[day_id][hour_id].append([info["user_traj"][i][0],
                                                info["user_traj"][i][1]])
    
    traj.append({"user_id":user_id,
                 "traj":merge_day(overall_traj, all_days, start),
                 "home":map_id_to_tokens_py(info["home"][0]*args.max_grid+info["home"][1]+1),
                 "work":map_id_to_tokens_py(info["work"][0]*args.max_grid+info["work"][1]+1)})

traj_split = []
for info in traj:

    temp = info["traj"]
    days = len(temp)//14 # 分割成两周的轨迹，去尾
    for i in range(days):
        traj_split.append({"user_id":info["user_id"],
                           "traj":temp[(i*14):((i*14)+7)], # 每14天取前7天
                           "home":info["home"],
                           "work":info["work"]})

n = len(traj_split)# user数
train_data = traj_split[0:round(0.7 * n)]

Note=open(out_dir,mode='w')
for info in train_data:
    line = info["user_id"] + ' '
    for idx_j,j in enumerate(info["traj"]):
        for point in j:
            grid_id = point[0] * args.max_grid + point[1] + 1
            lon, lat = map_id_to_tokens_py(grid_id)
            line = line + str(lon) + ',' + str(lat) + ',' + point[2] + ';'
    line = line[:-1]
    line += '?' + str(info["home"][0]) + ',' + str(info["home"][1]) + ',' +\
                 str(info["work"][0]) + ',' + str(info["work"][1])
    Note.write(line)
    Note.write('\n')
Note.close()
print("well done!!!")

