# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

#############################################################################################################
# added by controller
import os
from custom_msgs.msg import GlobalWaypointSetpoint, LocalWaypointSetpoint
from custom_msgs.msg import Heartbeat
#############################################################################################################

from rclpy.qos import QoSProfile
import cv2
from cv_bridge import CvBridge
import numpy as np
import onnx
import onnxruntime as ort
import time
import random
import math


class PathPlanning:
    def __init__(self, model_path, image_path, map_size=1000):
        self.model = onnx.load(model_path)
        self.ort_session = ort.InferenceSession(model_path)
        self.map_size = map_size
        self.raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.raw_image_flipped = cv2.flip(self.raw_image, 0)
        self.image_new = np.where(self.raw_image_flipped < 150, 0, 1)  # 130
        # Heightmap 하얀 부분이 더 높은 장애물임
        # 150m로 이동할 때, 장애물이 150보다 작으면 지나갈 수 있으니 0 150보다 크면 1
        # 150m 이상 높은 장애물 모두 통과 가능하도록 경로 산출

    def compute_path(self, Init, Target, step_num):

        # Initialization
        MapSize = step_num - 1000
        Waypoint2 = np.zeros((step_num, 3))
        Waypoint = np.zeros((step_num, 3))
        Obs_mat = np.array([[]])
        Waypoint[0] = Init[:]

        # 변수 초기화
        ds = 0      # ds의 초기값 설정 (이 값은 문맥에 맞게 조정해야 할 수도 있음)
        Image_New =  self.image_new
        Init = np.array(Init)
        Target = np.array(Target)
        Pos = Init  # 초기 위치로 Pos 설정

        # 경로 계산
        waypoint = np.zeros((step_num, 3))
        for i in range(1, step_num):
            ## Initialize
            Obs1 = np.zeros((1, 3))
            Obs2 = np.zeros((1, 3))
            Obs3 = np.zeros((1, 12))

            ## Make Obs3(Lidar Terms)
            MaxLidar = 100  # Lidar의 최대 거리 비율 조정 factor
            scalefactor = 20  # map size 비율 조정 factor


            for j in range(0, 12):  # 0 - 11
                LidarState = Pos + ds
                for k in range(1, 5):
                    LidarState = LidarState + scalefactor * np.array(
                        [np.cos(((30) * (j - 1)) * np.pi / 180), 0,
                         np.sin(((30) * (j - 1)) * np.pi / 180)])  # Radian으로 해야 함

                    # Map Size 넘어갈만큼 진전시켰으면 Obs3 0으로 하고 Data 수집 멈추기
                    if LidarState[0] >= MapSize or LidarState[2] >= MapSize:
                        Diff = LidarState - (Pos + 1 * ds)
                        if np.linalg.norm(Diff) <= MaxLidar:
                            Obs3[0][j] = (np.linalg.norm(Diff)) / scalefactor  # Scaling 필요
                            break
                        else:
                            Obs3[0][j] = MaxLidar / scalefactor
                            break

                    if LidarState[0] < 0 or LidarState[2] < 0:
                        Diff = LidarState - (Pos + 1 * ds)
                        if np.linalg.norm(Diff) <= MaxLidar:
                            Obs3[0][j] = (np.linalg.norm(Diff)) / scalefactor  # s Scaling 필요
                            break
                        else:
                            Obs3[0][j] = MaxLidar / scalefactor
                            break

                    if Image_New[int(LidarState[2])][int(LidarState[0])] > 0:  # 꼭 후처리된 Map에서 # 안의 순서 주의
                        Diff = LidarState - (Pos + 1 * ds)
                        if np.linalg.norm(Diff) <= MaxLidar:
                            Obs3[0][j] = (np.linalg.norm(Diff)) / scalefactor  # Scaling 필요
                            break
                        else:
                            Obs3[0][j] = MaxLidar / scalefactor
                            break

                    Obs3[0][j] = MaxLidar / scalefactor

            # Observation for onnx range [-2500, 2500]
            # Target[0] = Target[0] - 2500
            # Target[2] = Target[2] - 2500
            # Pos[0] = Pos[0] - 2500
            # Pos[2] = Pos[2] - 2500
            Obs1 = (Target - Init) / np.linalg.norm(Target - Init)
            Obs2 = (Target - Pos) / scalefactor

            ## Make Observation
            Obs = np.random.randn(1, 18)
            Obs[0][0] = Obs1[0]
            Obs[0][1] = Obs1[1]
            Obs[0][2] = Obs1[2]
            Obs[0][3] = Obs2[0]
            Obs[0][4] = Obs2[1]
            Obs[0][5] = Obs2[2]

            Obs[0][6] = Obs3[0][0]
            Obs[0][7] = Obs3[0][1]
            Obs[0][8] = Obs3[0][2]
            Obs[0][9] = Obs3[0][3]
            Obs[0][10] = Obs3[0][4]
            Obs[0][11] = Obs3[0][5]
            Obs[0][12] = Obs3[0][6]
            Obs[0][13] = Obs3[0][7]
            Obs[0][14] = Obs3[0][8]
            Obs[0][15] = Obs3[0][9]
            Obs[0][16] = Obs3[0][10]
            Obs[0][17] = Obs3[0][11]


            Act = self.ort_session.run(None, {"obs_0": Obs.astype(np.float32)})

            ## Make Move
            Act = Act[2][0][0]  # np.linalg.norm(Act[2][0] - Act[2][1])

            for j in range(0, 5):
                # LOS 벡터 계산
                LOS_2D_N = (Target - Pos) / np.linalg.norm(Target - Pos)

                # Act 값을 사용하여 액션 각도 계산
                if Act > 0:
                    Action_angle = (90 * np.exp(5 * Act) / (np.exp(5) - 1) - 90 / (np.exp(5) - 1)) * np.pi / 180
                elif Act <= 0:
                    Action_angle = (-90 * np.exp(5 * (-Act)) / (np.exp(5) - 1) + 90 / (np.exp(5) - 1)) * np.pi / 180

                # 액션 벡터 계산
                Action_Vec = np.array([
                    LOS_2D_N[0] * np.cos(Action_angle) + LOS_2D_N[2] * np.sin(Action_angle),
                    0,
                    -LOS_2D_N[0] * np.sin(Action_angle) + LOS_2D_N[2] * np.cos(Action_angle)
                ])

                # 이동 벡터 계산 및 적용
                dS = 5 * Action_Vec

                Waypoint2[i][0] = dS[0]
                Waypoint2[i][1] = dS[1]
                Waypoint2[i][2] = dS[2]



                Pos = Pos + dS

            # Observation for onnx range [-2500, 2500]
            # Target[0] = Target[0] + 2500
            # Target[2] = Target[2] + 2500
            # Pos[0] = Pos[0] + 2500
            # Pos[2] = Pos[2] + 2500

            ## Set End Condtion
            if (np.linalg.norm(Target - Pos) < 25):
                break

            Waypoint[i][0] = Pos[0]
            Waypoint[i][1] = Pos[1]
            Waypoint[i][2] = Pos[2]


            # 경로 저장
            self.path_x = Waypoint[:i + 1, 0]
            self.path_y = Waypoint[:i + 1, 2]

#############################################################################################################
# added by controller
# altitude 150 -> 5
        self.path_z = 5 * np.ones(len(self.path_x))
#############################################################################################################


    def plot_binary(self, output_path, step_num):

        MapSize = step_num - 1000

        # 결과 이미지 생성 및 저장
        path_x = self.path_x
        path_y = self.path_y

        Image_New = self.image_new
        Image_New2 = Image_New * 255
        Image_New2 = np.uint8(np.uint8((255 - Image_New2)))

        # Image_New2 = cv2.flip(Image_New2, 0)
        Image_New2 = cv2.flip(Image_New2, 1)
        Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        # Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        # Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        imageLine = Image_New2.copy()
        # 이미지 크기에 따른 그리드 간격 설정
        grid_interval = 20

        # Image_New2 이미지에 그리드 그리기
        for x in range(0, imageLine.shape[1], grid_interval):  # 이미지의 너비에 따라
            cv2.line(imageLine, (x, 0), (x, imageLine.shape[0]), color=(125, 125, 125), thickness=1)

        for y in range(0, imageLine.shape[0], grid_interval):  # 이미지의 높이에 따라
            cv2.line(imageLine, (0, y), (imageLine.shape[1], y), color=(125, 125, 125), thickness=1)

        # 이미지에 맞게 SAC Waypoint 변경 후 그리기
        for i in range(1, step_num - 1):  # Changed to step_num - 1
            for m in range(0, len(self.path_x) - 2):
                Im_i = int(self.path_x[m + 1])
                Im_j = MapSize - int(self.path_y[m + 1])

                Im_iN = int(self.path_x[m + 2])
                Im_jN = MapSize - int(self.path_y[m + 2])

                # 각 웨이포인트에 점 찍기 (thickness 2)
                cv2.circle(imageLine, (Im_i, Im_j), radius=2, color=(0, 255, 0), thickness=2)

                # 웨이포인트 사이를 선으로 연결 (thickness 1)
                cv2.line(imageLine, (Im_i, Im_j), (Im_iN, Im_jN), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(output_path, imageLine)  ################################
        # cv2.imshow('Binary Path Image', imageLine)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


    def plot_original(self, output_path, step_num):

        MapSize = step_num - 1000

        ## Plot and Save Image
        path_x = self.path_x
        path_y = self.path_y

        imageLine2 = self.raw_image

        # 이미지에 맞게 SAC Waypoint 변경 후 그리기
        for i in range(1, step_num - 1):  # Changed to step_num - 1
            for m in range(0, len(self.path_x) - 2):
                Im_i = int(self.path_x[m + 1])
                Im_j = MapSize - int(self.path_y[m + 1])

                Im_iN = int(self.path_x[m + 2])
                Im_jN = MapSize - int(self.path_y[m + 2])

                # 각 웨이포인트에 점 찍기 (thickness 2)
                cv2.circle(imageLine2, (Im_i, Im_j), radius=2, color=(0, 255, 0), thickness=2)

                # 웨이포인트 사이를 선으로 연결 (thickness 1)
                cv2.line(imageLine2, (Im_i, Im_j), (Im_iN, Im_jN), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(output_path, imageLine2)  ################################
        # cv2.imshow('Binary Path Image', imageLine2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def total_waypoint_distance(self):
        total_distance = 0
        for i in range(1, len(self.path_x)):
            dx = self.path_x[i] - self.path_x[i - 1]
            dy = self.path_y[i] - self.path_y[i - 1]
            total_distance += np.sqrt(dx**2 + dy**2)
        return total_distance

    def init_to_target_distance(self):
        dx = self.path_x[-1] - self.path_x[0]
        dy = self.path_y[-1] - self.path_y[0]
        return np.sqrt(dx**2 + dy**2)

    def print_distance_length(self):
        total_wp_distance = self.total_waypoint_distance()
        init_target_distance = self.init_to_target_distance()

        # # 절대오차 계산
        # absolute_error = abs(total_wp_distance - init_target_distance)
        #
        # # 상대오차 계산 (0으로 나누는 경우 예외 처리)
        # if init_target_distance != 0:
        #     relative_error = absolute_error / init_target_distance
        #     print(f"절대오차: {absolute_error:.2f}, 상대오차: {relative_error:.2%}")
        # else:
        #     print(f"절대오차: {absolute_error:.2f}, 상대오차: 계산 불가 (분모가 0)")

        length =  (total_wp_distance )
        print("SAC Path Length: {:.2f}".format(length))

        return length


class RRT:
    def __init__(self, model_path, image_path, map_size=1000):
        self.model = onnx.load(model_path)
        self.ort_session = ort.InferenceSession(model_path)
        self.map_size = map_size
        self.raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.raw_image_flipped = cv2.flip(self.raw_image, 0)
        self.image_new = np.where(self.raw_image_flipped < 150, 0, 1)  # 130
        # Heightmap 하얀 부분이 더 높은 장애물임
        # 150m로 이동할 때, 장애물이 150보다 작으면 지나갈 수 있으니 0 150보다 크면 1
        # 150m 이상 높은 장애물 모두 통과 가능하도록 경로 산출

    # Definition
    def collision_check(self, Map, from_wp, to_wp):
        N_grid = len(Map) + 5000

        min_x = math.floor(min(np.round(from_wp[0]), np.round(to_wp[0])))
        max_x = math.ceil(max(np.round(from_wp[0]), np.round(to_wp[0])))
        min_y = math.floor(min(np.round(from_wp[1]), np.round(to_wp[1])))
        max_y = math.ceil(max(np.round(from_wp[1]), np.round(to_wp[1])))

        if max_x > N_grid - 1:
            max_x = N_grid - 1
        if max_y > N_grid - 1:
            max_y = N_grid - 1

        check1 = Map[min_y][min_x]
        check2 = Map[min_y][max_x]
        check3 = Map[max_y][min_x]
        check4 = Map[max_y][max_x]

        flag_collision = max(check1, check2, check3, check4)

        return flag_collision

    def RRT_PathPlanning(self, Start, Goal):

        TimeStart = time.time()

        # Initialization
        Image =  self.image_new

        #N_grid = len(Image)
        N_grid = 5000

        # print(Start)
        Init = np.array([Start[0], 2, Start[1]])
        Target = np.array([Goal[0], 2, Goal[1]])

        Start = np.array([[Init[0]], [Init[2]]])
        Goal = np.array([[Target[0]], [Target[2]]])

        Start = Start.astype(float)
        Goal = Goal.astype(float)

        # User Parameter
        step_size = np.linalg.norm(Start - Goal, 2) / 500
        Search_Margin = 0

        ##.. Algorithm Initialize
        q_start = np.array([Start, 0, 0], dtype=object)  # Coord, Cost, Parent
        q_goal = np.array([Goal, 0, 0], dtype=object)

        idx_nodes = 1

        nodes = q_start
        nodes = np.vstack([nodes, q_start])
        # np.vstack([q_start, q_goal])
        ##.. Algorithm Start

        flag_end = 0
        N_Iter = 0
        while (flag_end == 0):
            # Set Searghing Area
            Search_Area_min = Goal - Search_Margin
            Search_Area_max = Goal + Search_Margin
            q_rand = Search_Area_min + (Search_Area_max - Search_Area_min) * np.random.uniform(0, 1, [2, 1])

            # Pick the closest node from existing list to branch out from
            dist_list = []
            for i in range(0, idx_nodes + 1):
                dist = np.linalg.norm(nodes[i][0] - q_rand)
                if (i == 0):
                    dist_list = [dist]
                else:
                    dist_list.append(dist)

            val = min(dist_list)
            idx = dist_list.index(val)

            q_near = nodes[idx]
            # q_new = Tree()
            # q_new = collections.namedtuple('Tree', ['coord', 'cost', 'parent'])
            new_coord = q_near[0] + (q_rand - q_near[0]) / val * step_size

            # Collision Check
            flag_collision = self.collision_check(Image, q_near[0], new_coord)
            # print(q_near[0], new_coord)

            # flag_collision = 0

            # Add to Tree
            if (flag_collision == 0):
                Search_Margin = 0
                new_cost = nodes[idx][1] + np.linalg.norm(new_coord - q_near[0])
                new_parent = idx
                q_new = np.array([new_coord, new_cost, new_parent], dtype=object)
                # print(nodes[0])

                nodes = np.vstack([nodes, q_new])
                # nodes = list(zip(nodes, q_new))
                # nodes.append(q_new)
                # print(nodes[0])

                Goal_Dist = np.linalg.norm(new_coord - q_goal[0])

                idx_nodes = idx_nodes + 1

                if (Goal_Dist < step_size):
                    flag_end = 1
                    nodes = np.vstack([nodes, q_goal])
                    idx_nodes = idx_nodes + 1
            else:
                Search_Margin = Search_Margin + N_grid / 100

                if Search_Margin >= N_grid:
                    Search_Margin = N_grid - 1
            N_Iter = N_Iter + 1
            if N_Iter > 100000:
                break

        flag_merge = 0
        idx = 0
        idx_parent = idx_nodes - 1
        path_x_inv = np.array([])
        path_y_inv = np.array([])
        while (flag_merge == 0):
            path_x_inv = np.append(path_x_inv, nodes[idx_parent][0][0])
            path_y_inv = np.append(path_y_inv, nodes[idx_parent][0][1])

            idx_parent = nodes[idx_parent][2]
            idx = idx + 1

            if idx_parent == 0:
                flag_merge = 1

        path_x = np.array([])
        path_y = np.array([])
        for i in range(0, idx - 2):
            path_x = np.append(path_x, path_x_inv[idx - i - 1])
            path_y = np.append(path_y, path_y_inv[idx - i - 1])


        self.path_x = path_x
        self.path_y = path_y
        self.path_z = 150 * np.ones(len(self.path_x))

        TimeEnd = time.time()


    def plot_RRT(self,output_path):

        MapSize = self.map_size

        ## Plot and Save Image
        path_x = self.path_x
        path_y = self.path_y


        ## Plot and Save Image
        imageLine2 = self.raw_image




        # 이미지에 맞게 SAC Waypoint 변경 후 그리기
        for m in range(0, len(path_x) - 2):
            Im_i = int(path_x[m + 1])
            Im_j = MapSize - int(path_y[m + 1])

            Im_iN = int(path_x[m + 2])
            Im_jN = MapSize - int(path_y[m + 2])

            # 각 웨이포인트에 점 찍기 (thickness 2)
            cv2.circle(imageLine2, (Im_i, Im_j), radius=2, color=(0, 255, 0), thickness=1)

            # 웨이포인트 사이를 선으로 연결 (thickness 1)
            cv2.line(imageLine2, (Im_i, Im_j), (Im_iN, Im_jN), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(output_path, imageLine2)  ################################


    def plot_RRT_binary(self,output_path):
        MapSize = self.map_size


        ## Plot and Save Image
        path_x = self.path_x
        path_y = self.path_y

        Image_New = self.image_new
        Image_New2 = Image_New * 255
        Image_New2 = np.uint8(np.uint8((255 - Image_New2)))

        # Image_New2 = cv2.flip(Image_New2, 0)
        Image_New2 = cv2.flip(Image_New2, 1)
        Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        # Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        # Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        imageLine = Image_New2.copy()
        # 이미지 크기에 따른 그리드 간격 설정
        grid_interval = 20

        # Image_New2 이미지에 그리드 그리기
        for x in range(0, imageLine.shape[1], grid_interval):  # 이미지의 너비에 따라
            cv2.line(imageLine, (x, 0), (x, imageLine.shape[0]), color=(125, 125, 125), thickness=2)

        for y in range(0, imageLine.shape[0], grid_interval):  # 이미지의 높이에 따라
            cv2.line(imageLine, (0, y), (imageLine.shape[1], y), color=(125, 125, 125), thickness=1)


        # 이미지에 맞게 SAC Waypoint 변경 후 그리기
        for i in range(1, len(path_x) - 2):  # Changed to step_num - 1
            for m in range(0, len(path_x) - 2):
                Im_i = int(path_x[m + 1])
                Im_j = MapSize - int(path_y[m + 1])

                Im_iN = int(path_x[m + 2])
                Im_jN = MapSize - int(path_y[m + 2])

                # 각 웨이포인트에 점 찍기 (thickness 2)
                cv2.circle(imageLine, (Im_i, Im_j), radius=2, color=(0, 255, 0), thickness=1)

                # 웨이포인트 사이를 선으로 연결 (thickness 1)
                cv2.line(imageLine, (Im_i, Im_j), (Im_iN, Im_jN), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(output_path, imageLine)  ################################



    def calculate_and_print_path_info(self):
        LenRRT = 0
        for cal in range(len(self.path_x) - 1):
            First = np.array([self.path_x[cal], self.path_y[cal]])
            Second = np.array([self.path_x[cal + 1], self.path_y[cal + 1]])

            U = (Second - First) / np.linalg.norm(Second - First)

            State = First
            for cal_step in range(500):
                State = State + U

                if np.linalg.norm(Second - State) < 20:
                    break

                # Add collision check code here if needed

            Len_temp = np.linalg.norm(Second - First)
            LenRRT += Len_temp

        print("RRT 경로 길이:", LenRRT)
        return LenRRT

    def total_waypoint_distance(self):
        total_distance = 0
        for i in range(1, len(self.path_x)):
            dx = self.path_x[i] - self.path_x[i - 1]
            dy = self.path_y[i] - self.path_y[i - 1]
            total_distance += np.sqrt(dx**2 + dy**2)
        return total_distance

    def init_to_target_distance(self):
        dx = self.path_x[-1] - self.path_x[0]
        dy = self.path_y[-1] - self.path_y[0]
        return np.sqrt(dx**2 + dy**2)

    def print_distance_length(self):
        total_wp_distance = self.total_waypoint_distance()
        init_target_distance = self.init_to_target_distance()

        # # 절대오차 계산
        # absolute_error = abs(total_wp_distance - init_target_distance)
        #
        # # 상대오차 계산 (0으로 나누는 경우 예외 처리)
        # if init_target_distance != 0:
        #     relative_error = absolute_error / init_target_distance
        #     print(f"절대오차: {absolute_error:.2f}, 상대오차: {relative_error:.2%}")
        # else:
        #     print(f"절대오차: {absolute_error:.2f}, 상대오차: 계산 불가 (분모가 0)")

        length = total_wp_distance 
        print("RRT: Path Length: {:.2f}".format(length))

        return length




class PathPlanningServer(Node):  # topic 이름과 message 타입은 서로 매칭되어야 함

    def __init__(self):
        super().__init__('minimal_subscriber')
        
        self.bridge = CvBridge()

        # mode change
        self.mode = os.environ['MODE']
        
        # initialize global waypoint
        self.Init_custom = [0.0,0.0,0.0]
        self.Target_custom = [0.0,0.0,0.0]

        # Initialiaztion
        ## Range [-2500, 2500]으로 바꾸기
        self.MapSize = 1000  # size 500
        self.Step_Num_custom = self.MapSize + 1000

#############################################################################################################
# added by controller
        # file path
        self.image_path = '/home/user/ros_ws/src/pathplanning/pathplanning/map/1000-003.png'
        self.model_path = "/home/user/ros_ws/src/pathplanning/pathplanning/model/90_exp_263k.onnx"
        self.model_path2 = "/home/user/ros_ws/src/pathplanning/pathplanning/model/test26.onnx"

        # path plannig complete flag
        self.path_plannig_start         = False         # flag whether path planning start 
        self.path_planning_complete     = False         # flag whether path planning is complete 

        # heartbeat signal of another module node
        self.controller_heartbeat           = False     # flag controller heartbeat
        self.path_following_heartbeat       = False     # flag path following heartbeat
        self.collision_avoidance_heartbeat  = False     # flag collision avoidance heartbeat

        # declare global waypoint subscriber from controller
        self.global_waypoint_subscriber                 =   self.create_subscription(GlobalWaypointSetpoint, '/global_waypoint_setpoint',  self.global_waypoint_callback ,10)
        
        # declare heartbeat_subscriber 
        self.controller_heartbeat_subscriber            =   self.create_subscription(Heartbeat, '/controller_heartbeat',            self.controller_heartbeat_call_back,            10)
        self.path_following_heartbeat_subscriber        =   self.create_subscription(Heartbeat, '/path_following_heartbeat',        self.path_following_heartbeat_call_back,        10)
        self.collision_avoidance_heartbeat_subscriber   =   self.create_subscription(Heartbeat, '/collision_avoidance_heartbeat',   self.collision_avoidance_heartbeat_call_back,   10)
        
        # declare local waypoint publisher to controller
        self.local_waypoint_publisher      =   self.create_publisher(LocalWaypointSetpoint, '/local_waypoint_setpoint_from_PP', 10)

        # declare heartbeat_publisher 
        self.heartbeat_publisher           =   self.create_publisher(Heartbeat, '/path_planning_heartbeat', 10)

        print("                                          ")
        print("===== Path Planning Node is Running  =====")
        print("                                          ")

        # declare heartbeat_timer
        period_heartbeat_mode =   1        
        self.heartbeat_timer  =   self.create_timer(period_heartbeat_mode, self.publish_heartbeat)
#############################################################################################################


#############################################################################################################
# added by controller

    # publish local waypoint and path planning complete flag
    def local_waypoint_publish(self):
        msg = LocalWaypointSetpoint()
        msg.path_planning_complete = self.path_planning_complete
        msg.waypoint_x             = self.waypoint_x
        msg.waypoint_y             = self.waypoint_y
        msg.waypoint_z             = self.waypoint_z
        self.local_waypoint_publisher.publish(msg)
        print("                                          ")
        print("==  Sended local waypoint to controller ==")
        print("                                          ")

# heartbeat check function
    # heartbeat publish
    def publish_heartbeat(self):
        msg = Heartbeat()
        msg.heartbeat = True
        self.heartbeat_publisher.publish(msg)

    # heartbeat subscribe from controller
    def controller_heartbeat_call_back(self,msg):
        self.controller_heartbeat = msg.heartbeat

    # heartbeat subscribe from path following
    def path_following_heartbeat_call_back(self,msg):
        self.path_following_heartbeat = msg.heartbeat

    # heartbeat subscribe from collision avoidance
    def collision_avoidance_heartbeat_call_back(self,msg):
        self.collision_avoidance_heartbeat = msg.heartbeat
#############################################################################################################
    
    # added by controller
    # update global waypoint and path plannig start flag if subscribe global waypoint from controller
    def global_waypoint_callback(self, msg):
        # check heartbeat
        if self.controller_heartbeat == True and self.path_following_heartbeat == True and self.collision_avoidance_heartbeat == True:
            if self.path_plannig_start == False and self.path_planning_complete == False:

                self.Init_custom        =   msg.start_point
                self.Target_custom      =   msg.goal_point
                self.path_plannig_start = True

                print("                                          ")
                print("===== Recieved Path Planning Request =====")
                print("                                          ")

            else:
                pass

            # Mode: 1 (현재 경로계획만 계산), 2 (현재 + 작년 경로계획과 같이 계산하여 정량평가까지 완료)
            # Node Input: Image 경로 & Mode & 시작점 & 도착점, Output: wp (or Cost도)

            if self.mode == 1 and self.path_planning_complete == False :

                # model 90 deg
                # start path planning
                planner = PathPlanning(self.model_path, self.image_path)
                planner.compute_path(self.Init_custom, self.Target_custom,
                                 self.Step_Num_custom)  # start point , target point,step_num, max lidar, scale factor

                planner.plot_binary("/home/user/ros_ws/src/pathplanning/pathplanning/Results_Images/SAC_Result_biary.png", self.Step_Num_custom)
                planner.plot_original("/home/user/ros_ws/src/pathplanning/pathplanning/Results_Images/SAC_Result_og.png", self.Step_Num_custom)
                print("                                          ")
                print("=====   Path Planning Complete!!     =====")
                print("                                          ")

                planner.print_distance_length()
                print("                                           ")

                # setting msg
                self.path_planning_complete     =       True
                self.waypoint_x                 =       planner.path_x.tolist()
                self.waypoint_y                 =       planner.path_y.tolist()
                self.waypoint_z                 =       planner.path_z.tolist()

                # publish local waypoint and path planning complete flag
                self.local_waypoint_publish()

            elif self.mode == 2:

                # model 90 deg
                planner = PathPlanning(self.model_path, self.image_path)
                planner.compute_path(self.Init_custom, self.Target_custom,
                                     self.Step_Num_custom)  # start point , target point,step_num, max lidar, scale factor
                planner.plot_binary("/home/user/ros_ws/src/pathplanning/pathplanning/Results_Images/SAC_Result_biary_01.png", self.Step_Num_custom)
                planner.plot_original("/home/user/ros_ws/src/pathplanning/pathplanning/Results_Images/SAC_Result_og_01.png", self.Step_Num_custom)


                # model test26
                planner2 = PathPlanning(self.model_path2, self.image_path)
                planner2.compute_path(self.Init_custom, self.Target_custom, self.Step_Num_custom)  # start point , target point,
                planner2.plot_binary("/home/user/ros_ws/src/pathplanning/pathplanning/Results_Images/SAC_Result_biary_02.png", self.Step_Num_custom)
                planner2.plot_original("/home/user/ros_ws/src/pathplanning/pathplanning/Results_Images/SAC_Result_og_02.png", self.Step_Num_custom)

                # Cost Calculation
                ratio = planner.print_distance_length()
                ratio2 = planner2.print_distance_length()

                cost = ((ratio - ratio2)/ratio2) * 100
                print("                                           ")
                print("---------------Results----------------")
                print("Cost: {:.2f}%".format(cost))
                print("                                           ")



            elif self.mode == 3:

                # model 90 deg
                planner = PathPlanning(self.model_path, self.image_path)
                planner.compute_path(self.Init_custom, self.Target_custom,
                                     self.Step_Num_custom)  # start point , target point,step_num, max lidar, scale factor
                planner.plot_binary("/home/user/ros_ws/src/pathplanning/pathplanning/Results_Images/SAC_Result_binary.png", self.Step_Num_custom)
                planner.plot_original("/home/user/ros_ws/src/pathplanning/pathplanning/Results_Images/SAC_Result_og.png", self.Step_Num_custom)

                # RRT
                start_coord = (self.Init_custom[0],self.Init_custom[2])  # Replace with your desired start coordinates
                goal_coord = (self.Target_custom[0],self.Target_custom[2])

                N = 10
                planner3 = RRT(self.model_path,self.image_path)
                total_path_ratio = []
                for i in range(N):
                    print(f"RRT Running iteration {i+1}/{N}")

                    # Call the RRT path planning method
                    planner3.RRT_PathPlanning(start_coord, goal_coord)

                    # Plot the results
                    planner3.plot_RRT(f"/home/user/ros_ws/src/pathplanning/pathplanning/Results_Images/Results_Images/RRT_Result_og_{i+1}.png")
                    planner3.plot_RRT_binary(f"/home/user/ros_ws/src/pathplanning/pathplanning/Results_Images/Results_Images/RRT_Result_Binary_{i+1}.png")

                    # Calculate and print the distance ratio
                    distance_ratio = planner3.print_distance_length()
                    total_path_ratio.append(distance_ratio)

                # Display the results from all iterations
                min_path_ratio = min(total_path_ratio)
                print("                                           ")
                print("---------------Results----------------")
                print("RRT Min Path Length:: {:.2f}".format(min_path_ratio))

                # Cost Calculation
                ratio = planner.print_distance_length()
                ratio2 = min_path_ratio

                cost = ((ratio - ratio2)/ratio2) * 100

                print("Cost: {:.2f}%".format(cost))
                print("                                           ")
        else:
            pass


def main(args=None):
    rclpy.init(args=args)
    SAC_module = PathPlanningServer()
    try:
        rclpy.spin(SAC_module)
    except KeyboardInterrupt:
        SAC_module.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        SAC_module.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

