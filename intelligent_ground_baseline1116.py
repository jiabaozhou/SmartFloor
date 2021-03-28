import os
import numpy as np
import pandas as pd
import re
import cv2
import copy
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pyqtgraph as pg
import pyqtgraph.exporters as pge
from multiprocessing import Pool
import sys


def upscale_opencv(img_data, pixelsize, id_track, frame_name, b_thickness, nested_canvas=False):
    height = len(img_data)
    width = len(img_data[0])
    # What type of canvas are we using? We use 2 simple for loops by default.
    if nested_canvas:
        result_img = np.zeros((height * pixelsize + b_thickness, width * pixelsize + b_thickness, 3), np.uint8)
        for i in range(0, width):
            for j in range(0, height):
                pt1 = (pixelsize * i + b_thickness, pixelsize * j + b_thickness)
                pt2 = (pixelsize * (i + 1) - b_thickness, pixelsize * (j + 1) - b_thickness)
                cv2.rectangle(result_img, pt1, pt2, (255, 255, 255), -1)
    elif not nested_canvas:
        result_img = np.zeros((height * pixelsize, width * pixelsize, 3), np.uint8)
        for i in range(1, width):
            pt1 = (pixelsize * i - b_thickness, 0)
            pt2 = (pixelsize * i + b_thickness, height * pixelsize)
            cv2.rectangle(result_img, pt1, pt2, (255, 255, 255), -1)
        for j in range(1, height):
            pt1 = (0, pixelsize * j - b_thickness)
            pt2 = (width * pixelsize, pixelsize * j + b_thickness)
            cv2.rectangle(result_img, pt1, pt2, (255, 255, 255), -1)
    # 身份id
    for id_index in id_track.keys():
        # 帧名字
        if frame_name in id_track[id_index].keys():
            if id_index % 9 == 0:
                for point_index in range(len(id_track[id_index][frame_name])):
                    y, x = id_track[id_index][frame_name][point_index][0], id_track[id_index][frame_name][point_index][
                        1]
                    pt1 = (pixelsize * x + b_thickness, pixelsize * y + b_thickness)
                    pt2 = (pixelsize * (x + 1) - b_thickness, pixelsize * (y + 1) - b_thickness)
                    cv2.rectangle(result_img, pt1, pt2, (id_index, 255 - id_index, 125 - id_index), -1)
            elif id_index % 8 == 0:
                for point_index in range(len(id_track[id_index][frame_name])):
                    y, x = id_track[id_index][frame_name][point_index][0], id_track[id_index][frame_name][point_index][
                        1]
                    pt1 = (pixelsize * x + b_thickness, pixelsize * y + b_thickness)
                    pt2 = (pixelsize * (x + 1) - b_thickness, pixelsize * (y + 1) - b_thickness)
                    cv2.rectangle(result_img, pt1, pt2, (125 - id_index, id_index, 255 - id_index), -1)
            elif id_index % 7 == 0:
                for point_index in range(len(id_track[id_index][frame_name])):
                    y, x = id_track[id_index][frame_name][point_index][0], id_track[id_index][frame_name][point_index][
                        1]
                    pt1 = (pixelsize * x + b_thickness, pixelsize * y + b_thickness)
                    pt2 = (pixelsize * (x + 1) - b_thickness, pixelsize * (y + 1) - b_thickness)
                    cv2.rectangle(result_img, pt1, pt2, (255 - id_index, 125 - id_index, id_index), -1)
            elif id_index % 6 == 0:
                for point_index in range(len(id_track[id_index][frame_name])):
                    y, x = id_track[id_index][frame_name][point_index][0], id_track[id_index][frame_name][point_index][
                        1]
                    pt1 = (pixelsize * x + b_thickness, pixelsize * y + b_thickness)
                    pt2 = (pixelsize * (x + 1) - b_thickness, pixelsize * (y + 1) - b_thickness)
                    cv2.rectangle(result_img, pt1, pt2, (255 - id_index, id_index, 125 + id_index), -1)
            elif id_index % 5 == 0:
                for point_index in range(len(id_track[id_index][frame_name])):
                    y, x = id_track[id_index][frame_name][point_index][0], id_track[id_index][frame_name][point_index][
                        1]
                    pt1 = (pixelsize * x + b_thickness, pixelsize * y + b_thickness)
                    pt2 = (pixelsize * (x + 1) - b_thickness, pixelsize * (y + 1) - b_thickness)
                    cv2.rectangle(result_img, pt1, pt2, (125 + id_index, 255 - id_index, id_index), -1)
            elif id_index % 4 == 0:
                for point_index in range(len(id_track[id_index][frame_name])):
                    y, x = id_track[id_index][frame_name][point_index][0], id_track[id_index][frame_name][point_index][
                        1]
                    pt1 = (pixelsize * x + b_thickness, pixelsize * y + b_thickness)
                    pt2 = (pixelsize * (x + 1) - b_thickness, pixelsize * (y + 1) - b_thickness)
                    cv2.rectangle(result_img, pt1, pt2, (id_index, 125 + id_index, 255 - id_index), -1)
            elif id_index % 3 == 0:
                for point_index in range(len(id_track[id_index][frame_name])):
                    y, x = id_track[id_index][frame_name][point_index][0], id_track[id_index][frame_name][point_index][
                        1]
                    pt1 = (pixelsize * x + b_thickness, pixelsize * y + b_thickness)
                    pt2 = (pixelsize * (x + 1) - b_thickness, pixelsize * (y + 1) - b_thickness)
                    cv2.rectangle(result_img, pt1, pt2, (120 + id_index, id_index, 135 - id_index), -1)
            elif id_index % 2 == 0:
                for point_index in range(len(id_track[id_index][frame_name])):
                    y, x = id_track[id_index][frame_name][point_index][0], id_track[id_index][frame_name][point_index][
                        1]
                    pt1 = (pixelsize * x + b_thickness, pixelsize * y + b_thickness)
                    pt2 = (pixelsize * (x + 1) - b_thickness, pixelsize * (y + 1) - b_thickness)
                    cv2.rectangle(result_img, pt1, pt2, (135 - id_index, 120 + id_index, id_index), -1)
            else:
                for point_index in range(len(id_track[id_index][frame_name])):
                    y, x = id_track[id_index][frame_name][point_index][0], id_track[id_index][frame_name][point_index][
                        1]
                    pt1 = (pixelsize * x + b_thickness, pixelsize * y + b_thickness)
                    pt2 = (pixelsize * (x + 1) - b_thickness, pixelsize * (y + 1) - b_thickness)
                    cv2.rectangle(result_img, pt1, pt2, (id_index, 135 - id_index, 120 + id_index), -1)
    return result_img


# 2020.11.03.09:34获取站立姿态位置和结束位置
def compute_pose_foot(img_data, thresh, new_down_foot_list, result_pose, end_down_foot_list, mirror_baseline):
    # temp_img_data = np.pad(img_data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    # temp_mirror_baseline = np.pad(mirror_baseline, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    # for down_index in range(len(new_down_foot_list)):
    #     [i, j] = new_down_foot_list[down_index]
    #     temp_list = np.array([[temp_mirror_baseline[i][j] - temp_img_data[i][j], temp_mirror_baseline[i + 1][j] - temp_img_data[i + 1][j], temp_mirror_baseline[i + 2][j] - temp_img_data[i + 2][j]],
    #                  [temp_mirror_baseline[i][j+1] - temp_img_data[i][j+1], temp_mirror_baseline[i+1][j+1] - temp_img_data[i+1][j+1], temp_mirror_baseline[i+2][j+1] - temp_img_data[i+2][j+1]],
    #                  [temp_mirror_baseline[i][j+2] - temp_img_data[i][j+2], temp_mirror_baseline[i+1][j+2] - temp_img_data[i+1][j+2], temp_mirror_baseline[i + 2][j + 2] - temp_img_data[i + 2][j + 2]]])
    #     list_max = np.amax(np.array(temp_list))
    #     index = np.where(temp_list == list_max)
    #     if list_max > thresh:
    #         result_pose.append([i-1+index[0][0], j-1+index[1][0]])
    #         new_down_foot_list[down_index] = [i-1+index[0][0], j-1+index[1][0]]
    #     else:
    #         end_down_foot_list.append([i, j])
    for [i, j] in new_down_foot_list:
        try:
            if mirror_baseline[i][j] - img_data[i][j] > thresh and [i - 1, j - 1] not in result_pose and [i - 1,
                                                                                                          j] not in result_pose \
                    and [i - 1, j + 1] not in result_pose and [i, j - 1] not in result_pose \
                    and [i, j + 1] not in result_pose and [i + 1, j - 1] not in result_pose \
                    and [i + 1, j] not in result_pose and [i, j + 1] not in result_pose:
                result_pose.append([i, j])
            else:
                end_down_foot_list.append([i, j])
        except Exception as e:
            continue
    return result_pose, end_down_foot_list


# 2020.09.09.17.28计算落脚点
def compute_down_foot(diff_frame, img_data, mirror_baseline, down_thresh, new_down_foot_list, result_pose):
    down_foot_list = []
    for i in range(len(diff_frame)):
        for j in range(len(diff_frame[0])):
            # 判断落脚 当前帧数据点-上一帧数据点变化是否符合（50， 8000）且翻转平移阈值-当前帧数据点大于落脚点阈值
            if 50 < diff_frame[i][j] < 8000 and mirror_baseline[i][j] - img_data[i][j] > down_thresh:
                if [i, j] not in result_pose and [i, j] not in new_down_foot_list:
                    down_foot_list.append([i, j])
    return down_foot_list


def compute_up_foot(diff_frame, new_down_foot_list, result_pose):
    up_foot_list = []
    for [i, j] in new_down_foot_list:
        # 判断抬脚
        if diff_frame[i][j] < 0 and [i, j] not in result_pose:
            up_foot_list.append([i, j])
    return up_foot_list


def compute_distance(point_A, point_B):
    return np.linalg.norm(np.array(point_A[:2]) - np.array(point_B[:2]))


def compute_midpoint(point_A, point_B):
    return (np.array(point_A[:2]) + np.array(point_B[:2])) / 2


def compute_direction(midpoint_list):
    direction = 0  # 'None'
    theda_x = midpoint_list[1]
    theda_y = midpoint_list[0]
    if -1 < theda_x < 1 and theda_y < 0:
        direction = 1  # 'up'
    elif -1 < theda_x < 1 and theda_y > 0:
        direction = 2  # 'down'
    elif theda_x < 0 and -1 < theda_y < 1:
        direction = 3  # 'left'
    elif theda_x > 0 and -1 < theda_y < 1:
        direction = 4  # 'right'
    elif theda_x >= 1 and theda_y <= -1:
        direction = 5  # 'right_up'
    elif theda_x >= 1 and theda_y >= 1:
        direction = 6  # 'right_down'
    elif theda_x <= -1 and theda_y >= 1:
        direction = 7  # 'left_down'
    elif theda_x <= -1 and theda_y <= -1:
        direction = 8  # 'left_up'
    return direction


def compute_point_whether_exists(temp_result_pose, id_track, count_stand_time, interval_thresh, frame_name,
                                 temp_number_of_people, theda_diff_frame, direction):
    for number_of_people_index in list(id_track.keys()):
        # 统计一个人当前帧存在的落脚点数量（不超过2）
        count_foot = 0
        # 获取历史轨迹最新的帧名
        frame_index = list(id_track[number_of_people_index].keys())[-1]
        # count_stand_time_index = list(count_stand_time[number_of_people_index].keys())[-1]
        count_stand_time_index = list(count_stand_time[number_of_people_index].keys())
        # 历史帧与当前帧计算差值判断是否连续
        if int(frame_name) - int(frame_index) < interval_thresh:
            result_pose_index = 0
            # 遍历获取当前帧的落脚点,注意循环时的深浅拷贝
            while result_pose_index < len(temp_result_pose):
                # 当前帧落脚点存在于已有人的轨迹中,id_track = {id:{frame_name:[x,y,theta]}},theta为当前帧数据与动态阈值差值
                if temp_result_pose[result_pose_index] in np.array(id_track[number_of_people_index][frame_index])[:,
                                                          :2].tolist():
                    count_foot += 1
                    # 统计站立点数据多于1个
                    if len(count_stand_time[number_of_people_index]) > 1:
                        if temp_result_pose[result_pose_index] not in np.array(
                                count_stand_time[number_of_people_index][count_stand_time_index[-1]])[:, :2].tolist() \
                                and temp_result_pose[result_pose_index] not in np.array(
                            count_stand_time[number_of_people_index][count_stand_time_index[-2]])[:, :2].tolist():
                            count_stand_time[number_of_people_index].update({frame_name: [
                                [temp_result_pose[result_pose_index][0], temp_result_pose[result_pose_index][1],
                                 theda_diff_frame[temp_result_pose[result_pose_index][0]][
                                     temp_result_pose[result_pose_index][1]], direction]]})
                    else:
                        if temp_result_pose[result_pose_index] not in np.array(
                                count_stand_time[number_of_people_index][count_stand_time_index[-1]])[:, :2].tolist():
                            count_stand_time[number_of_people_index].update({frame_name: [
                                [temp_result_pose[result_pose_index][0], temp_result_pose[result_pose_index][1],
                                 theda_diff_frame[temp_result_pose[result_pose_index][0]][
                                     temp_result_pose[result_pose_index][1]], direction]]})
                    # 如果当前帧名已在历史轨迹帧名中
                    if frame_name in id_track[number_of_people_index].keys():
                        id_track[number_of_people_index][frame_name].append(
                            [temp_result_pose[result_pose_index][0], temp_result_pose[result_pose_index][1],
                             theda_diff_frame[temp_result_pose[result_pose_index][0]][
                                 temp_result_pose[result_pose_index][1]], direction])
                    # 如果当前帧名不在历史轨迹帧名中
                    else:
                        id_track[number_of_people_index].update({frame_name: [
                            [temp_result_pose[result_pose_index][0], temp_result_pose[result_pose_index][1],
                             theda_diff_frame[temp_result_pose[result_pose_index][0]][
                                 temp_result_pose[result_pose_index][1]], direction]]})
                    temp_result_pose.pop(result_pose_index)
                    # 满足一个人当前帧存在的落脚点数量等于2,跳出循环取，下一个人
                    if count_foot == 2:
                        temp_number_of_people += 1
                        break
                else:
                    result_pose_index += 1
        # 不符合时间间隔
        else:
            del id_track[number_of_people_index]
            del count_stand_time[number_of_people_index]
    return temp_result_pose, id_track, count_stand_time, temp_number_of_people


# 2020.11.2 09：38
def compute_remaining_points(temp_result_pose, id_track, count_stand_time, interval_thresh, frame_name, pre_step_thresh,
                             number_of_people, temp_number_of_people, logger, theda_diff_frame, direction):
    result_pose_index = 0
    # 获得站立点索引
    while result_pose_index < len(temp_result_pose):
        people_id_index = []
        distance = []
        midpoint_list = []
        for number_of_people_index in list(id_track.keys()):
            # 获取历史轨迹最新的帧名
            frame_index = list(id_track[number_of_people_index].keys())[-1]
            count_stand_time_index = list(count_stand_time[number_of_people_index].keys())
            # 历史帧与当前帧计算差值判断是否连续
            if int(frame_name) - int(frame_index) < interval_thresh:
                if frame_name in id_track[number_of_people_index].keys():
                    # 统计一个人当前帧存在的站立点数量（不超过2）
                    if len(id_track[number_of_people_index][frame_name]) == 2:
                        break
                # 当前帧站立点不存在于已有人的轨迹中
                # 获取当前人历史轨迹最近一帧的最后站立点
                if len(list(count_stand_time[number_of_people_index].keys())) == 2:
                    # 前两步
                    before_two_steps = count_stand_time[number_of_people_index][count_stand_time_index[-2]][-1]
                    # 前一步
                    before_one_step = count_stand_time[number_of_people_index][count_stand_time_index[-1]][-1]
                    # 预测当前步中点
                    pre_current_step = 1.5 * np.array(before_one_step[:2]) - 0.5 * np.array(before_two_steps[:2])
                    # 预测当前步和实际站立点之间的距离
                    # l_cur_pre = compute_distance(temp_result_pose[result_pose_index], pre_current_step)
                    # 计算前一步和当前站立点的中点
                    midpoint_cur_one = compute_midpoint(before_one_step, temp_result_pose[result_pose_index])
                    # 计算前一步和前两步站立点的中点
                    midpoint_two_one = compute_midpoint(before_one_step, before_two_steps)
                    # 预测当前步的中点和实际站立点与前一步的中点的距离
                    l_cur_pre = compute_distance(midpoint_cur_one, pre_current_step)
                    logger.info(
                        "before_two_steps:{}, before_one_step:{}, current_pose:{}, pre_current_step_mid:{}, midpoint_cur_one:{}, midpoint_list:{}".format(
                            before_two_steps,
                            before_one_step,
                            temp_result_pose[result_pose_index],
                            pre_current_step.tolist(),
                            midpoint_cur_one.tolist(),
                            (np.array(midpoint_cur_one) - np.array(midpoint_two_one)).tolist()))
                    logger.info("2 distance:{} thresh:{}".format(l_cur_pre, pre_step_thresh))
                    # if 0.9 * l_cur_pre > pre_step_thresh:
                    if l_cur_pre > pre_step_thresh:
                        temp_number_of_people += 1
                    else:
                        people_id_index.append(number_of_people_index)
                        distance.append(l_cur_pre)
                        midpoint_list.append((np.array(midpoint_cur_one) - np.array(midpoint_two_one)).tolist())
                elif len(list(count_stand_time[number_of_people_index].keys())) > 2:
                    # 前三步
                    before_three_steps = count_stand_time[number_of_people_index][count_stand_time_index[-3]][-1]
                    # 前两步
                    before_two_steps = count_stand_time[number_of_people_index][count_stand_time_index[-2]][-1]
                    # 前一步
                    before_one_step = count_stand_time[number_of_people_index][count_stand_time_index[-1]][-1]
                    # 前三步和前两步中点
                    midpoint_three_two = compute_midpoint(before_three_steps, before_two_steps)
                    # 前两步和前一步中点
                    midpoint_two_one = compute_midpoint(before_two_steps, before_one_step)
                    # 前一步和预测当前步中点
                    midpoint_pre_one = 2 * np.array(midpoint_two_one) - np.array(midpoint_three_two)
                    # 当前步和前一步中点
                    midpoint_cur_one = compute_midpoint(temp_result_pose[result_pose_index], before_one_step)
                    # 预测当前步中点和实际落脚点中点阈值之间的距离
                    l_cur_pre = compute_distance(midpoint_cur_one, midpoint_pre_one)
                    logger.info(
                        "before_three_steps:{}, before_two_steps:{}, before_one_step:{}, current_pose:{}, midpoint_pre_one:{}, midpoint_cur_one:{}, midpoint_list:{}".format(
                            before_three_steps, before_two_steps, before_one_step, temp_result_pose[result_pose_index],
                            midpoint_pre_one.tolist(), midpoint_cur_one.tolist(),
                            (np.array(midpoint_cur_one) - np.array(midpoint_two_one)).tolist()))
                    logger.info("2+ distance:{} thresh:{}".format(l_cur_pre, 0.95 * pre_step_thresh))
                    # 大于阈值则统计人数加1
                    if l_cur_pre > 0.95 * pre_step_thresh:
                        temp_number_of_people += 1
                    else:
                        people_id_index.append(number_of_people_index)
                        distance.append(l_cur_pre)
                        midpoint_list.append((np.array(midpoint_cur_one) - np.array(midpoint_two_one)).tolist())
                else:
                    # 前一步
                    before_one_step = count_stand_time[number_of_people_index][count_stand_time_index[-1]][-1]
                    l_cur_pre = compute_distance(before_one_step, temp_result_pose[result_pose_index])
                    logger.info("1 distance:{} thresh:{}".format(l_cur_pre, 2 * pre_step_thresh))
                    if l_cur_pre < 2 * pre_step_thresh:
                        people_id_index.append(number_of_people_index)
                        distance.append(l_cur_pre)
                        midpoint_list.append([0, 0])
                    else:
                        temp_number_of_people += 1
            # 不符合时间间隔
            else:
                del id_track[number_of_people_index]
                del count_stand_time[number_of_people_index]
        # 所有的历史轨迹遍历完有新轨迹点出现
        if temp_number_of_people > len(id_track):
            number_of_people += 1
            id_track[number_of_people] = {frame_name: [
                [temp_result_pose[result_pose_index][0], temp_result_pose[result_pose_index][1],
                 theda_diff_frame[temp_result_pose[result_pose_index][0]][temp_result_pose[result_pose_index][1]],
                 direction]]}
            count_stand_time[number_of_people] = {frame_name: [
                [temp_result_pose[result_pose_index][0], temp_result_pose[result_pose_index][1],
                 theda_diff_frame[temp_result_pose[result_pose_index][0]][temp_result_pose[result_pose_index][1]],
                 direction]]}
        # 所有的历史轨迹遍历完没有新轨迹点出现且存在距离数组
        elif distance:
            data_index = distance.index(min(distance))
            people_id_num = people_id_index[data_index]
            direction = compute_direction(midpoint_list[data_index])
            # 如果当前数据不在count_stand_time中，count_stand_time为统计的历史站立点数据仅包含站立第一帧数据
            if len(count_stand_time[people_id_num]) > 1:
                if temp_result_pose[result_pose_index] not in np.array(
                        count_stand_time[people_id_num][list(count_stand_time[people_id_num].keys())[-1]])[:,
                                                              :2].tolist() \
                        and temp_result_pose[result_pose_index] not in np.array(
                    count_stand_time[people_id_num][list(count_stand_time[people_id_num].keys())[-2]])[:, :2].tolist():
                    count_stand_time[people_id_num].update(
                        {frame_name: [[temp_result_pose[result_pose_index][0], temp_result_pose[result_pose_index][1],
                                       theda_diff_frame[temp_result_pose[result_pose_index][0]][
                                           temp_result_pose[result_pose_index][1]], direction]]})
            else:
                if temp_result_pose[result_pose_index] not in np.array(
                        count_stand_time[people_id_num][list(count_stand_time[people_id_num].keys())[-1]])[:,
                                                              :2].tolist():
                    count_stand_time[people_id_num].update(
                        {frame_name: [[temp_result_pose[result_pose_index][0], temp_result_pose[result_pose_index][1],
                                       theda_diff_frame[temp_result_pose[result_pose_index][0]][
                                           temp_result_pose[result_pose_index][1]], direction]]})
            # 如果帧名存在，添加站立点
            if frame_name in id_track[people_id_num].keys():
                id_track[people_id_num][frame_name].append(
                    [temp_result_pose[result_pose_index][0], temp_result_pose[result_pose_index][1],
                     theda_diff_frame[temp_result_pose[result_pose_index][0]][temp_result_pose[result_pose_index][1]],
                     direction])
            # 如果帧名不存在，添加站立点
            else:
                id_track[people_id_num].update({frame_name: [
                    [temp_result_pose[result_pose_index][0], temp_result_pose[result_pose_index][1],
                     theda_diff_frame[temp_result_pose[result_pose_index][0]][
                         temp_result_pose[result_pose_index][1]], direction]]})
        temp_result_pose.pop(result_pose_index)
    return id_track, count_stand_time, number_of_people


# 2020.10.16.18:35计算轨迹跟踪
def compute_track_points(result_pose, frame_name, logger, number_of_people, id_track, count_stand_time, interval_thresh,
                         pre_step_thresh, theda_diff_frame):
    direction = 0  # 'None'
    temp_result_pose = copy.deepcopy(result_pose)
    if id_track:
        temp_number_of_people = 1
        temp_result_pose, id_track, count_stand_time, temp_number_of_people = compute_point_whether_exists(
            temp_result_pose, id_track, count_stand_time, interval_thresh, frame_name, temp_number_of_people,
            theda_diff_frame, direction)
        if temp_result_pose:
            id_track, count_stand_time, number_of_people = compute_remaining_points(temp_result_pose, id_track,
                                                                                    count_stand_time, interval_thresh,
                                                                                    frame_name, pre_step_thresh,
                                                                                    number_of_people,
                                                                                    temp_number_of_people, logger,
                                                                                    theda_diff_frame, direction)
    else:
        # 检测到的包含落脚点第一帧
        for i in range(len(temp_result_pose)):
            number_of_people = i
            count_stand_time[number_of_people] = {frame_name: [[temp_result_pose[i][0], temp_result_pose[i][1],
                                                                theda_diff_frame[temp_result_pose[i][0]][
                                                                    temp_result_pose[i][1]], direction]]}
            id_track[number_of_people] = {frame_name: [[temp_result_pose[i][0], temp_result_pose[i][1],
                                                        theda_diff_frame[temp_result_pose[i][0]][
                                                            temp_result_pose[i][1]], direction]]}
    return id_track, count_stand_time, number_of_people


def del_pose_foot(up_foot_list, result_pose, new_down_foot_list, down_foot_list):
    for [i, j] in up_foot_list:
        if [i, j] in result_pose:
            result_pose.remove([i, j])
        if [i, j] in new_down_foot_list:
            new_down_foot_list.remove([i, j])
    for [i, j] in down_foot_list:
        if [i, j] in result_pose:
            result_pose.remove([i, j])
    return result_pose, new_down_foot_list


def diff_image(lines, logger, save_diff_frame_path, save_curve_img_path):
    col, row = 0, 0
    count_frame = 0
    # 落脚阈值
    global down_thresh
    down_thresh = 100  # 300
    # 判断是否为踩地姿态的阈值
    thresh = 200  # 300
    time_count = []
    img_data_list = []
    new_down_foot_list = []
    new_baseline = []
    new_mirror_baseline = []
    theda_diff_frame = []
    # 落脚点间隔阈值
    interval_thresh = 2 * 1000
    pre_step_thresh = 6
    init_count_frame = 30
    start_track_points = {}
    number_of_people = 0
    id_track = {}
    count_stand_time = {}
    curve_plots = {}
    # 从传感器获取数据
    b_thickness = 1
    for i in range(len(lines)):
        # 获取数据的长和宽
        if re.search('timestamp: .*size: (.*) X (.*)', lines[i]):
            frame_name, row, col = re.search('timestamp: .*frame_name: (.*), size: (.*) X (.*)', lines[i]).groups()
            img_data = [[0.0 for col_i in range(int(col))] for row_j in range(int(row))]
            count_row = 0
            yield row, col, down_thresh
        else:
            if len(lines[i]) > 1:
                for j in range(int(col)):
                    img_data[count_row][j] = int(lines[i].strip().split()[j])
                count_row += 1
        # 判断是否获取完整的一帧数据
        if count_row == int(row) and len(lines[i]) > 1:
            logger.info("frame_name：{}".format(frame_name))
            logger.info("current_frame:\n{}".format(pd.DataFrame(img_data)))
            if count_frame >= init_count_frame:
                # time_count.append(int(frame_name))
                time_count.append(
                    # "%s.%03d" % (datetime.strptime(frame_name[:-3], '%Y%m%d%H%M%S'), int(frame_name[-3:])))
                    datetime.strptime(frame_name, '%Y%m%d%H%M%S%f'))
                img_data_list.append(img_data)
                start_time = time.time()
                result_pose = []
                end_down_foot_list = []
                # 判断是否为第一帧数据
                if count_frame == 0 + init_count_frame:
                    previous_frame = copy.deepcopy(img_data)
                    frame0 = copy.deepcopy(img_data)
                    new_baseline.append(frame0)
                    new_mirror_baseline.append(frame0)
                logger.info("previous_frame:\n{}".format(pd.DataFrame(previous_frame)))
                # 上一帧 - 当前帧
                diff_frame = np.array(previous_frame) - np.array(img_data)
                logger.info('diff_frame:\n{}'.format(pd.DataFrame(diff_frame)))
                previous_frame = copy.deepcopy(img_data)
                if count_frame == 1 + init_count_frame:
                    frame1 = copy.deepcopy(img_data)
                    M = compute_mean(frame0, frame1)
                    new_baseline.append(M)
                    new_mirror_baseline.append(M)
                if count_frame == 2 + init_count_frame:
                    frame2 = copy.deepcopy(img_data)
                    M = compute_mean(M, frame2)
                    new_baseline.append(M)
                    new_mirror_baseline.append(M)
                    thresh_data = copy.deepcopy(M)
                    logger.info("thresh_data:\n{}".format(pd.DataFrame(thresh_data)))
                if count_frame == 3 + init_count_frame:
                    frame3 = copy.deepcopy(img_data)
                    M = compute_mean(M, frame3)
                    new_baseline.append(M)
                    new_mirror_baseline.append(M)
                if count_frame == 4 + init_count_frame:
                    frame4 = copy.deepcopy(img_data)
                    M = compute_mean(M, frame4)
                    new_baseline.append(M)
                    new_mirror_baseline.append(M)
                if count_frame == 5 + init_count_frame:
                    frame5 = copy.deepcopy(img_data)
                    M = compute_mean(M, frame5)
                    new_baseline.append(M)
                    new_mirror_baseline.append(M)
                if count_frame == 6 + init_count_frame:
                    frame6 = copy.deepcopy(img_data)
                    M = compute_mean(M, frame6)
                    new_baseline.append(M)
                    new_mirror_baseline.append(M)
                if count_frame == 7 + init_count_frame:
                    frame7 = copy.deepcopy(img_data)
                    M = compute_mean(M, frame7)
                    new_baseline.append(M)
                    new_mirror_baseline.append(M)
                    canvas_start_time = time.time()
                    for row_j in range(int(row)):
                        curve_plots[row_j] = {}
                        for col_i in range(int(col)):
                            curve_plots[row_j][col_i] = {}
                            curve_plots[row_j][col_i]['plot'] = pg.PlotWidget()
                            curve_plots[row_j][col_i]['raw_data'] = curve_plots[row_j][col_i]['plot'].plot(
                                np.array(img_data_list)[:, row_j, col_i], pen='r',
                                symbol='o',
                                symbolPen='r', symbolBrush=0.5,
                                name='{},{}'.format(row_j, col_i), linewidth=1)
                            curve_plots[row_j][col_i]['filter'] = curve_plots[row_j][col_i]['plot'].plot(
                                np.array(new_baseline)[:, row_j, col_i],
                                pen='g',
                                name='{}'.format('filter'), linewidth=1)
                            curve_plots[row_j][col_i]['mirror_filter'] = curve_plots[row_j][col_i]['plot'].plot(
                                np.array(new_mirror_baseline)[:, row_j, col_i] - down_thresh,
                                pen='b', name='{}'.format('filter'),
                                linewidth=1)
                    # print("canvasing time: ", time.time()-canvas_start_time, 'processing frame {}'.format(count_frame))
                if count_frame > 7 + init_count_frame:
                    baseline, mirror_baseline = filtering(
                        [frame0, frame1, frame2, frame3, frame4, frame5, frame6, frame7, img_data], thresh_data)
                    new_baseline.append(baseline)
                    new_mirror_baseline.append(mirror_baseline)
                    # print(len(np.array(img_data_list)[:,0,0]))
                    # print(len(np.array(new_baseline)[:,0,0]))
                    # print(len(np.array(new_mirror_baseline)[:,0,0]))
                    if len(np.array(img_data_list)[:, 0, 0]) != len(np.array(new_baseline)[:, 0, 0]) or len(
                            np.array(img_data_list)[:, 0, 0]) != len(np.array(new_mirror_baseline)[:, 0, 0]) or len(
                            np.array(new_mirror_baseline)[:, 0, 0]) != len(np.array(new_baseline)[:, 0, 0]):
                        print('nonono')
                    yield time_count, img_data_list, new_baseline, new_mirror_baseline
                    # for row_j in range(int(row)):
                    #     for col_i in range(int(col)):
                    #         curve_plots[row_j][col_i]['raw_data'].setData(np.array(img_data_list)[:, row_j, col_i])
                    #         curve_plots[row_j][col_i]['filter'].setData(np.array(new_baseline)[:, row_j, col_i])
                    #         curve_plots[row_j][col_i]['mirror_filter'].setData(np.array(new_mirror_baseline)[:, row_j,
                    #                                                            col_i] - down_thresh)
                    #         ex = pg.exporters.ImageExporter(curve_plots[row_j][col_i]['plot'].plotItem)
                    #         ex.parameters()['width'] = 640
                    #         ex.export(os.path.join(save_curve_img_path, 'filename{}_{}.png'.format(row_j, col_i)))
                    frame3 = copy.deepcopy(frame4)
                    frame4 = copy.deepcopy(frame5)
                    frame5 = copy.deepcopy(frame6)
                    frame6 = copy.deepcopy(frame7)
                    frame7 = copy.deepcopy(img_data)
                    # print("iteration time: ", time.time()-iteration_time, 'processing frame {}'.format(count_frame))
                    # 基线 - 当前帧
                    # print('OOO')
                    theda_diff_frame = np.array(baseline) - np.array(img_data)
                    logger.info('theda_diff_frame:\n{}'.format(pd.DataFrame(theda_diff_frame)))
                    logger.info('baseline:\n{}'.format(pd.DataFrame(baseline)))
                    down_foot_list = compute_down_foot(diff_frame, img_data, mirror_baseline, down_thresh,
                                                       new_down_foot_list, result_pose)
                    logger.info('down_foot_list:\n{}'.format(down_foot_list))
                    new_down_foot_list.extend(down_foot_list)
                    # 获取站立姿态位置和结束站立姿态位置
                    result_pose, end_down_foot_list = compute_pose_foot(img_data, thresh, new_down_foot_list,
                                                                        result_pose, end_down_foot_list,
                                                                        mirror_baseline)
                    if new_down_foot_list and end_down_foot_list:
                        for index in end_down_foot_list:
                            if index in new_down_foot_list and index in down_foot_list:
                                new_down_foot_list.remove(index)
                                down_foot_list.remove(index)
                    # 获取抬脚点位置
                    up_foot_list = compute_up_foot(diff_frame, new_down_foot_list, result_pose)
                    logger.info('up_foot_list:\n{}'.format(up_foot_list))
                    # 删除新的落脚点,去除尖峰噪点
                    result_pose, new_down_foot_list = del_pose_foot(up_foot_list, result_pose, new_down_foot_list,
                                                                    down_foot_list)
                    if result_pose:
                        start_track_points[frame_name] = result_pose
                    logger.info('start_track_points:\n{}'.format(start_track_points))
                    logger.info('new_down_foot_list:\n{}'.format(new_down_foot_list))
                    logger.info('result_pose:\n{}'.format(result_pose))
                    if result_pose:
                        id_track, count_stand_time, number_of_people = compute_track_points(result_pose, frame_name,
                                                                                            logger, number_of_people,
                                                                                            id_track, count_stand_time,
                                                                                            interval_thresh,
                                                                                            pre_step_thresh,
                                                                                            theda_diff_frame)
                        logger.info('id_track:\n{}'.format(id_track))
                        logger.info('count_stand_time:\n{}'.format(count_stand_time))
                        upscale_opencv_start = time.time()
                        result_img = upscale_opencv(img_data, 50, id_track, frame_name, b_thickness)
                        logger.debug("upscale_opencv run time:{:.4f}".format(time.time() - upscale_opencv_start))
                        cv2.imwrite(os.path.join(save_diff_frame_path, str(frame_name) + '.jpg'), result_img)
                        yield result_img, frame_name
                    # draw_curve_img(time_count, img_data_list, new_baseline, new_mirror_baseline, thresh, save_curve_img_path)
                    # print('run time:{:.4f}s'.format(time.time() - start_time))
            # print('count frame: ', count_frame)
            count_frame += 1
    #     if count_frame % (4*init_count_frame) == 0 and count_frame > 0:
    #         for row_j in range(int(row)):
    #             for col_i in range(int(col)):
    #                 curve_plots[row_j][col_i]['raw_data'].setData(np.array(img_data_list)[:, row_j, col_i])
    #                 curve_plots[row_j][col_i]['filter'].setData(np.array(new_baseline)[:, row_j, col_i])
    #                 curve_plots[row_j][col_i]['mirror_filter'].setData(np.array(new_mirror_baseline)[:, row_j,
    #                                                                    col_i] - down_thresh)
    #                 ex = pg.exporters.ImageExporter(curve_plots[row_j][col_i]['plot'].plotItem)
    #                 ex.parameters()['width'] = 640
    #                 ex.export(os.path.join(save_curve_img_path, 'filename{}_{}.png'.format(row_j, col_i)))
    # # draw_curve_img(time_count, img_data_list, new_baseline, save_curve_img_path, thresh, new_mirror_baseline)
    for row_j in range(int(row)):
        for col_i in range(int(col)):
            file = np.array([np.array(time_count),
                             np.array(img_data_list)[:, row_j, col_i],
                             np.array(new_baseline)[:, row_j, col_i],
                             np.array(new_mirror_baseline)[:, row_j, col_i] - down_thresh])
            np.savetxt(os.path.join(save_curve_img_path, 'filename{}_{}.csv'.format(row_j, col_i)), file, fmt="%s", delimiter=",")


def compute_mean(*args):
    M = 0
    for frame in args:
        M += np.array(frame)
    M = M / len(args)
    return M


def filtering(data_list, thresh_data):
    new_data_list = [[[0 for col in range(len(data_list))] for num in range(len(data_list[0][0]))] for row in
                     range(len(data_list[0]))]
    new_baseline = [[[0 for col in range(len(data_list))] for num in range(len(data_list[0][0]))] for row in
                    range(len(data_list[0]))]
    # 帧数
    for frame_index in range(len(data_list)):
        # 行
        for i in range(len(data_list[0])):
            # 列
            for j in range(len(data_list[0][0])):
                new_data_list[i][j][frame_index] = data_list[frame_index][i][j]
    row_index = 0
    while row_index < len(new_data_list):
        col_index = 0
        while col_index < len(new_data_list[0]):
            new_data_list[row_index][col_index].remove(np.array(new_data_list[row_index][col_index]).max(axis=0))
            new_data_list[row_index][col_index].remove(np.array(new_data_list[row_index][col_index]).min(axis=0))
            new_data_list[row_index][col_index] = np.sum(
                np.array(new_data_list[row_index][col_index]) / len(new_data_list[row_index][col_index]))
            new_baseline[row_index][col_index] = -np.array(new_data_list[row_index][col_index]) + 2 * np.array(
                thresh_data[row_index][col_index]) - 500  # 200
            col_index += 1
        row_index += 1
    return new_data_list, new_baseline


def draw_curve_img(time_count, img_data_list, baseline, new_mirror_baseline, thresh, save_curve_img_path):
    for m in range(len(img_data_list[0])):
        for n in range(len(img_data_list[0][0])):
            fig = plt.figure(figsize=[25, 5])
            fig.subplots_adjust(bottom=0.4)
            plt.ylim(-19000, -13500)
            plt.plot(time_count, np.array(img_data_list)[:, m, n], 'r.-', label='{},{}'.format(m, n), linewidth=1)
            plt.plot(time_count, np.array(baseline)[:, m, n], 'g', label='{}'.format('filter'), linewidth=1)
            plt.plot(time_count, np.array(new_mirror_baseline)[:, m, n], 'y', label='{}'.format('baseline'),
                     linewidth=1)
            plt.plot(time_count, (np.array(new_mirror_baseline)[:, m, n] - thresh).tolist(), 'deepskyblue',
                     label='{}'.format('baseline-thresh'), linewidth=1)
            # 不显示刻度
            plt.xticks([])
            plt.legend()
            plt.grid(linestyle="-.")
            plt.savefig(os.path.join(save_curve_img_path, 'filename{}_{}.png'.format(m, n)), dpi=500)
            plt.cla()  # Clear axis即清除当前图形中的当前活动轴。其他轴不受影响


def test_showtime():
    time_count = [20201023113001000, 20201023113102001]
    time_count = ["%s.%03d" % (datetime.strptime(str(content)[:-3], '%Y%m%d%H%M%S'), int(str(content)[-3:])) for content
                  in time_count]
    print("time_count", time_count)


def gen_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def gen_logs(path, name, formatter):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # logging.INFO
    log_info = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    gen_dirs(path)
    log_name = os.path.join(path, log_info + '.log')
    logfile = logging.FileHandler(log_name, mode='w', encoding='utf-8')
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)
    return logger, log_info


def get_frame_data(lines):
    for i in range(len(lines)):
        # 获取数据的长和宽
        if re.search('timestamp: .*size: (.*) X (.*)', lines[i]):
            frame_name, row, col = re.search('timestamp: .*frame_name: (.*), size: (.*) X (.*)', lines[i]).groups()
            img_data = [[0.0 for col_i in range(int(col))] for row_j in range(int(row))]
            count_row = 0
        else:
            if len(lines[i]) > 1:
                for j in range(int(col)):
                    img_data[count_row][j] = int(lines[i].strip().split()[j])
                count_row += 1
        # 判断是否获取完整的一帧数据
        if count_row == int(row) and len(lines[i]) > 1:
            return frame_name, img_data


def main(file_path):
    # test_showtime()
    current_path = os.getcwd()
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置不折叠数据
    pd.set_option('display.expand_frame_repr', False)
    file_path = file_path.split('/')
    txt_file_name = file_path[-1]
    fn_no_extension = os.path.splitext(txt_file_name)[0]
    input_log_path = os.path.join(current_path, 'Logs_txt_data', txt_file_name)
    input_log_dirname = os.path.dirname(input_log_path)
    save_diff_frame_path = os.path.join(input_log_dirname, fn_no_extension, 'diff_result_img')
    save_curve_img_path = os.path.join(input_log_dirname, fn_no_extension, 'curve_img')
    gen_dirs(save_diff_frame_path)
    gen_dirs(save_curve_img_path)
    log_path = os.path.join(current_path, 'Logs', txt_file_name.split('.')[0])
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s:%(message)s")
    logger, log_info = gen_logs(log_path, 'log', formatter)
    with open(input_log_path, 'r') as f:
        lines = f.readlines()
    # pool = Pool()
    # pool.map(creat_thumbnail, images)  # 关键点，images是一个可迭代对象
    # pool.close()
    # pool.join()
    # logger.info("frame_name：{}".format(frame_name))
    # logger.info("current_frame:\n{}".format(pd.DataFrame(img_data)))
    global frames_gen
    frames_gen = diff_image(lines, logger, save_diff_frame_path, save_curve_img_path)

# app = pg.mkQApp()
# app.exec()
# if __name__ == "__main__":
#     main()
