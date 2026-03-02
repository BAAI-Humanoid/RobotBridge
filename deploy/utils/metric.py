import torch

TORSO_INDEX = 15
DESIRED_BODY_INDICES = [0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29]

"""
0  pelvis
1  left_hip_pitch_link
2  left_hip_roll_link
3  left_hip_yaw_link
4  left_knee_link
5  left_ankle_pitch_link
6  left_ankle_roll_link
7  right_hip_pitch_link
8  right_hip_roll_link
9  right_hip_yaw_link
10 right_knee_link
11 right_ankle_pitch_link
12 right_ankle_roll_link
13 waist_yaw_link
14 waist_roll_link
15 torso_link
16 left_shoulder_pitch_link
17 left_shoulder_roll_link
18 left_shoulder_yaw_link
19 left_elbow_link
20 left_wrist_roll_link
21 left_wrist_pitch_link
22 left_wrist_yaw_link
23 right_shoulder_pitch_link
24 right_shoulder_roll_link
25 right_shoulder_yaw_link
26 right_elbow_link
27 right_wrist_roll_link
28 right_wrist_pitch_link
29 right_wrist_yaw_link
"""