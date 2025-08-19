# Copyright 2025 NXP

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
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from itertools import combinations

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading
import transformations

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk
from threading import Lock

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = True


class WindowProgressTable:
    def __init__(self, root, shelf_count):
        self.root = root
        self.root.title("Shelf Objects & QR Link")
        self.root.attributes("-topmost", True)

        self.row_count = 2
        self.col_count = shelf_count

        self.boxes = []
        for row in range(self.row_count):
            row_boxes = []
            for col in range(self.col_count):
                box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
                            relief="solid", font=("Helvetica", 14))
                box.insert(tk.END, "NULL")
                box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
                row_boxes.append(box)
            self.boxes.append(row_boxes)

        # Make the grid layout responsive.
        for row in range(self.row_count):
            self.root.grid_rowconfigure(row, weight=1)
        for col in range(self.col_count):
            self.root.grid_columnconfigure(col, weight=1)

    def change_box_color(self, row, col, color):
        self.boxes[row][col].config(bg=color)

    def change_box_text(self, row, col, text):
        self.boxes[row][col].delete(1.0, tk.END)
        self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
    global box_app
    root = tk.Tk()
    box_app = WindowProgressTable(root, shelf_count)
    root.mainloop()


class WarehouseExplore(Node):
    """ Initializes warehouse explorer node with the required publishers and subscriptions.

        Returns:
            None
    """
    def __init__(self):
        super().__init__('warehouse_explore')

        self.action_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose')

        self.subscription_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pose',
            self.pose_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_global_map = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.global_map_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_simple_map = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.simple_map_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_simple_map = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback_for_shelf_detection,
            QOS_PROFILE_DEFAULT)

        self.subscription_status = self.create_subscription(
            Status,
            '/cerebri/out/status',
            self.cerebri_status_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_behavior = self.create_subscription(
            BehaviorTreeLog,
            '/behavior_tree_log',
            self.behavior_tree_log_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_shelf_objects = self.create_subscription(
            WarehouseShelf,
            '/shelf_objects',
            self.shelf_objects_callback,
            QOS_PROFILE_DEFAULT)

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        self.create_subscription(
            WarehouseShelf,
            "/shelf_data",
            self.fetch_and_process_latest_shelf_data,
            QOS_PROFILE_DEFAULT)

        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Publisher for output image (for debug purposes).
        self.publisher_qr_decode = self.create_publisher(
            CompressedImage,
            "/debug_images/qr_code",
            QOS_PROFILE_DEFAULT)

        self.publisher_shelf_data = self.create_publisher(
            WarehouseShelf,
            "/shelf_data",
            QOS_PROFILE_DEFAULT)

        self.declare_parameter('shelf_count', 1)
        self.declare_parameter('initial_angle', 0.0)

        self.shelf_count = \
            self.get_parameter('shelf_count').get_parameter_value().integer_value
        self.initial_angle = \
            self.get_parameter('initial_angle').get_parameter_value().double_value

        # --- Robot State ---
        self.armed = False
        self.logger = self.get_logger()

        # --- Robot Pose ---
        self.pose_curr = PoseWithCovarianceStamped()
        self.buggy_pose_x = 0.0
        self.buggy_pose_y = 0.0
        self.buggy_center = (0.0, 0.0)
        self.world_center = (0.0, 0.0)

        # --- Map Data ---
        self.simple_map_curr = None
        self.global_map_curr = None

        # --- Goal Management ---
        self.xy_goal_tolerance = 0.5
        self.goal_completed = True  # No goal is currently in-progress.
        self.goal_handle_curr = None
        self.cancelling_goal = False
        self.recovery_threshold = 10

        # --- Goal Creation ---
        self._frame_id = "map"

        # --- Exploration Parameters ---
        self.max_step_dist_world_meters = 7.0
        self.min_step_dist_world_meters = 4.0
        self.full_map_explored_count = 0

        # --- QR Code Data ---
        self.qr_code_str = "Empty"
        if PROGRESS_TABLE_GUI:
            self.table_row_count = 0
            self.table_col_count = 0

        # --- Shelf Data ---
        self.shelf_objects_curr = WarehouseShelf()

        # -------------------------------------------------------------
        
        # storing initial pose and orientation ----------------------------
        self.initial_pose_stored = False
        self.initial_pose_x = 0.0
        self.initial_pose_y = 0.0
        self.initial_yaw = 0.0

        # self.declare_parameter('initial_angle', 0.0)
        self.shelf_count = self.get_parameter('shelf_count').get_parameter_value().integer_value
        self.initial_angle = self.get_parameter('initial_angle').get_parameter_value().double_value

        self.shelf_data_msg = WarehouseShelf()


        # --- Initial angle for shelf ---------------------------------------
        self.shelf_angle = 0.0
        self._angle_lock = threading.Lock()
        # -------------------------------------------------------------------
        # --- Shelf Detection ---
        self.map_ready_for_detection = False
        self.go_to_QR=False

        # ---Buggy Actions---
        self.shelf_coordinates_stored = False  

	    # ---Ordinates for buggy for scanning---
        self.shelf_coordinates = np.zeros((self.shelf_count, 5, 2))

        self.scan_started = False
        self.timer = self.create_timer(1.0, self.check_and_start_scanning)
        
        #---navigation updates---
        self.navigation_goal = None
        self.goal_reached_callback = None
        self.goal_check_timer = None
        
        self.scan_delay = 2.0  # seconds to wait at each scan point
        self.last_scan_time = 0
        self.scan_delay_active = False
        self.current_scan_data = None  # To store fresh sensor data

        #--No of shelves matched---
        self.num_matched_shelves = 0
        #---QR Code String---
        self.qr_code_str = ""

        self.shelf_id = 1




    def pose_callback(self, message):
        """Callback function to handle pose updates.

        Args:
            message: ROS2 message containing the current pose of the rover.

        Returns:
            None
        """
        self.pose_curr = message
        self.buggy_pose_x = message.pose.pose.position.x
        self.buggy_pose_y = message.pose.pose.position.y
        self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)

        # Store initial pose once
        if not self.initial_pose_stored:
            self.initial_pose_x = self.buggy_pose_x
            self.initial_pose_y = self.buggy_pose_y

            # Extract yaw from quaternion
            q = message.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            self.initial_yaw = math.atan2(siny_cosp, cosy_cosp)

            self.initial_pose_stored = True

    def simple_map_callback(self, message):
        """Callback function to handle simple map updates.

        Args:
            message: ROS2 message containing the simple map data.

        Returns:
            None
        """
        self.simple_map_curr = message
        map_info = self.simple_map_curr.info
        self.world_center = self.get_world_coord_from_map_coord(
            map_info.width / 2, map_info.height / 2, map_info
        )

    def global_map_callback(self, message):
        """Callback function to handle global map updates.

        Args:
            message: ROS2 message containing the global map data.

        Returns:
            None
        """
        self.global_map_curr = message

        if not self.goal_completed:
            return

        height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
        map_array = np.array(self.global_map_curr.data).reshape((height, width))

        frontiers = self.get_frontiers_for_space_exploration(map_array)

        map_info = self.global_map_curr.info
        if frontiers:
            closest_frontier = None
            min_distance_curr = float('inf')

            for fy, fx in frontiers:
                fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy,
                                             map_info)
                distance = euclidean((fx_world, fy_world), self.buggy_center)
                if (distance < min_distance_curr and
                    distance <= self.max_step_dist_world_meters and
                    distance >= self.min_step_dist_world_meters):
                    min_distance_curr = distance
                    closest_frontier = (fy, fx)

            if closest_frontier:
                fy, fx = closest_frontier
                goal = self.create_goal_from_map_coord(fx, fy, map_info)
                self.send_goal_from_world_pose(goal)
                print("Sending goal for space exploration.")
                return
            else:
                self.max_step_dist_world_meters += 2.0
                new_min_step_dist = self.min_step_dist_world_meters - 1.0
                self.min_step_dist_world_meters = max(0.25, new_min_step_dist)

            self.full_map_explored_count = 0
        elif self.full_map_explored_count < 2:
            self.full_map_explored_count += 1
            print(f"Nothing found in frontiers; count = {self.full_map_explored_count}")
            self.map_ready_for_detection = True

    def get_frontiers_for_space_exploration(self, map_array):
        """Identifies frontiers for space exploration.

        Args:
            map_array: 2D numpy array representing the map.

        Returns:
            frontiers: List of tuples representing frontier coordinates.
        """
        frontiers = []
        for y in range(1, map_array.shape[0] - 1):
            for x in range(1, map_array.shape[1] - 1):
                if map_array[y, x] == -1:  # Unknown space and not visited.
                    neighbors_complete = [
                        (y, x - 1),
                        (y, x + 1),
                        (y - 1, x),
                        (y + 1, x),
                        (y - 1, x - 1),
                        (y + 1, x - 1),
                        (y - 1, x + 1),
                        (y + 1, x + 1)
                    ]

                    near_obstacle = False
                    for ny, nx in neighbors_complete:
                        if map_array[ny, nx] > 0:  # Obstacles.
                            near_obstacle = True
                            break
                    if near_obstacle:
                        continue

                    neighbors_cardinal = [
                        (y, x - 1),
                        (y, x + 1),
                        (y - 1, x),
                        (y + 1, x),
                    ]

                    for ny, nx in neighbors_cardinal:
                        if map_array[ny, nx] == 0:  # Free space.
                            frontiers.append((ny, nx))
                            break

        return frontiers


    def camera_image_callback(self, message):
        """Callback function to handle incoming camera images.

        Args:
            message: ROS2 message of the type sensor_msgs.msg.CompressedImage.

        Returns:
            None
        """
        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image from camera feed")

            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(image)

            if bbox is not None and data:
                self.qr_code_str = data
                # self.get_logger().info(f"[QR Detected] {data}")

                shelf_msg = WarehouseShelf()
                shelf_msg.object_name = ["No objects detected yet."]
                shelf_msg.object_count = [0]
                shelf_msg.qr_decoded = self.qr_code_str
                # self.publisher_shelf_data.publish(shelf_msg)

            # Optional line for visualizing image on foxglove.
            self.publish_debug_image(self.publisher_qr_decode, image)

        except Exception as e:
            self.get_logger().error(f"[camera_image_callback] Runtime error: {str(e)}")

    def shelf_objects_callback(self, message):
        # self.get_logger().info("Shelf objects callback triggered")
        """Callback function to handle shelf objects updates.

        Args:
            message: ROS2 message containing shelf objects data.

        Returns:
            None
        """
        self.shelf_objects_curr = message
        # Process the shelf objects as needed.

        # How to send WarehouseShelf messages for evaluation.
        """
        * Example for sending WarehouseShelf messages for evaluation.
            shelf_data_message = WarehouseShelf()

            shelf_data_message.object_name = ["car", "clock"]
            shelf_data_message.object_count = [1, 2]
            shelf_data_message.qr_decoded = "test qr string"

            self.publisher_shelf_data.publish(shelf_data_message)
        """
        #* Alternatively, you may store the QR for current shelf as self.qr_code_str.
        if self.shelf_objects_curr.object_name:
            self.shelf_objects_curr.qr_decoded = self.qr_code_str
            # self.publisher_shelf_data.publish(self.shelf_objects_curr)
        #This, will publish the current detected objects with the last QR decoded.
        

        # Optional code for populating TABLE GUI with detected objects and QR data.
        """
        if PROGRESS_TABLE_GUI:
            shelf = self.shelf_objects_curr
            obj_str = ""
            for name, count in zip(shelf.object_name, shelf.object_count):
                obj_str += f"{name}: {count}\n"

            box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
            box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
            self.table_row_count += 1

            box_app.change_box_text(self.table_row_count, self.table_col_count, self.qr_code_str)
            box_app.change_box_color(self.table_row_count, self.table_col_count, "yellow")
            self.table_row_count = 0
            self.table_col_count += 1
        """

    def get_diagonal_endpoints(self,world_coords, expected_length=1.458, tolerance=0.10):

        if len(world_coords) < 2:
            print("[WARN] Not enough points for diagonal calculation.")
            return None
        min_valid = expected_length * (1 - tolerance)
        max_valid = expected_length * (1 + tolerance)
        
        self.go_to_QR=False

        diagonals = []
        pair_dists = []
        for (i, p1), (j, p2) in combinations(enumerate(world_coords), 2):
            dist = euclidean(p1, p2)
            diagonals.append(dist)
            if min_valid <= dist <= max_valid:
               pair = tuple(sorted((i, j)))
               pair_dists.append((dist, pair, p1, p2))

        if not pair_dists:
           print("[WARN] No valid diagonal found.")
           return None

        # Ensure unique diagonals (no shared endpoints)
        selected_pairs = []
        used_indices = set()
        for dist, pair, p1, p2 in sorted(pair_dists, reverse=True, key=lambda x: x[0]):
           if not (pair[0] in used_indices or pair[1] in used_indices):
               selected_pairs.append((p1, p2, dist))
               used_indices.update(pair)
           if len(selected_pairs) == 2:
               break

        if len(selected_pairs) == 2:
           print("[INFO] Found two diagonal pairs.")
           (p1a, p1b, _), (p2a, p2b, _) = selected_pairs
        else:
           print("[WARN] Could not find two unique diagonals.")
           return None

        # COM
        com_x = (p1a[0] + p1b[0] + p2a[0] + p2b[0]) / 4
        com_y = (p1a[1] + p1b[1] + p2a[1] + p2b[1]) / 4

        # Orientation
        def safe_yaw(a, b):
           if a == b:
               print("[WARN] Identical points for yaw calculation.")
               return 0.0
           return self.create_yaw_from_vector(a[0], a[1], b[0], b[1])

        if (p1a[0], p1a[1]) > (p1b[0], p1b[1]):
            yaw1 = safe_yaw(p1a, p1b)
        else:
            yaw1 = safe_yaw(p1b, p1a)
        if (p2a[0], p2a[1]) > (p2b[0], p2b[1]):
            yaw2 = safe_yaw(p2a, p2b)
        else:
            yaw2 = safe_yaw(p2b, p2a)
        
        # Print the yaw angles in degrees for debugging
        print(f"Yaw1: {math.degrees(yaw1):.2f}°, Yaw2: {math.degrees(yaw2):.2f}°")
        
        # Compute absolute angle difference in degrees
        yaw_diff = abs(math.degrees(yaw1 - yaw2)) % 180  # limit to [0, 180)
        
        # Print the yaw difference for debugging
        print(f"Yaw difference: {yaw_diff:.2f}°")

        # Custom acceptance/rejection logic
        if yaw_diff == 0:
            print("[WARN] Rejected: yaw difference is 0° (parallel diagonals).")
            return None
        if 80 <= yaw_diff <= 100:
            print(f"[WARN] Rejected: yaw difference {yaw_diff:.2f}° (square-like).")
            return None
        if not (33 <= yaw_diff <= 45 or 130 <= yaw_diff <= 145):
             print(f"[WARN] Rejected: yaw difference {yaw_diff:.2f}° (not in rectangle range).")
             return None
        if  33 <= yaw_diff <= 45 :
            self.go_to_QR = True   
    
        shelf_yaw = (yaw1 + yaw2) / 2

        return (com_x, com_y, shelf_yaw)

    def map_callback_for_shelf_detection(self, message):

        self.shelf_coordinates = [[[0.0, 0.0] for _ in range(5)] for _ in range(self.shelf_count)]

        if not self.map_ready_for_detection:
            return
        
        #If shelf_coordinates is NOT all zeros, skip detection
        #if not np.all(self.shelf_coordinates == 0):
            #return  

        self.simple_map_curr = message
        map_info = self.simple_map_curr.info
        width = map_info.width
        height = map_info.height

        # 1. Convert occupancy data to numpy array
        map_array = np.array(message.data).reshape((height, width))

        # 2. Mark as occupied
        shelf_mask = (map_array == 100).astype(np.uint8)

        # 3. Label connected components
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_clusters = label(shelf_mask, structure=structure)
        shelf_detected=0

        for cluster_id in range(1, num_clusters + 1):
            cluster_coords = np.argwhere(labeled_array == cluster_id)
            world_coords = [self.get_world_coord_from_map_coord(x, y, map_info) for y, x in cluster_coords]
            diagonals = self.get_diagonal_endpoints(world_coords)
            cluster_size = len(cluster_coords)
            if 0 <cluster_size < 80 :
              print(f"[INFO] Cluster {cluster_id} rejected due to size: {cluster_size}")
              continue  # Skip this cluster
            if diagonals:
                com_x, com_y, yaw = diagonals
                offset = 2 # Offset from center to corners 
                offset_ = 2.70
                sin_yaw = math.sin(yaw)
                cos_yaw = math.cos(yaw)
                if self.go_to_QR:
                    self.shelf_coordinates[shelf_detected][0] = [com_x, com_y]
                    self.shelf_coordinates[shelf_detected][1] = [com_x + offset * cos_yaw, com_y + offset * sin_yaw]  # Front
                    self.shelf_coordinates[shelf_detected][2] = [com_x - offset * cos_yaw, com_y - offset * sin_yaw]  # Back
                    self.shelf_coordinates[shelf_detected][3] = [com_x - offset_ * sin_yaw, com_y + offset_ * cos_yaw]  # Left
                    self.shelf_coordinates[shelf_detected][4] = [com_x + offset_ * sin_yaw, com_y - offset_ * cos_yaw]  # Right
                    shelf_detected += 1
                else:
                    self.shelf_coordinates[shelf_detected][0] = [com_x, com_y]
                    self.shelf_coordinates[shelf_detected][3] = [com_x + offset_ * cos_yaw, com_y + offset_ * sin_yaw]  # Front
                    self.shelf_coordinates[shelf_detected][4] = [com_x - offset_* cos_yaw, com_y - offset_ * sin_yaw]  # Back
                    self.shelf_coordinates[shelf_detected][1] = [com_x - offset * sin_yaw, com_y + offset * cos_yaw]  # Left
                    self.shelf_coordinates[shelf_detected][2] = [com_x + offset * sin_yaw, com_y - offset * cos_yaw]  # Right
                    shelf_detected += 1

            else:
                print(f"[INFO] Cluster {cluster_id} did not pass diagonal check.")

        for shelf_idx, points in enumerate(self.shelf_coordinates):
           print(f"Shelf {shelf_idx}:")
           for point_idx, (x, y) in enumerate(points):
              print(f"  Point {point_idx}: x={x:.2f}, y={y:.2f}")
        
        self.shelf_coordinates_stored = True
        self.map_ready_for_detection = False  # Reset for next detection cycle
        # self.identify_and_scan_shelf_list(self.shelf_coordinates)

    def check_and_start_scanning(self):
        if not self.scan_started and getattr(self, 'shelf_coordinates_stored', False):
            self.get_logger().info("All shelves detected. Starting scan navigation.")
            self.scan_started = True
            self.identify_and_scan_shelf_list_for_all(self.shelf_coordinates)

    def find_matching_shelf(self, shelf_list, initial_angle_deg, reference_point, initial_yaw, angle_tolerance_deg=10.0):
        """
        Finds the shelf aligned with the robot's initial direction.

        Args:
        shelf_list (list): List of shelves, each as [ [cx,cy], [x1,y1], ..., [x4,y4] ].
        initial_angle_deg (float): Robot's initial facing angle (degrees).
        reference_point (list): [x, y] position of robot.
        initial_yaw (float): Robot's yaw angle in radians (from IMU).
        angle_tolerance_deg (float): Acceptable angle difference threshold.

        Returns:
        Tuple: (matched_shelf_idx, matched_shelf_data) or (-1, None) if not found.
        """
        initial_x, initial_y = reference_point
        initial_angle_rad = math.radians(initial_angle_deg)

        def normalize_angle(angle):
           return math.atan2(math.sin(angle), math.cos(angle))

        matched_shelves = {}

        for shelf_idx, shelf_data in enumerate(shelf_list):
            if len(shelf_data) < 1:
               continue  # Skip invalid entries

            cx, cy = shelf_data[0]
            dx = cx - initial_x
            dy = cy - initial_y

            # Get angle from robot to shelf in radians, normalized to [0, 2π)
            angle_to_shelf = math.atan2(dy, dx)
            if angle_to_shelf < 0:
                angle_to_shelf += 2 * math.pi
            relative_angle = angle_to_shelf - initial_yaw
            angle_diff = abs(normalize_angle(relative_angle - initial_angle_rad))
            angle_diff_deg = math.degrees(angle_diff)
            
            self.get_logger().info(
                   f"Shelf[{shelf_idx}] COM=({cx:.2f},{cy:.2f}), "
                   f"ref=({initial_x:.2f},{initial_y:.2f}), "
                   f"angle_to_shelf={math.degrees(angle_to_shelf):.2f}°, "
                   f"initial_yaw={math.degrees(initial_yaw):.2f}°, "
                   f"initial_angle={initial_angle_deg:.2f}°, "
                   f"angle_diff={angle_diff_deg:+.2f}°"
            )

            if angle_diff_deg <= angle_tolerance_deg:
                matched_shelves[shelf_idx] = shelf_data



        if not matched_shelves:
            self.get_logger().warn("No shelves matched initial direction.")
        else:
            self.get_logger().info(f"Matched shelves: {list(matched_shelves.keys())}")

        return matched_shelves
    
    def _check_goal_reached(self):
        if not self.navigation_goal or not hasattr(self, 'buggy_center'):
            return  # wait for data

        curr_x, curr_y = self.buggy_center
        goal_x, goal_y = self.navigation_goal
        dist = math.hypot(curr_x - goal_x, curr_y - goal_y)

        # Get current yaw from robot pose
        q = self.pose_curr.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Get desired yaw from the stored goal pose orientation
        if hasattr(self, 'navigation_goal_pose'):
            qg = self.navigation_goal_pose.pose.orientation
            siny_cosp_g = 2 * (qg.w * qg.z + qg.x * qg.y)
            cosy_cosp_g = 1 - 2 * (qg.y * qg.y + qg.z * qg.z)
            desired_yaw = math.atan2(siny_cosp_g, cosy_cosp_g)
        else:
            desired_yaw = current_yaw  # fallback if goal pose not stored
    
        yaw_diff = abs(math.degrees(self.normalize_angle(current_yaw - desired_yaw)))
        yaw_tolerance_deg = 5.0
    
        if dist < self.goal_tolerance and yaw_diff < yaw_tolerance_deg:
            self.get_logger().info(
                f"[Reached] Goal at ({goal_x:.2f}, {goal_y:.2f}) within {dist:.2f} m and aligned within {yaw_diff:.2f}°"
            )
            if self.goal_check_timer:
                self.goal_check_timer.cancel()
            self.navigation_goal = None
    
            if self.goal_reached_callback:
                callback_fn = self.goal_reached_callback
                self.goal_reached_callback = None
                callback_fn()
    
        elif time.time() - self.goal_start_time > self.goal_timeout:
            self.get_logger().warn(
                f"[Timeout] Failed to reach goal ({goal_x:.2f}, {goal_y:.2f}) in {self.goal_timeout} s."
            )
            if self.goal_check_timer:
                self.goal_check_timer.cancel()
                self.goal_check_timer = None
            self.navigation_goal = None
            if self.goal_reached_callback:
                callback_fn = self.goal_reached_callback
                self.goal_reached_callback = None
                callback_fn()

    def normalize_angle(self, angle_rad):
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
 
    def calculate_dynamic_timeout(self, distance):
        """Calculate timeout based on distance and speed"""
        BASE_TIMEOUT = 20.0  # seconds
        METERS_PER_SECOND = 0.5  # Conservative speed estimate
        return max(BASE_TIMEOUT, 10 + distance / METERS_PER_SECOND * 1.2)  # 50% buffer


    def wait_until_reached(self, goal_x, goal_y, callback=None, tolerance=0.3):
        curr_x, curr_y = self.buggy_center
        distance = math.hypot(curr_x - goal_x, curr_y - goal_y)
    
        self.goal_timeout = self.calculate_dynamic_timeout(distance)
        self.navigation_goal = (goal_x, goal_y)
        self.goal_reached_callback = callback
        self.goal_start_time = time.time()
        self.goal_tolerance = tolerance

        if self.goal_check_timer:
            self.goal_check_timer.cancel()
        self.goal_check_timer = self.create_timer(0.1, self._check_goal_reached)

    def start_next_shelf(self):
        if not self.shelf_list:
            self.get_logger().info("✅ All shelves scanned.")
            return

        # Collect all matched shelves (dictionary of index → shelf_data)
        self.matched_shelves = self.find_matching_shelf(
            self.shelf_list, self.current_angle,
            self.reference_point, self.initial_yaw,
            self.angle_tolerance_deg
        )

        if not self.matched_shelves:
            self.get_logger().warn("⚠️ No aligned shelves found.")
            return

        self.matched_shelf_queue = list(self.matched_shelves.items())  # Convert to list of tuples
        self.try_next_matched_shelf()

    def try_next_matched_shelf(self):
        """Try the next shelf from matched candidates."""
        if not self.matched_shelf_queue:
            self.get_logger().error("❌ All matched shelves tried. None matched required QR shelf ID.")
            return

        matched_idx, matched_data = self.matched_shelf_queue.pop(0)
        self.get_logger().info(f"🔄 Trying matched shelf index {matched_idx}")

        self.current_shelf_index = matched_idx
        self.current_shelf_data = matched_data
        self.current_scan_point = 1

        self.start_next_scan_point()

    
    def start_next_scan_point(self):
        if self.current_scan_point > 4:
            self.get_logger().info("✅ Finished scanning current shelf.")
            # Update reference point and angle for next shelf
            self.reference_point = self.current_shelf_data[0]  # center
            self.initial_yaw = 0  # reset
            self.current_angle = self.shelf_angle
        
            # Remove scanned shelf
            self.shelf_list.pop(self.current_shelf_index)

            # Move to next shelf
            self.start_next_shelf()
            return

        # Get pose toward the current scan point
        pose = self.create_pose_stamped(
            self.current_shelf_data[self.current_scan_point],
            self.current_shelf_data[0]
        )

        goal_x = pose.pose.position.x
        goal_y = pose.pose.position.y

        self.get_logger().info(f"📍 Navigating to Scan Point {self.current_scan_point}")
        self.send_goal_from_world_pose(pose)

        # Store current scan point before incrementing
        current_point = self.current_scan_point
    
        # Move to next scan point after this one completes
        self.current_scan_point += 1
    
        # Custom callback based on scan point
        if current_point == 1:
            self.wait_until_reached(goal_x, goal_y, callback=self.handle_point1_reached)
        elif current_point == 2:
            self.wait_until_reached(goal_x, goal_y, callback=self.handle_point2_reached)
        elif current_point == 3:
            self.wait_until_reached(goal_x, goal_y, callback=self.handle_point3_reached)
        else:  # point 4
            self.wait_until_reached(goal_x, goal_y, callback=self.handle_point4_reached)
    
    def handle_point1_reached(self):
        """Handle logic after reaching point 1 (QR scan point)"""
        qr_found = False

        if self.qr_code_str and self.qr_code_str.lower() != "empty":
            try:
                parts = self.qr_code_str.split('_')
                shelf_id_no = int(parts[0])
                if self.shelf_id == shelf_id_no:
                    # ✅ Correct shelf found
                    qr_found = True
                    shelf_msg = WarehouseShelf()
                    shelf_msg.object_name = ["No objects detected yet."]
                    shelf_msg.object_count = [0]
                    shelf_msg.qr_decoded = self.qr_code_str
    
                    self.publisher_shelf_data.publish(shelf_msg)
                    self.get_logger().info(f"✅ Correct shelf QR at point 1: {self.qr_code_str}")

                    self.shelf_id += 1
                    self.current_scan_point = 3  # Skip point 2
            except Exception as e:
                self.get_logger().error(f"Invalid QR format at point 1: {self.qr_code_str} — {e}")
        else:
            self.get_logger().warn("⚠️ No QR code detected at point 1")

        if qr_found:
            self.start_next_scan_point()
        else:
            # Fallback to point 2 to retry QR scan
            self.get_logger().info("🔁 Going to point 2 for secondary QR check")
            self.start_next_scan_point()

    
    def handle_point2_reached(self):
        """Handle logic after reaching point 2 (secondary QR scan point)"""
        qr_found = False

        if self.qr_code_str and self.qr_code_str.lower() != "empty":
            try:
                parts = self.qr_code_str.split('_')
                shelf_id_no = int(parts[0])
                if self.shelf_id == shelf_id_no:
                    qr_found = True
                    shelf_msg = WarehouseShelf()
                    shelf_msg.object_name = ["No objects detected yet."]
                    shelf_msg.object_count = [0]
                    shelf_msg.qr_decoded = self.qr_code_str
    
                    self.publisher_shelf_data.publish(shelf_msg)
                    self.get_logger().info(f"✅ Correct shelf QR at point 2: {self.qr_code_str}")
    
                    self.shelf_id += 1
                    self.current_scan_point = 3
            except Exception as e:
                self.get_logger().error(f"Invalid QR format at point 2: {self.qr_code_str} — {e}")
        else:
            self.get_logger().warn("⚠️ No QR code detected at point 2")
    
        if qr_found:
            self.start_next_scan_point()
        else:
            self.get_logger().warn("❌ Failed to verify shelf at point 2 — trying next matched shelf.")
            self.try_next_matched_shelf()

    def handle_point3_reached(self):
        """Handle logic after reaching point 3 (object detection point)"""
        # Publish current object detection data
        if hasattr(self, 'shelf_objects_curr') and self.shelf_objects_curr.object_name:
            total_objects = sum(self.shelf_objects_curr.object_count)
            if total_objects >= 6:
                self.get_logger().info(f"✅ Found {total_objects} objects (minimum requirement met)")
                # Publish data on topic shelf_data
                self.shelf_objects_curr.qr_decoded = self.qr_code_str
                self.publisher_shelf_data.publish(self.shelf_objects_curr)
                # Skip point 4 and go to next shelf
                self.current_scan_point = 5  # Will trigger shelf completion
                self.start_next_scan_point()
            else:
                self.get_logger().warn(f"⚠️ Only found {total_objects} objects, proceeding to point 4")
                # Publish data and proceed to point 4
                self.shelf_objects_curr.qr_decoded = self.qr_code_str
                self.publisher_shelf_data.publish(self.shelf_objects_curr)
                self.start_next_scan_point()
        else:
            self.get_logger().error("No object detection data available at point 3")
            self.start_next_scan_point()

    def handle_point4_reached(self):
        """Handle logic for when point 4 is reached"""
        if hasattr(self, 'shelf_objects_curr'):
            self.shelf_objects_curr.qr_decoded = self.qr_code_str
            self.publisher_shelf_data.publish(self.shelf_objects_curr)
            self.get_logger().info("Published object data from point 4")
        else:
            self.get_logger().warn("No object data available at point 4")
        self.start_next_scan_point()


    # --- Subfunction: Create Oriented Pose (unchanged) ---
    def create_pose_stamped(self,target_point, face_point):
            """
            Creates a PoseStamped where the robot will be placed at target_point
            and oriented to face face_point.

            Args:
                target_point (Tuple[float, float]): The (x, y) position for the goal.
                face_point (Tuple[float, float]): The (x, y) point to face toward.

            Returns:
                PoseStamped: The goal pose with orientation facing the face_point.
            """
            x, y = target_point
            fx, fy = face_point

            # Step 1: Calculate yaw from target_point toward face_point
            yaw = self.create_yaw_from_vector(fx, fy, x, y)

            # Step 2: Use helper to create the full pose with orientation
            return self.create_goal_from_world_coord(x, y, yaw)


    def identify_and_scan_shelf_list_for_all(self, shelf_list, angle_tolerance_deg=10.0):
        """
        Iteratively identifies and scans all shelves aligned with current direction.

        Args:
            shelf_list (list): List of shelves, each as [[x0,y0], [x1,y1], ..., [x4,y4]]
            angle_tolerance_deg (float): Tolerance to match shelf orientation
        """
        # --- Ensure required initial state ---
        if not hasattr(self, 'initial_pose_x') or not hasattr(self, 'initial_angle'):
           self.get_logger().error("Initial pose or angle not set.")
           return

        # Step 2: Store required state
        self.reference_point = [self.initial_pose_x, self.initial_pose_y]
        self.current_angle = self.initial_angle
        self.initial_yaw = self.initial_yaw
        self.angle_tolerance_deg = angle_tolerance_deg
        self.shelf_list = shelf_list

        # Step 3: Start scanning
        self.start_next_shelf()


    def publish_debug_image(self, publisher, image):
        """Publishes images for debugging purposes.

        Args:
            publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
            image: Image given by an n-dimensional numpy array.

        Returns:
            None
        """
        if image.size:
            message = CompressedImage()
            _, encoded_data = cv2.imencode('.jpg', image)
            message.format = "jpeg"
            message.data = encoded_data.tobytes()
            publisher.publish(message)

    

    def cerebri_status_callback(self, message):
        """Callback function to handle cerebri status updates.

        Args:
            message: ROS2 message containing cerebri status.

        Returns:
            None
        """
        if message.mode == 3 and message.arming == 2:
            self.armed = True
        else:
            # Initialize and arm the CMD_VEL mode.
            msg = Joy()
            msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
            msg.axes = [0.0, 0.0, 0.0, 0.0]
            self.publisher_joy.publish(msg)

    def behavior_tree_log_callback(self, message):
        """Alternative method for checking goal status.

        Args:
            message: ROS2 message containing behavior tree log.

        Returns:
            None
        """
        for event in message.event_log:
            if (event.node_name == "FollowPath" and
                event.previous_status == "SUCCESS" and
                event.current_status == "IDLE"):
                # self.goal_completed = True
                # self.goal_handle_curr = None
                pass

    

    # function for encodeing angle info and storing and publishing encoded info.
    def fetch_and_process_latest_shelf_data(self, msg: WarehouseShelf):
        
        """
        Callback for the /shelf_data topic.
        Extracts and logs the shelf angle from a QR code like: '1_315.0_xyz...'
        """
        try:
            if not hasattr(msg, 'qr_decoded'):
                self.get_logger().warn("Missing qr_decoded in shelf_data message.")
                return

            qr_data = msg.qr_decoded.strip()

            if not qr_data or qr_data.lower() == "empty":
                self.get_logger().debug("Empty or placeholder QR code received.")
                return

            # Split by underscore and extract angle
            parts = qr_data.split("_")
            if len(parts) < 2:
                self.get_logger().warn(f"QR code format invalid: '{qr_data}'")
                return

            try:
                shelf_angle = float(parts[1])
                with self._angle_lock:
                   if not hasattr(self, 'shelf_angle') or abs(self.shelf_angle - shelf_angle) > 1e-3:
                        self.shelf_angle = shelf_angle
                        self.get_logger().info(f"[QR Angle Updated] Shelf Angle: {shelf_angle}°")
            except ValueError:
                    self.get_logger().error(f"Invalid angle in QR string: '{parts[1]}'")

        except Exception as e:
            self.get_logger().error(f"Error in shelf_data_callback: {e}")


    def rover_move_manual_mode(self, speed, turn):
        """Operates the rover in manual mode by publishing on /cerebri/in/joy.

        Args:
            speed: The speed of the car in float. Range = [-1.0, +1.0];
                   Direction: forward for positive, reverse for negative.
            turn: Steer value of the car in float. Range = [-1.0, +1.0];
                  Direction: left turn for positive, right turn for negative.

        Returns:
            None
        """
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, speed, 0.0, turn]
        self.publisher_joy.publish(msg)

    def cancel_goal_callback(self, future):
        """
        Callback function executed after a cancellation request is processed.

        Args:
            future (rclpy.Future): The future is the result of the cancellation request.
        """
        cancel_result = future.result()
        if cancel_result:
            self.logger.info("Goal cancellation successful.")
            self.cancelling_goal = False  # Mark cancellation as completed (success).
            return True
        else:
            self.logger.error("Goal cancellation failed.")
            self.cancelling_goal = False  # Mark cancellation as completed (failed).
            return False

    def cancel_current_goal(self):
        """Requests cancellation of the currently active navigation goal."""
        if self.goal_handle_curr is not None and not self.cancelling_goal:
            self.cancelling_goal = True  # Mark cancellation in-progress.
            self.logger.info("Requesting cancellation of current goal...")
            cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
            cancel_future.add_done_callback(self.cancel_goal_callback)

    def goal_result_callback(self, future):
        """
        Callback function executed when the navigation goal reaches a final result.

        Args:
            future (rclpy.Future): The future that is result of the navigation action.
        """
        status = future.result().status
        # NOTE: Refer https://docs.ros2.org/foxy/api/action_msgs/msg/GoalStatus.html.

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.logger.info("Goal completed successfully!")
        else:
            self.logger.warn(f"Goal failed with status: {status}")

        self.goal_completed = True  # Mark goal as completed.
        self.goal_handle_curr = None  # Clear goal handle.

    def goal_response_callback(self, future):
        """
        Callback function executed after the goal is sent to the action server.

        Args:
            future (rclpy.Future): The future that is server's response to goal request.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.logger.warn('Goal rejected :(')
            self.goal_completed = True  # Mark goal as completed (rejected).
            self.goal_handle_curr = None  # Clear goal handle.
        else:
            self.logger.info('Goal accepted :)')
            self.goal_completed = False  # Mark goal as in progress.
            self.goal_handle_curr = goal_handle  # Store goal handle.

            get_result_future = goal_handle.get_result_async()
            get_result_future.add_done_callback(self.goal_result_callback)

    def goal_feedback_callback(self, msg):
        """
        Callback function to receive feedback from the navigation action.

        Args:
            msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
        """
        distance_remaining = msg.feedback.distance_remaining
        number_of_recoveries = msg.feedback.number_of_recoveries
        navigation_time = msg.feedback.navigation_time.sec
        estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

        self.logger.debug(f"Recoveries: {number_of_recoveries}, "
                  f"Navigation time: {navigation_time}s, "
                  f"Distance remaining: {distance_remaining:.2f}, "
                  f"Estimated time remaining: {estimated_time_remaining}s")

        if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
            self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
            self.cancel_current_goal()  # Unblock by discarding the current goal.

    def send_goal_from_world_pose(self, goal_pose):
        """
        Sends a navigation goal to the Nav2 action server.

        Args:
            goal_pose (geometry_msgs.msg.PoseStamped): The goal pose in the world frame.

        Returns:
            bool: True if the goal was successfully sent, False otherwise.
        """
        if self.goal_handle_curr is not None and not self.goal_completed:
            self.get_logger().warn("Previous goal still in progress. Cancelling...")
            self.cancel_current_goal()
            time.sleep(0.5)  # Brief pause for cancellation to complete

        self.goal_completed = False
        goal = NavigateToPose.Goal()
        goal.pose = goal_pose
        self.navigation_goal_pose = goal_pose  # Store for yaw comparison


        if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
            self.get_logger().error('Action server not available!')
            return False

        goal_future = self.action_client.send_goal_async(
            goal, 
            feedback_callback=self.goal_feedback_callback
        )
        goal_future.add_done_callback(self.goal_response_callback)
        return True
    

    def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
        """Helper function to get map origin and resolution."""
        if map_info:
            origin = map_info.origin
            resolution = map_info.resolution
            return resolution, origin.position.x, origin.position.y
        else:
            return None

    def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
                       -> Tuple[float, float]:
        """Converts map coordinates to world coordinates."""
        if map_info:
            resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
            world_x = (map_x + 0.5) * resolution + origin_x
            world_y = (map_y + 0.5) * resolution + origin_y
            return (world_x, world_y)
        else:
            return (0.0, 0.0)

    def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
                       -> Tuple[int, int]:
        """Converts world coordinates to map coordinates."""
        if map_info:
            resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
            map_x = int((world_x - origin_x) / resolution)
            map_y = int((world_y - origin_y) / resolution)
            return (map_x, map_y)
        else:
            return (0, 0)

    def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
        """Helper function to create a Quaternion from a yaw angle."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = sy
        q.w = cy
        return q

    def create_yaw_from_vector(self, dest_x: float, dest_y: float,
                   source_x: float, source_y: float) -> float:
        """Calculates the yaw angle from a source to a destination point.
            NOTE: This function is independent of the type of map used.

            Input: World coordinates for destination and source.
            Output: Angle (in radians) with respect to x-axis.
        """
        delta_x = dest_x - source_x
        delta_y = dest_y - source_y
        yaw = math.atan2(delta_y, delta_x)

        return yaw

    def create_goal_from_world_coord(self, world_x: float, world_y: float,
                     yaw: Optional[float] = None) -> PoseStamped:
        """Creates a goal PoseStamped from world coordinates.
            NOTE: This function is independent of the type of map used.
        """
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = self._frame_id

        goal_pose.pose.position.x = world_x
        goal_pose.pose.position.y = world_y

        if yaw is None and self.pose_curr is not None:
            # Calculate yaw from current position to goal position.
            source_x = self.pose_curr.pose.pose.position.x
            source_y = self.pose_curr.pose.pose.position.y
            yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
        elif yaw is None:
            yaw = 0.0
        else:  # No processing needed; yaw is supplied by the user.
            pass

        goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

        pose = goal_pose.pose.position
        print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
        return goal_pose

    def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
                       yaw: Optional[float] = None) -> PoseStamped:
        """Creates a goal PoseStamped from map coordinates."""
        world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

        return self.create_goal_from_world_coord(world_x, world_y, yaw)


def main(args=None):
    rclpy.init(args=args)

    warehouse_explore = WarehouseExplore()

    if PROGRESS_TABLE_GUI:
        gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
        gui_thread.start()

    rclpy.spin(warehouse_explore)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node objects)
    warehouse_explore.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
