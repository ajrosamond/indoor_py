import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped  
from nav_msgs.msg import OccupancyGrid  
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import AprilTagDetectionArray
from rclpy.qos import ReliabilityPolicy, QoSProfile
import math
import random
import numpy as np  
import time
import csv
import os


LINEAR_VEL = 0.25
STOP_DISTANCE = 0.2
LIDAR_ERROR = 0.05
SAFE_STOP_DISTANCE = STOP_DISTANCE + LIDAR_ERROR
RIGHT_SIDE_INDEX = 270
RIGHT_FRONT_INDEX = 210
LEFT_FRONT_INDEX = 150
LEFT_SIDE_INDEX = 90
RECOVERY_TIMEOUT = 5
MAP_SIZE = 100  
MAP_RESOLUTION = 0.05  

class ApartmentExplorer(Node):

    def __init__(self):
        super().__init__('apartment_explorer')
        self.scan_cleaned = []
        self.turtlebot_moving = True
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscriber1 = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback1,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.subscriber2 = self.create_subscription(
            PointStamped,
            '/TurtleBot3Burger/gps',  
            self.gps_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', 10)  #Map publisher error?
        self.laser_forward = 0
        self.gps_data = None
        self.start_position = None
        self.distance_from_start = 0.0
        self.recovery_timer = 0  
        self.gps_path = []  
        self.map = np.zeros((MAP_SIZE, MAP_SIZE))  
        self.last_spin_time = time.time()  # Initialize the time for the last spin
        self.spin_interval = 4
        timer_period = 0.5
        self.cmd = Twist()
        self.timer = self.create_timer(timer_period, self.control_loop)
        self.reverse_mode = False
        self.corner_detection_count = 0  #Counter to track corners do I need both counters?
        self.csv_file_path = 'gps_data.csv'
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Time', 'X', 'Y']) 


    def listener_callback1(self, msg1):
        scan = msg1.ranges
        self.scan_cleaned = []
        for reading in scan:
            if reading == float('Inf'):
                self.scan_cleaned.append(3.5)
            elif math.isnan(reading):
                self.scan_cleaned.append(0.0)
            else:
                self.scan_cleaned.append(reading)
        self.update_map()

    def gps_callback(self, msg3):
        x, y = msg3.point.x, msg3.point.y
        self.gps_data = (x, y)
        self.get_logger().info(f"GPS data received: {self.gps_data}")

        self.gps_path.append(self.gps_data)

        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.get_clock().now().to_msg().sec, x, y])  #Time is useless


        if self.start_position is None:
            self.start_position = self.gps_data
        else:
            self.distance_from_start = self.calculate_distance(self.start_position, self.gps_data)
        
        self.update_robot_position()


    def calculate_distance(self, start, current):
        return math.sqrt((current[0] - start[0])**2 + (current[1] - start[1])**2)

    def update_robot_position(self):
        if self.gps_data is None:
            return

        x, y = self.gps_data

        # Convert GPS coordinates to map indices
        map_x = int((x - (-MAP_SIZE * MAP_RESOLUTION / 2)) / MAP_RESOLUTION)
        map_y = int((y - (-MAP_SIZE * MAP_RESOLUTION / 2)) / MAP_RESOLUTION)

        # Update map without penalizing revisits
        if 0 <= map_x < MAP_SIZE and 0 <= map_y < MAP_SIZE:
            self.map[map_x, map_y] = 1  # Mark the position as visited
            self.get_logger().info(f"Visited: ({map_x}, {map_y})")



    def update_map(self):
        if not self.gps_data:
            return  
        
        map_x = int((self.gps_data[0] / MAP_RESOLUTION) + MAP_SIZE // 2)
        map_y = int((self.gps_data[1] / MAP_RESOLUTION) + MAP_SIZE // 2)

        for angle in range(len(self.scan_cleaned)):
            distance = self.scan_cleaned[angle]
            if distance < 3.5: 
               
                obstacle_x = map_x + int(distance * math.cos(math.radians(angle)) / MAP_RESOLUTION)
                obstacle_y = map_y + int(distance * math.sin(math.radians(angle)) / MAP_RESOLUTION)

                #Update the map with the obstacle position
                if 0 <= obstacle_x < MAP_SIZE and 0 <= obstacle_y < MAP_SIZE:
                    self.map[obstacle_x, obstacle_y] = 100  #Mark obstacle in the map

        #Producing publish error?
        self.publish_map()

    def publish_map(self):
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header.frame_id = "map"
        occupancy_grid.info.resolution = MAP_RESOLUTION
        occupancy_grid.info.width = MAP_SIZE
        occupancy_grid.info.height = MAP_SIZE
        occupancy_grid.info.origin.position.x = -MAP_SIZE * MAP_RESOLUTION / 2.0
        occupancy_grid.info.origin.position.y = -MAP_SIZE * MAP_RESOLUTION / 2.0
        occupancy_grid.info.origin.position.z = 0.0
        occupancy_grid.info.origin.orientation.w = 1.0

        normalized_map = np.zeros_like(self.map, dtype=np.int8)

        #Fill the normalized map with appropriate values
        normalized_map[self.map == 0] = -1    
        normalized_map[self.map == 1] = 100   

        occupancy_grid.data = normalized_map.flatten().tolist()

        #Publish context error?
        self.map_publisher.publish(occupancy_grid)

    def control_loop(self):
        if not self.turtlebot_moving:
            return

        if len(self.scan_cleaned) == 0:
            return

        # LIDAR readings
        left_lidar_min = min(self.scan_cleaned[LEFT_SIDE_INDEX:LEFT_FRONT_INDEX])
        right_lidar_min = min(self.scan_cleaned[RIGHT_FRONT_INDEX:RIGHT_SIDE_INDEX])
        front_lidar_min = min(self.scan_cleaned[LEFT_FRONT_INDEX:RIGHT_FRONT_INDEX])

        forward_threshold = 1.0
        side_threshold = 0.3
        escape_threshold = 0.5

        if front_lidar_min > forward_threshold:
            self.get_logger().info("Open path ahead. Moving forward.")
            self.cmd.linear.x = LINEAR_VEL
            self.cmd.angular.z = 0.0
            self.publisher_.publish(self.cmd)
            return

        if front_lidar_min < SAFE_STOP_DISTANCE:
            self.get_logger().info("Obstacle ahead. Adjusting course.")
            self.cmd.linear.x = 0.0
            if right_lidar_min > left_lidar_min:
                self.cmd.angular.z = -0.5
            else:
                self.cmd.angular.z = 0.5  
            self.publisher_.publish(self.cmd)
            return

        if left_lidar_min < side_threshold and right_lidar_min < side_threshold:
            self.get_logger().info("Walls on both sides. Moving cautiously forward.")
            self.cmd.linear.x = 0.1
            self.cmd.angular.z = 0.0
        elif left_lidar_min < side_threshold:
            self.get_logger().info("Following wall on the left.")
            self.cmd.linear.x = 0.1
            self.cmd.angular.z = -0.3 
        elif right_lidar_min < side_threshold:
            self.get_logger().info("Following wall on the right.")
            self.cmd.linear.x = 0.1
            self.cmd.angular.z = 0.3  
        else:
            self.get_logger().info("Open space. Exploring randomly.")
            self.cmd.linear.x = 0.15
            self.cmd.angular.z = random.choice([0.2, -0.2])

        self.publisher_.publish(self.cmd)
        
def main(args=None):
    rclpy.init(args=args)
    explorer_node = ApartmentExplorer()
    try:
        rclpy.spin(explorer_node)
    except KeyboardInterrupt:
        #Stop the simulation and print the GPS path when interrupted
        explorer_node.get_logger().info("Simulation stopped. Printing GPS path.")
        #explorer_node.print_gps_path()
    finally:
        explorer_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()  #Shutdown ROS only after printing the GPS path could remove

if __name__ == '__main__':
    main()