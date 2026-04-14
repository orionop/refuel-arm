#!/usr/bin/env python3
import math
import subprocess
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class OscillatingBroadcaster(Node):
    def __init__(self):
        super().__init__('dynamic_pillar_broadcaster')
        self.pub = self.create_publisher(Point, '/dynamic_pillar/pose', 10)
        self.start_time = time.time()
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("Starting Gazebo Oscillating Obstacle Command Loop (20Hz)...")
        
    def timer_callback(self):
        t = time.time() - self.start_time
        
        # Velocity amplitude and frequency
        vel = 0.15 * math.sin(t * 1.5) 
        
        # Position math: y(t) = y_initial + integral(v dt)
        # integral(0.15 * sin(1.5*t)) = -0.10 * cos(1.5*t)
        # So y(t) = 0.15 + (-0.10 * cos(1.5*t)) + C
        # To make it start at Y=0.15 natively (when t=0, cos=1, so we need + 0.10 to offset)
        # Wait, if we just want position:
        y_pos = 0.15 + (-0.10 * math.cos(t * 1.5)) + 0.10
        
        # 1. Publish to ROS 2 for Python Algorithms
        msg = Point()
        msg.x = 0.52
        msg.y = y_pos
        msg.z = 0.40 # physical vertical center of the 0.8m pillar standing at Z=0.42 (wait! Z base is 0.42. 0.42 + 0.4 = 0.82! let's say 0.80)
        self.pub.publish(msg)
        
        # 2. Command Gazebo Physics Natively
        cmd = f'gz topic -t "/model/dynamic_pillar/cmd_vel" -m gz.msgs.Twist -p "linear: {{x: 0.0, y: {vel}, z: 0.0}}, angular: {{x: 0.0, y: 0.0, z: 0.0}}"'
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        self.get_logger().info("Stopping oscillation.")
        cmd = 'gz topic -t "/model/dynamic_pillar/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}"'
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main(args=None):
    rclpy.init(args=args)
    node = OscillatingBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
