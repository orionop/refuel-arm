#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import math
import time

class ObstacleOscillator(Node):
    def __init__(self):
        super().__init__('obstacle_oscillator')
        
        # Topic for the Gazebo Joint Position Controller (bridged from ROS2)
        self.publisher_ = self.create_publisher(
            Float64, 
            '/model/dynamic_cylinder/joint/oscillation_joint/0/cmd_pos', 
            10)
        
        self.timer_period = 0.05  # 20Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.start_time = time.time()
        self.amplitude = 0.35   # Meters (total swing 0.7m)
        self.frequency = 0.15   # Hz (slow oscillation)
        
        self.get_logger().info(f"Oscillator started: Amp={self.amplitude}m, Freq={self.frequency}Hz")

    def timer_callback(self):
        elapsed = time.time() - self.start_time
        # Simple sine wave: pos = A * sin(2 * pi * f * t)
        pos = self.amplitude * math.sin(2 * math.pi * self.frequency * elapsed)
        
        msg = Float64()
        msg.data = pos
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleOscillator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
