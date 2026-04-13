#!/usr/bin/env python3
import math
import subprocess
import time

def main():
    print("Starting Gazebo Oscillating Obstacle Command Loop...")
    start_time = time.time()
    
    # Send continuous velocities to the Gazebo joint velocity controller
    # This bypasses ROS entirely and scripts the Gazebo joint natively!
    try:
        while True:
            t = time.time() - start_time
            # Sine wave oscillating velocity
            vel = 0.2 * math.sin(t * 1.5) 
            
            cmd = f'gz topic -t "/model/dynamic_pillar/joint/x_slider/cmd_vel" -m gz.msgs.Double -p "data: {vel}"'
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            time.sleep(0.05) # 20 Hz
    except KeyboardInterrupt:
        print("\nStopping oscillation.")
        # Halt the obstacle
        cmd = 'gz topic -t "/model/dynamic_pillar/joint/x_slider/cmd_vel" -m gz.msgs.Double -p "data: 0.0"'
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    main()
