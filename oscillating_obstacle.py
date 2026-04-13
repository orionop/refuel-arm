#!/usr/bin/env python3
import math
import subprocess
import time

def main():
    print("Starting Gazebo Oscillating Obstacle Command Loop...")
    start_time = time.time()
    
    try:
        while True:
            t = time.time() - start_time
            # Sine wave oscillating velocity along the Y axis
            vel = 0.3 * math.sin(t * 1.5) 
            
            cmd = f'gz topic -t "/model/dynamic_pillar/cmd_vel" -m gz.msgs.Twist -p "linear: {{x: 0.0, y: {vel}, z: 0.0}}, angular: {{x: 0.0, y: 0.0, z: 0.0}}"'
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            time.sleep(0.05) # 20 Hz
    except KeyboardInterrupt:
        print("\nStopping oscillation.")
        cmd = 'gz topic -t "/model/dynamic_pillar/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0}"'
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    main()
