"""
Interactive keyboard control for the SALP robot environment.

This script allows you to manually control the robot's motion using keyboard input.
You can adjust the inhale control, nozzle steering, and coast time in real-time.

Run this script and follow the on-screen instructions to control the robot.
"""

import sys
import os
import numpy as np
import pygame

# Add project root to path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle


def main():
    """Run the interactive robot control demo."""
    
    # Initialize Pygame
    pygame.init()
    
    # Create robot with the calibrated "real robot" parameters (same as test_robot.py),
    # so manual runs are directly comparable to the trained SAC/PPO/RPPO agents.
    # These are just the defaults — they can be tweaked live via the in-app param panel.
    nozzle = Nozzle(
        length1=0.052, length2=0.038, length3=0.050,
        area=np.pi * 0.01 ** 2, mass=0.428, radius=0.1, inner_radius=0.022
    )
    nozzle.set_angles(angle1=0.0, angle2=0.0)  # init rotation matrices before Robot() uses them
    robot = Robot(
        dry_mass=0.738,
        init_length=0.26,
        init_width=0.135,
        max_contraction=0.04,
        nozzle=nozzle
    )
    robot.set_environment(density=1000)  # water density in kg/m^3
    robot.enable_history_recording()  # capture full intra-cycle animation for the GIF
    
    # Create environment with rendering enabled
    env = SalpRobotEnv(render_mode="human", width=900, height=700, robot=robot)
    obs, info = env.reset()
    
    print("\n" + "="*70)
    print(" SALP ROBOT INTERACTIVE KEYBOARD CONTROL")
    print("="*70)
    print("\nStarting interactive control session...\n")
    
    try:
        # Run interactive control
        env.interactive_control(max_cycles=None)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Safety net: save the session GIF if interactive_control exited abnormally
        # (normal exits already saved it). No-op if nothing was recorded.
        if env.is_recording():
            env.stop_recording()
        env.close()
        pygame.quit()
        print("Cleaned up resources")


if __name__ == "__main__":
    main()
