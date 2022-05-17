# import libraries
import rospy
import subprocess

# create 3 different terminals, running the nessesary commands to prepare the turtlebot for capturing footage
subprocess.call(['gnome-terminal', '-x', 'bash', '-c', 'roscore'])
rospy.sleep(2)
subprocess.call(['gnome-terminal', '-x', 'bash', '-c', 'roslaunch turtlebot_bringup minimal.launch'])
rospy.sleep(2)
subprocess.call(['gnome-terminal', '-x', 'bash', '-c', 'roslaunch turtlebot_bringup 3dsensor.launch'])
