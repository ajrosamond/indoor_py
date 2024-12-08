# Homework 1 ROS2 Notes


### TO INSTALL PACKAGE FOR ASSIGNMENT 

1. Set up environment variables for ROS. Make sure to replace '/home/rpi/shared' with your own shared folder location
<pre>
source /opt/ros/humble/setup.bash
</pre>
Also do any Windows or Mac specific setup

For example in Mac...
<pre>
export WEBOTS_HOME=/Applications/Webots.app
python3 local_simulation_server.py
</pre>

For example in windows...
<pre>
export WEBOTS_HOME=/mnt/c/Program\ Files/Webots
</pre>

2. Fork your own repository of f23_robotics (using web interface)

3. Clone your fork
<pre>
git clone <your github url for this repository>
</pre>

4. Make the package (for python, it really just installs the files
<pre>
cd f24_robotics
colcon build
</pre>

5. Set up variables to use the package you just created
<pre>
source install/setup.bash
</pre>

6. Start webots simulation with connect back to ROS in the virtual machine
<pre>
ros2 launch webots_ros2_homework1_python f23_robotics_1_launch.py
</pre>
### RUN SAMPLE CONTROLLER

<pre>
ros2 run webots_ros2_homework1_python webots_ros2_homework1_python
</pre>
