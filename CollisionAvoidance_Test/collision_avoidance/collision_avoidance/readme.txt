Jeonbuk National University 2023. 

This software was tested in Ubuntu 20.04 with ROS2 Galactic. 
It allows obstacle avoidance using depth image as input and it returns body linear velocities .

1-Start airsim simulator with PX4 SITL
Download th weight file from here : https://drive.google.com/file/d/1x13WFbRMzZ1D-XjIf8ofr7VtM1gwHWrV/view?usp=sharing
2- Run the ROS_2_pUB_depth.py   to extract and publish the depth image
3- Perform Inferance using the ONNX file by running the script obstacle_avoidance_ros2.py
