# 3D LiDAR SLAM and Localizaion on F1TENTH
ROS2 Packages and requirements for 3D LiDAR Localization and SLAM on the F1TENTH Platform with the Livox MID360 LiDAR.

## Installation

1. Create a new workspace for this project, add a `src` directory and clone this repository into it:
```
mkdir -p ~/f1tenth_ws_3dlidar/src
cd ~/f1tenth_ws_3dlidar/src
git clone git@gitlab.lrz.de:felix-jahncke-students/idp_moritzwagner.git
```
2. Run `./install.sh` from this directory
3. Substitute the LiDAR IP in `livox_ros_driver2/config/MID360_config.json` by changing "XX" to the last two digits of the LiDAR Serial Number, which can be found on a QR Code sticker on the back of the device.
4. Build the workspace in the main directory and source it:
```
cd ..
colcon build
source install/setup.bash
```
5. Rebuild the `f1tenth_stack` from `~/f1tenth_ws`:
```
cd ~/f1tenth_ws
colcon build --packages-select f1tenth_stack
source install/setup.bash
```

## Running the Code
Launch the modified `f1tenth_stack` with
```
cd ~/f1tenth_ws
source install/setup.bash
ros2 launch f1tenth_stack bringup_launch_mid360.py
```

\
**Note: To transfer maps between Mapping and Localization, the car must start in the same position for when the respective nodes are launched. Depending on the environment, there may be some room for error in initial positioning.**
### Mapping
To run the SLAM Mapping Task and generate a Map, run
```
cd ~/f1tenth_ws
source install/setup.bash
ros2 launch lidarslam lidarslam.launch.py
```
from the Workspace root directory. It will save the generated map every 15 seconds to this directory as `map.pcd`. Additionally, the topics `map, modified_map, path, modified_path` and `current_pose` are available in `rviz2` for visualization and further processing. Here, the "modified" topics are generated through additional loop closure. \
Map Saving may also be manually triggered with `ros2 service call /map_save std_srvs/Empty` with or without loop closure. \
The node will overwrite the `map.pcd` file on the next startup, so it must be moved or archived if its data is to be kept for Localization or Viewing. Archiving generated maps in the `~/[3D SLAM WS]/map/` directory may be convenient.

### Localization
To perform Localization, first, copy the map to be used to `~/[3D SLAM WS]/map/map.pcd`, where it will then be loaded from upon node launch:
```
ros2 launch scanmatcher_custom mapping_robot.launch.py 
```
The topics `map, path` and `current_pose` are available in `rviz2` for visualization and further processing.

## Configuration
Config files are available in `lidarslam_ros2/lidarslam/param/lidarslam.yaml` for SLAM and `scanmatcher_custom/param/mapping_robot.yaml` for Localization. For Information about configs see `lidarslam_ros2/README.md`.
