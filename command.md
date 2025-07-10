# IMPORTANT THAT PLEASE ENTER COMMAND IN NEW TERMINAL FOR EACH TIME 
## RUN THE MAP CREATED BY LIDAR BY TAKING INPUT FROM 2D ARRAY 
1.  **Open a new terminal and follow the following steps for running b3rb_ros_draw_map:**
    ```bash
    source ~/cognipilot/cranium/install/setup.bash
    ros2 run b3rb_ros_aim_india visualize
    ```

## (OPTIONAL: For debugging with Foxglove.)
2. **Open a new terminal and follow the following steps for building and running Foxglove.**
    ```bash
    cd ~/cognipilot/electrode/
    colcon build
    source ~/cognipilot/electrode/install/setup.bash
    ros2 launch electrode electrode.launch.py sim:=True
    ```
## <span style="background-color: #FFFF00">ADVANCED STEPS FOR FASTER DEVELOPMENT (OPTIONAL)</span>

3. **You may create a bash file (unique for each world) for faster execution. For example for warehouse_1:**
```
cd ~/cognipilot/cranium/
colcon build
source install/setup.bash
ros2 launch b3rb_gz_bringup sil.launch.py world:=nxp_aim_india_2025/warehouse_1 warehouse_id:=1 shelf_count:=2 initial_angle:=135.0 x:=0.0 y:=0.0 yaw:=0.0
```
