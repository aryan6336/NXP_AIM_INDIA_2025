# 🏆 NXP AIM India 2025 – Warehouse Treasure Hunt & Object Recognition Challenge

This repository documents our team's participation in the NXP AIM India 2025 B3RB Challenge, specifically the **Warehouse Treasure Hunt & Object Recognition** task.

---
### **Team Details**

**Team Name:** Mobotelligence

**Regional Qualifiers:** Out of over 5,800 participants, only 100 teams advanced to the regional finals—and we're proud that our team made it through!

**Regional Finale:** The regional finale took place at IIIT Delhi. Our team successfully explored all the shelves in the warehouse using our code, but we struggled with time optimization for object and QR scanning. Ultimately, our code's performance in this area prevented us from advancing to the Grand Finale.

---
### 📜 Certificate of Participation

We are proud to have received a certificate of participation for successfully qualifying for the NXP AIM India 2025 regional finals.

![NXP AIM India 2025 Certificate](https://drive.google.com/file/d/1T-Vg-KRIN-MG88ott5SXOP5QVTlNKfv0/view?usp=drivesdk)

--- 
### 🏁 Competition Overview

The competition is structured into three distinct phases.

#### 1. Team Registration & Preparation

Our team registered on the NXP AIM India portal and gained access to the necessary documentation and simulation environments. We received the official NXP_AIM_INDIA_2025 GitHub template and ROS2 packages, which formed the foundation of our project. All development and testing were initially conducted in the Gazebo simulation environment before being ported to hardware.

#### 2. Regional Qualifiers

This phase required us to submit a working ROS2 package that demonstrated our rover's autonomous capabilities. Our solution had to meet the following criteria:

* **Shelf Navigation:** Efficiently navigate shelves using SLAM and the Nav2 stack.
* **QR Code Decoding:** Accurately decode QR codes from the rover's camera feed.
* **Object Recognition:** Recognize objects on the shelves using the provided YOLOv5 (quantized TFLite) model.
* **Data Publishing:** Correctly publish all results to the `/shelf_data` topic.

Our submission was evaluated by the organizers in a Gazebo simulation with hidden warehouse maps. The top-performing teams, based on a combination of score and time efficiency, qualified for the Regional Finale.

---
### 🎯 Challenge Description

Our rover had to autonomously explore the warehouse and score points by performing a series of tasks. The main objectives were:

* **Locate Shelves:** Use the SLAM map (`/map`) and camera vision to find shelves.
* **Navigate:** Use Nav2 to send pose goals and avoid collisions while moving through the environment.
* **Read QR Codes:** Extract the **Shelf ID**, **heuristic angle**, and a **secret string** from QR codes.
* **Recognize Objects:** Use the provided YOLO model to identify objects on the shelves.
* **Publish Data:** Send the results in a `synapse_msgs/WarehouseShelf` message to the `/shelf_data` topic.
* **Reveal Next Shelf:** New shelves were only revealed after correctly publishing the data for the previous one.
* **Repeat:** The process continued until the final shelf, which ended the task.

---
### 🧮 Scoring System

Our final score was determined by a combination of accuracy and efficiency.

* **+0.5 pts:** Per correctly identified object.
* **–1.0 pts:** Per wrongly identified object.
* **+1.0 pts:** For completing an entire shelf (all objects correct).
* **+1.0 pts:** Per correctly decoded QR code.
* **–1.0 pts:** For every collision.
* **+15 pts (percentile bonus):** For time efficiency (if ≥80% objects were correctly identified).
* **+15 pts (percentile bonus):** For object accuracy (if ≥80% objects were correctly identified).
* **+2 pts:** Per training session attended.

**Important Notes:**

* Only the last published message for a given shelf was considered for scoring.
* Publishing a message for the final shelf marked the end of the task.

---
### 🖥️ Software & Hardware Setup

#### Hardware

Our solution was designed for the **NXP MR-B3RB Rover**, which features:

* **Sensors:** LIDAR, IMU, and encoders for SLAM and navigation.
* **Camera:** A front-facing camera for QR code detection and object recognition.
* **Mission Computer:** A NavQPlus computer running **ROS2 Cranium**.

#### Software

Our software stack was built on a foundation of open-source tools.

* **OS:** Ubuntu 22.04 LTS
* **ROS 2:** Humble Hawksbill
* **Simulation:** Gazebo (for simulating warehouse environments)
* **Autonomy:** Nav2 + SLAM Toolbox
* **Vision:** YOLOv5 (quantized TFLite) + OpenCV (for QR decoding)
* **Framework:** CogniPilot (AIRY release)

**Key ROS2 Packages:**

* `b3rb_ros_aim_india`: Our core ROS2 logic package.
* `synapse_msgs`: Custom message definitions for the challenge.
* `dream_world`: Contains the Gazebo simulation worlds.
* `b3rb`: Rover models and configuration files.

---
### 🛠️ Development Workflow

Our development process was broken down into a series of interconnected modules.

#### SLAM & Navigation

We used **SLAM Toolbox** to generate a map from LIDAR data and published it to the `/map` topic. Our navigation stack, **Nav2**, used this map along with the `/global_costmap` for localization. We sent pose goals via the `MapsToPose` action to direct the rover. Our strategy involved implementing a frontier-based exploration algorithm to efficiently find and navigate to the shelves.

#### QR Code Decoding

We subscribed to the `/camera/image_raw/compressed` topic and used a combination of **pyzbar** and **OpenCV** to extract QR codes. Once decoded, we parsed the string to get the **Shelf ID**, **heuristic angle**, and **secret string**.

#### Object Recognition

We ran the provided quantized YOLOv5 model (`yolov5n-int8.tflite`). Our node subscribed to the `/camera/image_raw/compressed` topic, processed the image, and published the detected objects to `/shelf_objects`.

#### Data Publishing

After successfully navigating to a shelf, decoding the QR, and recognizing the objects, we constructed a `WarehouseShelf` message. This message was then published to the `/shelf_data` topic with the correct **Shelf ID**, **QR string**, and a list of detected objects and their counts.

#### Curtain Logic

The simulation's logic for revealing new shelves was based on our successful data publishing. A new shelf's curtain would only drop after we correctly published the data for the previous one, forcing a sequential, treasure-hunt-style approach.

---
