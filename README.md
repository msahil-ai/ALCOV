
# 👁️ ALCOV: Real-Time Spatial Tracking & Footfall Analytics

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

ALCOV is a specialized computer vision pipeline engineered to resolve the complex challenge of spatial tracking, unique individual identification, and real-time footfall counting in dynamic environments. 

By converting raw video feeds into structured physical-digital analytics, this system provides actionable intelligence for spatial management, crowd control, and resource allocation.

## 🏢 Production Validation: Alcove Malls Deployment
> **Status:** Active in Production  
> An optimized iteration of the ALCOV algorithm is currently deployed at **Alcove Malls**. It serves as the core computer vision engine processing live surveillance feeds to monitor real-time footfall traffic, spatial density, and operational analytics at scale.

---

## 🏗️ System Architecture & Core Modules

The repository is modularized to separate detection, tracking, and application logic, allowing for scalable deployments across different hardware configurations (edge devices, local servers, or cloud).

* 👤 **`people_count.py` (Core Engine):** The primary module resolving the critical issue of tracking and accurately counting unique individuals passing through defined spatial thresholds.
* 🎥 **`live_camera_feed.py`:** Interfaces the tracking algorithm directly with real-time hardware feeds (CCTV/Webcams) for continuous monitoring.
* 🏃 **`track_people.py`:** Handles the spatial tracking assignment algorithms. Assigns persistent unique IDs to moving instances across frames to eliminate double-counting during occlusions.
* 📦 **`multi_object.py` & `object_count.py`:** Extends the neural network's capabilities beyond human tracking to identify and tally multiple object classes simultaneously.
* 🖼️ **`simple_obj_detection.py`:** A streamlined inference script optimized for static image analysis (includes test sample `jr.jpg`).

---

## ⚙️ How It Works (Pipeline Flow)

1. **Preprocessing:** Ingests raw frames from live hardware or pre-recorded video (`video2.mp4`) and normalizes them for inference.
2. **Detection:** Deploys object detection models to generate bounding boxes and confidence scores for human entities in the frame.
3. **Tracking & ID Assignment:** Utilizes centroid tracking kinematics to predict entity trajectory, assigning a persistent ID to ensure an individual is tracked seamlessly even in dense crowds.
4. **Analytics Trigger:** Monitors vector intersections against a defined virtual Region of Interest (ROI). When a tracked centroid crosses the threshold, the counting logic updates the central analytics state.

---

## 🚀 Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/msahil-ai/ALCOV.git](https://github.com/msahil-ai/ALCOV.git)
cd ALCOV

```

**2. Configure the Environment**
It is highly recommended to use a virtual environment.

```bash
# Install required dependencies (OpenCV, NumPy, etc.)
pip install -r requirements.txt

```

## 💻 Execution Guide

Run the modules based on your deployment requirement:

**Primary Footfall Analytics (Video Stream):**

```bash
python people_count.py

```

**Live Edge / CCTV Feed Processing:**

```bash
python live_camera_feed.py

```

**Multi-Class Object Tracking:**

```bash
python multi_object.py

```

---

*Developed by [Md Sahil*](https://www.linkedin.com/in/mdsahil1/)


