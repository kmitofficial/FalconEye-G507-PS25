🚀 FalconEye – Prompt-Guided Intelligent Tracking Rover

FalconEye is a product-oriented intelligent tracking system designed for real-time, robust object following in dynamic environments.
It enables users to specify a target using clicks, reference images, or natural language, and autonomously tracks and follows the target using a vision–language perception pipeline, distractor-aware tracking, and closed-loop motion control on edge hardware.

The system is optimized for performance, deployability, and practical usability, making it suitable for real-world robotics applications such as:

Human-following robots

Mobile surveillance

Assistive robotics

Smart delivery & service robots

Autonomous companions

🔍 Key Features

Multi-modal target specification

Click-based prompts (SAM)

Reference image prompts (CLIPSeg)

Natural language prompts (CLIPSeg)

Vision–Language Object Segmentation

Open-vocabulary object grounding

No task-specific retraining required

Robust Real-Time Tracking

Distractor-Aware Siamese Tracker (DaSiamRPN)

Handles occlusion, clutter, and appearance changes

Closed-Loop Motion Control

Maintains target centering

Regulates distance automatically

Edge Deployment

Runs on Jetson AGX Xavier

~40 FPS real-time performance

Product-Oriented Design

Focus on stability, responsiveness, and real-world usability

Modular, scalable architecture

🧠 System Overview

FalconEye converts user intent into persistent target tracking using the following pipeline:

User Prompt (Base Station)

Click / Image / Text input

Object Segmentation

SAM for spatial clicks

CLIPSeg for image/text prompts

Bounding Box Initialization

Mask → Bounding box

Visual Tracking

DaSiamRPN maintains target identity

Decision Module (Python)

Computes steering & speed

Motion Control (C++)

Low-latency motor execution

Autonomous Rover Movement

The system ensures the target stays centered in view and at a safe distance during motion.

🏗️ Architecture

FalconEye uses a distributed architecture:

Base Station (Laptop)

User interface

Prompt input (Click / Image / Text)

Target specification

Onboard Rover (Jetson AGX Xavier)

Vision–Language Segmentation

Visual Tracking

Decision Making (Python)

Real-Time Motor Control (C++)

This separation allows:

Smooth user interaction

High-speed onboard processing

Reliable motion execution

⚙️ Hardware & Software Stack
Hardware

Jetson AGX Xavier (32GB)

Web Camera

Differential Drive Rover

Motor Controller

Software

Python (Perception & Tracking)

C++ (Low-level Control)

OpenCV

PyTorch

SAM (Segment Anything)

CLIPSeg

DaSiamRPN

📊 Performance

Real-time tracking: ~40 FPS

Stable under occlusion & clutter

Smooth motion control

Consistent target centering

Reliable distance regulation

The system prioritizes responsiveness and robustness over offline accuracy metrics, making it suitable for real-world deployment.

🎯 Product Focus

FalconEye is designed as a performance-driven product prototype, not just a research demo.

The emphasis is on:

Real-time operation

Edge deployment

System stability

Practical usability

Modular design

Scalability for future features

This makes FalconEye suitable for commercial robotics use-cases where reliability and responsiveness matter more than academic benchmarks.

🚧 Current Capabilities

Prompt-based target selection

Real-time object tracking

Autonomous following

Distance control

Clutter & occlusion handling

🛣️ Future Roadmap

Long-term re-identification

Multi-target tracking

LiDAR-based obstacle avoidance

Voice-based navigation

Multi-camera fusion

Indoor & outdoor navigation

Cloud-based monitoring

📂 Repository Structure
FalconEye-G507-PS25/
│
├── Perception/
│   ├── SAM/
│   ├── CLIPSeg/
│
├── Tracking/
│   └── DaSiamRPN/
│
├── Control/
│   ├── Python_Decision_Module/
│   └── C++_Motor_Controller/
│
├── BaseStation_UI/
│
└── Docs/

📌 Use Cases

Human-following robots

Smart surveillance

Assistive mobility

Campus robots

Service robots

Autonomous companions

📖 References

Segment Anything (SAM) – Kirillov et al.

CLIPSeg – Lüdecke & Ecker

DaSiamRPN – Zhu et al.

👨‍💻 Author

P. Varun Sai
Department of Computer Science & Engineering
Keshav Memorial Institute of Technology
Hyderabad, India

