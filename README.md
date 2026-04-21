# TrafficVision-AI

Smart Traffic Violation Detection System  
Final Year Capstone + Research Project


# Overview

TrafficVision-AI is an AI-powered web-based system for detecting motorcycle helmet violations from traffic videos.

The system uses YOLO-based object detection combined with rule-based rider–vehicle association logic to detect helmet violations and generate evidence from traffic footage.

This project is developed as:

• Final Year Capstone Project  
• Research Project in Computer Vision & Robotics  


# Features

- Upload traffic video
- Motorcycle detection
- Rider detection
- Helmet violation detection
- Rider–vehicle association
- Processed video output
- Evidence generation
- Clean web interface


# System Methodology

Input Video  
↓  
YOLO Object Detection  
↓  
Motorcycle Detection  
↓  
Rider Association  
↓  
Helmet Detection  
↓  
Violation Identification  
↓  
Evidence Generation  
↓  
Web Display



# Technologies Used

- Python
- Flask
- YOLOv8 (Ultralytics)
- OpenCV
- HTML/CSS
- Machine Learning

---

#  Project Structure

app.py → Main Flask Application

ml/
processor.py → Detection pipeline

templates/ → HTML files

static/
css/ → Styling
images/ → Assets

docs/
screenshots/ → Project screenshots

# Installation

Clone repository: https://github.com/Sayem2935/TrafficVision-AI.git

Install dependencies: pip install -r requirements.txt

Run project: python app.py

Open browser: http://127.0.0.1:5000



# Research Contribution

- Custom traffic dataset collection
- Helmet vs No-Helmet classification
- Rider–vehicle spatial association
- Evaluation of heuristic vs ML-based detection
- Real-world traffic analysis (Bangladesh context)

---

# Future Work

- Custom trained helmet detection model
- License plate recognition integration
- Face extraction module
- Real-time camera integration
- Edge device deployment

---

# System Screenshots

# Home Page
![Home](docs/screenshots/home.png)

# Upload Page
![Upload](docs/screenshots/upload.png)

# Results Page
![Results](docs/screenshots/result.png)

---

# Author

Mohammad Sayem Uddin 
Software Engineering Student  
Robotics Major  

