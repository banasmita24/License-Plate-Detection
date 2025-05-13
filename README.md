# License Plate Detection using YOLOv8 and EasyOCR

This project implements automatic license plate detection and recognition using a custom-trained YOLOv8 model for object detection and EasyOCR for optical character recognition (OCR). It can accurately detect license plates from images or video frames and extract the license number.

## 📌 Project Structure

The system is divided into two main components:

1. **License Plate Detection**  
   - Utilizes a YOLOv8 model trained on a custom dataset to detect license plates.
2. **Text Recognition**  
   - Uses EasyOCR to read text from the detected license plate region.

## 🧰 Tech Stack

- **YOLOv8 (Ultralytics)** – Real-time object detection
- **EasyOCR** – Lightweight and accurate OCR
- **Python** – Backend logic and scripting
- **OpenCV** – Image preprocessing and visualization

## 🧠 Model Training

- Used CCPD (Chinese Car Parking Dataset) to train the model.
- Trained a YOLOv8 model on this dataset for detection.
- Evaluated the model based on accuracy and mAP (mean Average Precision).

## 🚀 Features

- Detects license plates in static images or videos
- Extracts license numbers using OCR
- Draws bounding boxes on detected plates

## 📂 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/license-plate-detection.git
cd license-plate-detection
