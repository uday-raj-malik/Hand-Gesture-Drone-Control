# Hand Gesture Controlled Drone System

This project implements a real-time hand gesture recognition system designed to control a drone using only a webcam. The system detects hand movements, extracts landmark features using MediaPipe, and classifies gestures with a trained neural network model. The goal is to build an intuitive, touch-free control interface by combining computer vision and deep learning.

---

## Supported Gestures

The system currently recognizes four gestures:

- **Up** – Increase altitude  
- **Down** – Decrease altitude  
- **Left** – Move left  
- **Right** – Move right  

These gestures can be mapped directly to drone commands.

---

## How It Works

1. MediaPipe detects 21 hand landmarks from each webcam frame.  
2. Landmark coordinates are normalized relative to the wrist to remove position bias.  
3. Additional features such as thumb direction, index finger direction, and relative orientation are computed.  
4. The final feature vector is passed to a trained neural network.  
5. The predicted gesture is displayed in real time with smoothing for stability.

---

## Technologies Used

- Python  
- OpenCV  
- MediaPipe  
- TensorFlow / Keras  
- NumPy, Pandas, Scikit-learn  

---

## Model Training

A custom dataset was collected using webcam recordings with multiple variations such as palm orientation, wrist rotation, distance from camera, and lighting conditions. This improves the robustness of the model in real-world usage.

To retrain the model, run:

python csv_maker.py  
python model_training.py  

This will generate the trained model along with the scaler and label encoder required for inference.

---

## Running the System

Activate your environment and run:

python live_gesture.py

Press ESC to exit the camera window.

---

## Possible Improvements

- Direct integration with drone SDKs (DJI Tello etc.)  
- Additional gestures such as Flip, Hover, or Stop  
- Gesture confirmation delay for safer control  
- TensorFlow Lite deployment for mobile or embedded systems  
- Visualization of prediction confidence  

---

