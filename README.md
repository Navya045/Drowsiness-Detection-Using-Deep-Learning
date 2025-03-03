# 🚗💤 **Drowsiness Detection Using Deep Learning**
A real-time **driver drowsiness detection system** using **Convolutional Neural Networks (CNNs)** to reduce road accidents caused by fatigue.

## **Project Overview**
- Developed an **AI-based system** to detect driver drowsiness and prevent accidents.
- Uses **CNN-based transfer learning (MobileNet)** for accurate detection.
- **Real-time alert system** that triggers an alarm when drowsiness is detected.
- Implements **eye closure and yawning detection** for fatigue monitoring.

---

## **Tech Stack**
- **Programming Language**: Python  
- **Deep Learning Frameworks**: TensorFlow, Keras  
- **Computer Vision**: OpenCV, Dlib  
- **Model Architecture**: MobileNet (Transfer Learning)  
- **Libraries**: NumPy, Scipy, Matplotlib, PyGame  

---

## **Model Training**
 **Neural Network Architecture:**  
- **Pre-trained MobileNet Model** (Lightweight CNN)
- Layers:
  - **Convolutional + ReLU Activation**
  - **Max-Pooling**
  - **Flatten Layer**
  - **Fully Connected (Dense) Layers**
  - **Softmax Activation** (Binary classification: Open vs. Closed eyes)
  
 **Training Details:**  
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 32
- **Epochs:** 10
- **Accuracy Achieved:** **95%+ on validation set**

---

## 📂 **Project Structure**
DrowsinessDetector/ │── audio/ # Alarm sound files │── haarcascades/ # Haarcascade models for face and eye detection │── drowsiness_detect.py # Main script for real-time drowsiness detection │── face_and_eye_detector_single_image.py # Detects drowsiness in static images │── face_and_eye_detector_webcam_video.py # Detects drowsiness in real-time video │── shape_predictor_68_face_landmarks.dat # Pre-trained facial landmark detector │── models/ # Trained CNN models │── dataset/ # Training & testing dataset └── README.md # Project Documentation

---

## 📊 **Model Performance & Results**
| **Model**               | **Accuracy** | **Use Case** |
|-------------------------|-------------|-------------|
| **CNN (MobileNet)** ✅  | **95.4%**   | Best Performance |
| **K-Nearest Neighbors (KNN)** | 89.3% | Moderate Accuracy |
| **Random Forest**       | 86.1%       | Requires More Features |
| **Logistic Regression** | 81.2%       | Not Ideal for Image Data |

### 🚨 **Feature Importance**
✔ **Eye Aspect Ratio (EAR)**: Determines eye openness.  
✔ **Mouth Aspect Ratio (MAR)**: Detects yawning behavior.  
✔ **Face Detection**: Haarcascades & Dlib shape predictor.

---

## 🛠️ **How It Works**
1. **Live Camera Feed**: Captures real-time video from the webcam.  
2. **Face & Eye Detection**: Uses Haarcascades to detect facial landmarks.  
3. **Feature Extraction**: Calculates EAR (eye aspect ratio) & MAR (mouth aspect ratio).  
4. **Drowsiness Classification**: CNN model predicts drowsiness based on eye closure & yawning.  
5. **Alert System**: If drowsiness is detected, an **alarm sound & alert message** is triggered.

---

## 📸 **Example Screenshots**
**Drowsiness Detected (Alarm Triggered)**
![d1 alert](https://github.com/user-attachments/assets/677e846e-21e7-4819-a929-0ff8ec133bcd)

**Yawning Detected**
![d2 alert](https://github.com/user-attachments/assets/36435f7c-c551-4e81-a506-05c0b1bb7b86)


**Normal State (No Alert)**
![d3](https://github.com/user-attachments/assets/8d25a08d-7229-4d87-97c2-a22054effb28)

---

## 🔧 **Installation & Setup**
### **Clone Repository**
```bash
git clone https://github.com/Navya045/Drowsiness-Detection-Using-Deep-Learning
cd DrowsinessDetector
```
### **Install Required Libraries**
##To install the necessary dependencies, run the following command:
```bash
pip install opencv-python numpy scipy dlib tensorflow keras pygame gtts imutils
```
## ** Run the Drowsiness Detection Script**
##To start the real-time drowsiness detection system, execute the following command:
```bash

python drowsiness_detect.py
```
## Using the Model for Face/Eye Detection**
##To run the face and eye detection, execute the following command:

```bash
python face_and_eye_detector_webcam_video.py
```
---
## **Results & Performance**
| **Feature**            | **Result** |
|------------------------|-----------|
| **Eye State Accuracy** | 95%+  |
| **Yawning Detection**  | 90%+  |
| **Real-time Speed**    | 30 FPS  |
| **Alert Accuracy**     | High  |

##  **Alerts Generated**
- **Drowsiness Alert:** If **eyes closed** for > 3 seconds 
- **Yawning Alert:** If **mouth open** for extended duration   

##  **Alert Sound & Voice Messages**
- **Buzzer Sound** (Plays when drowsiness is detected)  
- **Text-to-Speech (TTS)** (Voice message alert for yawning)  

---
## **Future Improvements**
### **Potential Enhancements:**
- **Integrating with Vehicle Speed Control**  
  *(Automatically reduce speed if drowsiness is detected)*
- **Cloud-Based Monitoring** 
  *(Send alerts to connected mobile apps for real-time notifications)*
- **Better Face Tracking** 
  *(Use **YOLO** or **MTCNN** for more accurate face detection)*
- **Deploy as an Android App**  
  *(Enable real-time monitoring via smartphone camera for accessibility)*
---
## **Conclusion**
This project successfully implements **real-time drowsiness detection** using **deep learning and computer vision**. The system can **alert drivers before accidents happen**, making it a **valuable tool for road safety**.   

If you like this project, **⭐️ STAR the repository & contribute!**  


