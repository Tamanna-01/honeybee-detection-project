---

# **Honeybee Detection and Counting Using YOLOv8**  

## **📌 Project Overview**  
This project aims to **automate the detection and counting of honeybees** in images using **deep learning-based object detection models**. The system employs **YOLOv8**, a state-of-the-art model optimized for small object detection, achieving an accuracy of **94.7%**.  

To make the model accessible and user-friendly, a **web-based application** has been developed using **Flask, HTML, and CSS**, allowing users to upload images and receive real-time honeybee count predictions.  

## **🚀 Features**  
✔️ Automated detection and counting of honeybees in images  
✔️ Trained YOLOv8 model with **94.7% accuracy**  
✔️ User-friendly **web interface** for image upload and result visualization  
✔️ Preprocessing techniques applied for **better detection accuracy**  
✔️ Model comparison with **YOLOv11, YOLOv8 and Faster R-CNN**  
✔️ Supports **future extensions** like real-time video processing and multi-species pollinator detection  

---

## **📂 Project Structure**  
```
├── dataset/                # Contains annotated dataset used for training  
├── models/                 # Saved trained models (YOLOv8s, YOLOv11, Faster R-CNN)  
├── static/                 # Static files (CSS, images, JavaScript for frontend)  
│   ├── css/                # Stylesheets for web interface  
│   ├── images/             # Placeholder images for UI  
├── templates/              # HTML files for web interface  
│   ├── index.html          # Main page for image upload  
│   ├── result.html         # Display results with honeybee count  
├── app.py                  # Flask backend for handling requests  
├── requirements.txt        # List of dependencies  
├── README.md               # Project documentation  
└── LICENSE                 # License file  
```

---

## **🛠️ Installation & Setup**  
Follow these steps to set up and run the project locally.  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/Tamanna-01/honeybee-detection-project.git
cd honeybee-detection-project
```

### **2️⃣ Install Dependencies**  
```bash
pip install flask ultralytics opencv-python numpy matplotlib seaborn
```

### **3️⃣ Run the Web Application**  
```bash
python app.py
```
This will start the Flask server. Open **`http://127.0.0.1:5000/`** in your browser to use the web interface.

---

## **🖼️ How to Use the Web App**  
1. Open the web application in your browser.  
2. Upload an image containing honeybees.  
3. Select the detection model (default: **YOLOv8s**).  
4. Click "Submit" to process the image.  
5. The model will return:  
   - A processed image with **bounding boxes** around detected honeybees.  
   - A text output showing the **total count of detected honeybees**.  

---

## **🧪 Model Comparison & Results**  
Multiple models were trained and compared for accuracy:  

| Model           | Accuracy  | Remarks |
|----------------|-----------|---------|
| **YOLOv8s**    | **94.7%** | Selected for final deployment |
| YOLOv11        | 63.02%    | Performed well but lower accuracy |
| Faster R-CNN   | 51.9%     | Less efficient for small objects |

The final YOLOv8s model was **trained on 8,233 images**, with various preprocessing techniques applied to enhance accuracy.

---

## **🤝 Contributors**  
This project was developed by the following team members:  
- **Tamanna Shaw**  
- **Ayusha Rani Rowlo**  
- **Surya Teja Pallapu**  
- **Poorvika Marali**  

---

## **🛠️ Technologies Used**  
- **Python** (Flask, OpenCV, NumPy, Matplotlib, Seaborn)  
- **Deep Learning Frameworks** (YOLOv8, Faster RCNN, YOLOv11)  
- **Frontend Technologies** (HTML, CSS, JavaScript)  
- **Data Annotation Tools** (RoboFlow, LabelImg)  
- **Version Control** (Git, GitHub)  

---

## **📜 License**  
This project is licensed under the **MIT License**.  
You are free to modify, distribute, and use this project for both commercial and non-commercial purposes.  

---



**Star ⭐ the repository if you found this useful!** 🚀  

---
