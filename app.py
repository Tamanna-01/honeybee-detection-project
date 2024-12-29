from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images/'
app.config['RESULT_FOLDER'] = 'static/results/'

# Ensure upload and result directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load the YOLO model
model = YOLO("./best.pt")  # Update with the correct path to your model

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_min)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        
        image_file = request.files["image"]
        if image_file.filename == "":
            return redirect(request.url)
        
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Run YOLO model
            results = model.predict(source=image_path, conf=0.25, show=False)
            detections = results[0]
            bee_count = 0
            unique_boxes = []
            annotated_image = cv2.imread(image_path)

            for box in detections.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                is_unique = True
                for unique_box in unique_boxes:
                    if calculate_iou((x1, y1, x2, y2), unique_box) > 0.5:
                        is_unique = False
                        break
                if is_unique:
                    unique_boxes.append((x1, y1, x2, y2))
                    bee_count += 1
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(annotated_image, "honeybee", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            result_path = os.path.join(app.config['RESULT_FOLDER'], f"result_{image_file.filename}")
            cv2.putText(annotated_image, f"Bee Count: {bee_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(result_path, annotated_image)

            return render_template("index.html", bee_count=bee_count, result_image=result_path)

    return render_template("index.html", bee_count=None)

if __name__ == "__main__":
    app.run(debug=True)
