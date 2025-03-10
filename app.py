from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/result'
MODEL_PATH = 'model/best.pt'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


@app.route('/')
@app.route('/home.html')
def home():
    return render_template('home.html')


@app.route('/count.html', methods=['GET', 'POST'])
def count():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run model prediction
        results = model.predict(source=filepath, conf=0.25, show=False)

        # Process detections
        bee_count = 0
        detections = results[0]
        unique_boxes = []
        annotated_image = cv2.imread(filepath)

        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            is_unique = all(calculate_iou((x1, y1, x2, y2), unique_box) < 0.5 for unique_box in unique_boxes)

            if is_unique:
                unique_boxes.append((x1, y1, x2, y2))
                bee_count += 1
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(annotated_image, "honeybee", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add count to image
        # cv2.putText(annotated_image, f"Bee Count: {bee_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save annotated image
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        cv2.imwrite(result_path, annotated_image)

        return render_template('count.html', bee_count=bee_count, image_path=url_for('static', filename=f'result/{filename}'))

    return render_template('count.html')


if __name__ == '__main__':
    app.run(debug=True)
