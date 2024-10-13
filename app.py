from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import re
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = tf.keras.models.load_model('vehicle_classification_model.h5')
helmet_model = tf.keras.models.load_model('helmet_detection_model5.h5')

class_names = ['bike', 'car']
class_names_helmet = ['Without Helmet', 'With Helmet']

# ฟังก์ชันสำหรับประมวลผลภาพ
def prepare_image(img, img_width=150, img_height=150):
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def classify_image(img):
    img_array = prepare_image(img)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

def classify_helmet(img):
    img_array = prepare_image(img)
    prediction = helmet_model.predict(img_array)
    predicted_helmet_class = np.argmax(prediction)
    helmet_confidence = np.max(prediction)
    return predicted_helmet_class, helmet_confidence

# Route สำหรับอัปโหลดภาพและประมวลผล
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.form:
        # ดึงข้อมูล Base64 ของภาพ
        image_data = request.form['image']
        # ลบข้อมูลส่วนต้นที่บอกว่าเป็น Base64 ออก (data:image/png;base64,)
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        # แปลง Base64 ให้เป็นไฟล์รูปภาพ
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        img = img.convert('RGB')  # แปลงภาพให้เป็น RGB

        # เรียกการจำแนกรถ
        predicted_class, confidence = classify_image(img)

        if confidence < 0.9:
            result = "This is not a car or bike."
        else:
            if predicted_class == 1:
                result = f"Prediction: Car with confidence {confidence * 100:.2f}%"
            else:
                result = f"Prediction: Bike with confidence {confidence * 100:.2f}%"

                # ถ้าเป็นรถมอเตอร์ไซค์ ให้จำแนกหมวกกันน็อค
                predicted_helmet_class, helmet_confidence = classify_helmet(img)
                if predicted_helmet_class == 1:
                    result += f" | With Helmet: {helmet_confidence * 100:.2f}%"
                else:
                    result += f" | Without Helmet: {helmet_confidence * 100:.2f}%"

        return jsonify({'result': result})
    else:
        return "No image found", 400

# หน้าเว็บหลัก
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()