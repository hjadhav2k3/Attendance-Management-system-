from flask import Flask, render_template, Response, request, redirect
import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

app = Flask(__name__)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def capture_images(name, enrollment):
    camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        print("Could not open camera.")
        return

    if not os.path.exists('TrainingImage'):
        os.makedirs('TrainingImage')

    count = 0
    while count < 30:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"TrainingImage/{name}.{enrollment}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Capturing Images', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    image_paths = [os.path.join('TrainingImage', f) for f in os.listdir('TrainingImage')]
    faces = []
    ids = []
    for image_path in image_paths:
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')
        enrollment_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces.append(image_np)
        ids.append(enrollment_id)
    recognizer.train(faces, np.array(ids))
    recognizer.save('TrainingImageLabel/trainner.yml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        enrollment = request.form['enrollment']

        capture_images(name, enrollment)
        train_model()

        student_data = pd.DataFrame({'Name': [name], 'Enrollment': [enrollment]})
        if not os.path.exists('StudentDetails'):
            os.makedirs('StudentDetails')
        
        if not os.path.exists('StudentDetails/StudentDetail.csv'):
            student_data.to_csv('StudentDetails/StudentDetail.csv', index=False)
        else:
            student_data.to_csv('StudentDetails/StudentDetail.csv', mode='a', header=False, index=False)

        return redirect('/')
    return render_template('register.html')

@app.route('/check_students')
def check_students():
    df = pd.read_csv('StudentDetails/StudentDetail.csv')
    return render_template('check_students.html', students=df)

@app.route('/automatic_attendance', methods=['GET', 'POST'])
def automatic_attendance():
    if request.method == 'POST':
        subject = request.form['subject']
        df_students = pd.read_csv('StudentDetails/StudentDetail.csv')
        df_students.columns = df_students.columns.str.strip()

        if df_students.empty:
            return "No registered students found."

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('TrainingImageLabel/trainner.yml')
        
        camera = cv2.VideoCapture(0)
        attendance_records = []
        current_date = datetime.now().strftime('%Y-%m-%d')

        try:
            run_camera = True
            while run_camera:
                success, frame = camera.read()
                if not success:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    if confidence < 60:
                        name = df_students.loc[df_students['Enrollment'] == id_, 'Name'].values[0]
                        enrollment = id_

                        if not any(record['Enrollment'] == enrollment for record in attendance_records):
                            cv2.putText(frame, f"{name} recognized. Marking attendance...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.imshow('Automatic Attendance', frame)
                            cv2.waitKey(5000) 

                            attendance_records.append({
                                'Enrollment': enrollment,
                                'Name': name,
                                'Subject': subject,
                                'Date': current_date
                            })
                            cv2.putText(frame, f"Attendance marked for {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            run_camera = False

                    else:
                        cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.imshow('Automatic Attendance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    run_camera = False

        finally:
            camera.release()
            cv2.destroyAllWindows()

            if attendance_records:
                attendance_df = pd.DataFrame(attendance_records)
                if not os.path.exists('Attendance'):
                    os.makedirs('Attendance')
                if not os.path.exists('Attendance/SubjectsAttendance.csv'):
                    attendance_df.to_csv('Attendance/SubjectsAttendance.csv', index=False)
                else:
                    attendance_df.to_csv('Attendance/SubjectsAttendance.csv', mode='a', header=False, index=False)

        return redirect('/')

    return render_template('automatic_attendance.html')

if __name__ == '__main__':
    app.run(debug=True)
