from flask import Flask, render_template, Response, redirect, url_for, request, jsonify, flash, session
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import sqlite3
import datetime
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import base64
import os
from database_config import init_db, authenticate_user, create_user


app = Flask(__name__)

app.secret_key = os.urandom(24)

model_path = 'model/model.h5'
face_cascade_path = 'haar_cascade_classifier/haarcascade_frontalface_default.xml'

model = load_model(model_path)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# initializes database
def init_db():
    conn = sqlite3.connect('emotion_recognition_system_database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            firstname TEXT, 
            lastname TEXT,
            email TEXT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY, 
                face_id TEXT,
                username TEXT, 
                user_id INTEGER,
                emotion TEXT, 
                face_image BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# def clear_database():
#     try:
#         # connect to the SQLite database
#         conn = sqlite3.connect('emotion_recognition_system_database.db')
#         c = conn.cursor()
#         # delete all rows from the predictions table
#         c.execute("DELETE FROM predictions")
#         # commit the transaction to apply changes
#         conn.commit()
#         print("Database cleared successfully!")
#     except sqlite3.Error as e:
#         # handle any errors that occur during the operation
#         print(f"An error occurred: {e}")
#     finally:
#         # ensure the cursor and connection are closed properly
#         if c:
#             c.close()
#         if conn:
#             conn.close()

# clear_database()

init_db()

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user[0]
            session['username'] = username
            # flash('Login successful!', 'success')
            return redirect(url_for('index', login_success=True))
        else:
            # flash('Invalid username or password', 'danger')
            # return render_template('login.html')
            return redirect(url_for('login', login_failed=True))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            return redirect(url_for('signup', failed_password_match=True))
        
        if not firstname or not lastname or not email or not username or not password:
            return redirect(url_for('signup', fields_required=True))
        
        try:
            create_user(firstname, lastname, email, username, password)
            return redirect(url_for('login', signup_success=True))
        except sqlite3.IntegrityError:
            return redirect(url_for('signup', username_exists=True))
    return render_template('signup.html')

# webcam feed route
@app.route('/video_feed')
def video_feed():
    username = session.get('username')
    
    if not username:
        return redirect(url_for('login', username_required=True))
    
    def gen_frames():
        cap = cv2.VideoCapture(0)
        last_emotions = {}
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (800, 470))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
            
            for (x, y, w, h) in faces:
                face_id = f"{x}_{y}_{w}_{h}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                standardized_face = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = standardized_face.astype('float') / 255.0
                roi = np.expand_dims(img_to_array(roi), axis=0)
                prediction = model.predict(roi)[0]
                label = emotion_labels[np.argmax(prediction)]
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if face_id not in last_emotions or last_emotions[face_id] != label:
                    last_emotions[face_id] = label
                    face_image = frame[y:y+h, x:x+w]
                    save_emotion(face_id, label, face_image, username)
                    
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# save emotions
def save_emotion(face_id, emotion, face_image, username):
    conn = sqlite3.connect('emotion_recognition_system_database.db')
    c = conn.cursor()
    _, buffer = cv2.imencode('.jpg', face_image)
    face_image_binary = buffer.tobytes()
    
    # username = session.get('username')
    
    if username:
        c.execute("INSERT INTO predictions (username, face_id, emotion, face_image) VALUES (?, ?, ?, ?)", 
                (username, face_id, emotion, face_image_binary)) 
        conn.commit()
    else:
        print("No user is currently logged in.")
    conn.close()

@app.route('/history')
def show_history():
    
    username = session.get('username')
    
    conn = sqlite3.connect('emotion_recognition_system_database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM predictions WHERE username=? ORDER BY timestamp DESC LIMIT 10", (username,))
    rows = c.fetchall()
    conn.close()

    history_data = []
    for row in rows:
        id, face_id, username, user_id, emotion, face_image_data, timestamp = row
        timestamp_dt = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        formatted_timestamp = timestamp_dt.strftime('%B %-d, %Y - %-I:%M %p')
        if face_image_data:
            face_image = base64.b64encode(face_image_data).decode('utf-8')
            history_data.append({
                'face_id': face_id,
                'emotion': emotion,
                'timestamp': formatted_timestamp,
                'face_image': face_image
            })
    return render_template('history.html', history=history_data)

# other routes for distribution, emotion over time, emotion by user, prediction, etc.
@app.route('/plot_emotion_distribution')
def plot_emotion_distribution():
    conn = sqlite3.connect('emotion_recognition_system_database.db')
    c = conn.cursor()
    c.execute("SELECT emotion, COUNT(*) FROM predictions GROUP BY emotion")
    data = c.fetchall()
    conn.close()
    
    emotions = [row[0] for row in data]
    counts = [row[1] for row in data]
    
    # create a Plotly pie chart
    fig = px.pie(names=emotions, values=counts, title="Emotion Distribution")
    
    # convert Plotly figure to HTML
    plot_html = pio.to_html(fig, full_html=False)
    
    return render_template('plot.html', plot_html=plot_html, title='Emotion Distribution')

@app.route('/plot_emotion_over_time')
def plot_emotion_over_time():
    conn = sqlite3.connect('emotion_recognition_system_database.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, emotion FROM predictions")
    data = c.fetchall()
    conn.close()
    
    df = pd.DataFrame(data, columns=['timestamp', 'emotion'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['count'] = 1
    
    df_grouped = df.groupby(['timestamp', 'emotion']).count().reset_index()
    
    fig = px.line(df_grouped, x='timestamp', y='count', color='emotion', title="emotion over time", labels={'count' : 'count'})
        
    # convert Plotly figure to HTML
    plot_html = pio.to_html(fig, full_html=False)
    
    return render_template('plot.html', plot_html=plot_html, title='emotions over time')

@app.route('/plot_emotion_by_user')
def plot_emotion_by_user():
    conn = sqlite3.connect('emotion_recognition_system_database.db')
    c = conn.cursor()
    c.execute("SELECT username, emotion, COUNT(*) FROM predictions GROUP BY username, emotion")
    data = c.fetchall()
    conn.close()
    
    df = pd.DataFrame(data, columns=['username', 'emotion', 'count'])
    df = df.pivot(index='username', columns='emotion', values='count').fillna(0)
    
    # Create a Plotly stacked bar chart
    fig = px.bar(df, x=df.index, y=df.columns, title="Emotion by User", labels={'value': 'Count'})
    
    # Convert Plotly figure to HTML
    plot_html = pio.to_html(fig, full_html=False)
    
    return render_template('plot.html', plot_html=plot_html, title='Emotions by User')

@app.route('/predict_emotion_levels')
def predict_emotion_levels():
    conn = sqlite3.connect('emotion_recognition_system_database.db')
    query = "SELECT timestamp, emotion FROM predictions"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert timestamp to datetime and extract the hour
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    
    emotions_of_interest = ['angry', 'disgust', 'fear', 'sad', 'surprise']
    filtered_df = df[df['emotion'].isin(emotions_of_interest)]
    
    if filtered_df.empty:
        return "No records found for the specified emotions."
    
    # Aggregate counts by hour and emotion
    emotion_counts = filtered_df.groupby(['hour', 'emotion']).size().unstack(fill_value=0)

    # Initialize the plotly figure
    fig = go.Figure()

    # Loop through each emotion of interest
    for emotion in emotions_of_interest:
        if emotion in emotion_counts:
            emotion_data = emotion_counts[emotion].reset_index(name='count')
            fig.add_trace(go.Scatter(x=emotion_data['hour'], y=emotion_data['count'], mode='lines+markers', name=f'Observed {emotion.capitalize()} Counts'))

            # Perform regression analysis
            X = emotion_data[['hour']]
            y = emotion_data['count']
            model = LinearRegression()
            model.fit(X, y)

            # Predict counts for each hour
            hours = np.arange(0, 24).reshape(-1, 1)
            predictions = model.predict(hours)

            # Add regression line to the plot
            fig.add_trace(go.Scatter(x=hours.flatten(), y=predictions, mode='lines', name=f'Predicted {emotion.capitalize()} Counts', line=dict(dash='dash')))

    # Customize the layout
    fig.update_layout(
        title='Regression Analysis: Predicted Emotion Levels by Hour',
        xaxis_title='hour of day',
        yaxis_title='counts',
        legend_title='emotions',
        template='plotly_white'
    )

    # Convert Plotly figure to HTML
    plot_html = pio.to_html(fig, full_html=False)
    
    return render_template('plot.html', plot_html=plot_html, title='Emotion Level Predictions')

@app.route('/logout')
def log_out():
    session.pop('user_id', None)
    return redirect(url_for('login', logout_success=True))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)