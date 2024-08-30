import sqlite3

# creates users and predictions tables if they don't exist
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
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id TEXT,
            username TEXT,
            user_id INTEGER,
            emotion TEXT,
            face_image BLOB,  -- Field for storing image data
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    print("Database has been initiated...")
    conn.commit()
    conn.close()
    
# save predictions to the predictions table
def save_prediction(face_id, user_id, username, emotion, image=None):
    conn = sqlite3.connect('emotion_recognition_system_database.db')
    c = conn.cursor()
    c.execute("INSERT INTO predictions (face_id, user_id, username, emotion, image) VALUES (?, ?, ?, ?)", (face_id, user_id, username, emotion, image))
    print("Prediction has been saved...")
    conn.commit()
    conn.close()

# retreives predictions based for a user based on user_id
def get_history(user_id):
    conn = sqlite3.connect('emotion_recognition_system_database.db')
    c = conn.cursor()
    c.execute("SELECT face_id, username, user_id, emotion, face_image, timestamp FROM predictions WHERE user_id = ?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

# create new user in users table 
def create_user(firstname, lastname, email, username, password):
    conn = sqlite3.connect('emotion_recognition_system_database.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (firstname, lastname, email, username, password) VALUES (?, ?, ?, ?, ?)", 
              (firstname, lastname, email, username, password))
    print("User has been created...")
    conn.commit()
    conn.close()
    
# checks if user exists in users table with a given username and password
def authenticate_user(username, password):
    conn = sqlite3.connect('emotion_recognition_system_database.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    print("User has been authenticated...")
    return user

if __name__ == "__main__":
    init_db()