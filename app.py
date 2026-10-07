from flask import Flask, render_template, request, redirect, session, jsonify, url_for
import sqlite3
import os
from werkzeug.utils import secure_filename
from utils.database import create_database
from models.predict import predict_eye
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.secret_key = "netrasenseai_secret_key_2026"
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

create_database()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_history(email, r_type, result, details="", img=""):
    try:
        conn = sqlite3.connect("database.db")
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY AUTOINCREMENT, user_email TEXT, report_type TEXT, result TEXT, details TEXT, image_path TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")

        # --- DUPLICATE FIX ---
        cur.execute("SELECT result FROM reports WHERE user_email=? AND report_type=? ORDER BY id DESC LIMIT 1", (email, r_type))
        last = cur.fetchone()
        if last and last[0] == result:
            print(f"SKIPPED DUPLICATE: {r_type}")
            conn.close()
            return

        cur.execute("INSERT INTO reports (user_email, report_type, result, details, image_path) VALUES (?,?,?,?,?)", (email, r_type, result, details, img))
        conn.commit()
        conn.close()
        print(f"SAVED: {r_type} - {result}")
    except Exception as e:
        print("Save error:", e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fullname = request.form["fullname"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        if len(password) < 6:
            return render_template("register.html", error="Password must be at least 6 chars")
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users(fullname,email,password) VALUES(?,?,?)", (fullname, email, password))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("register.html", error="Email already registered!")
        conn.close()
        return redirect("/login")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user'] = email
            session['fullname'] = user[1]
            return redirect("/dashboard")
        else:
            return render_template("login.html", error="Invalid Email or Password")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if 'user' not in session: return redirect("/login")
    return render_template("dashboard.html")

@app.route("/ai_detection")
def ai_detection():
    if 'user' not in session: return redirect("/login")
    return render_template("ai_detection.html")

@app.route("/upload")
def upload():
    if 'user' not in session: return redirect("/login")
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'user' not in session: return redirect("/login")
    file_key = "image" if "image" in request.files else "eyeimage" if "eyeimage" in request.files else None
    if not file_key: return "No file uploaded"
    img_file = request.files[file_key]
    if img_file.filename == "" or not allowed_file(img_file.filename):
        return "Please select valid JPG/PNG image"
    upload_folder = os.path.join(app.root_path, "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    filename = f"{int(time.time())}_{secure_filename(img_file.filename)}"
    filepath = os.path.join(upload_folder, filename)
    img_file.save(filepath)
    try:
        prediction, confidence, description, symptoms, precautions, consult = predict_eye(filepath)
        try:
            conf_val = float(str(confidence).replace('%','').strip())
        except:
            conf_val = 90.0
        if conf_val < 55:
            prediction = f"Uncertain - Possible {prediction}"
    except Exception as e:
        return f"Prediction error: {str(e)} <br><a href='/upload'>Try Again</a>"

    save_history(session['user'], 'AI Detection', prediction, f"{confidence} | {description}", f"uploads/{filename}")

    image_url = url_for('static', filename=f"uploads/{filename}")
    return render_template("result.html", image_path=f"uploads/{filename}", image_url=image_url, prediction=prediction, confidence=confidence, confidence_value=conf_val, description=description, symptoms=symptoms, precautions=precautions, consult=consult)

@app.route("/symptoms", methods=["GET", "POST"])
def symptoms():
    if 'user' not in session: return redirect("/login")
    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")
        if not selected_symptoms:
            return render_template("symptoms.html", error="Select at least one symptom")
        result = []
        s = selected_symptoms
        if "Blurred Vision" in s: result.append("Cataract")
        if "Redness" in s: result.append("Conjunctivitis")
        if "Eye Pain" in s: result.append("Uveitis / Eye Strain")
        if "Watery Eyes" in s: result.append("Dry Eye / Conjunctivitis")
        if "Sensitivity to Light" in s: result.append("Uveitis / Corneal Issue")
        if "Swollen Eyelids" in s: result.append("Eyelid Disease / Allergy")
        if "Redness" in s and "Eye Pain" in s: result.append("Acute Conjunctivitis / Uveitis")
        if "Redness" in s and "Watery Eyes" in s: result.append("Viral Conjunctivitis")
        result = list(set(result))
        if not result: result.append("General Eye Issue - Consult Doctor")

        save_history(session['user'], 'Symptom Checker', ", ".join(result), "Symptoms: " + ", ".join(s))

        return render_template("symptom_result.html", symptoms=selected_symptoms, result=result)
    return render_template("symptoms.html")

@app.route("/visiontest", methods=["GET", "POST"])
@app.route("/vision", methods=["GET", "POST"])
def visiontest():
    if 'user' not in session: return redirect("/login")
    result = None
    if request.method == "POST":
        test_type = request.form.get("test_type", "color")
        ans = request.form.get("answer", "").strip()
        if test_type == "color":
            if ans == "26":
                session['color'] = "Normal - Color vision intact"
                result = "Normal - Color vision intact"
            else:
                session['color'] = "Color Vision Deficiency - Correct was 26"
                result = "Color Vision Deficiency - Correct was 26"
            save_history(session['user'], 'Vision Test - Color', session['color'], f"color test: {ans}")
        elif test_type == "visual":
            session['visual'] = ans
        elif test_type == "contrast":
            session['contrast'] = ans
    return render_template("vision_test.html", result=result, visual_result=session.get('visual', 'Not Tested'), color_result=session.get('color', 'Not Tested'), contrast_result=session.get('contrast', 'Not Tested'))

@app.route("/save_vision_result", methods=["POST"])
def save_vision_result():
    data = request.get_json()
    if not data: return jsonify({"status":"fail"}), 400
    test = data.get('test')
    value = data.get('value', '').strip()
    if test == 'visual': session['visual'] = value
    elif test == 'color': session['color'] = value
    elif test == 'contrast': session['contrast'] = value
    save_history(session['user'], f'Vision Test - {test.title()}', value, f"{test} test: {value}")
    return jsonify({"status":"ok"})

@app.route("/report")
@app.route("/vision_report")
@app.route("/my_reports")
def report():
    if 'user' not in session: return redirect("/login")
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM reports WHERE user_email=? ORDER BY id DESC", (session['user'],))
        all_reports = cur.fetchall()
    except:
        all_reports = []
    conn.close()
    return render_template("vision_report.html", visual_result=session.get('visual', 'Not Tested'), color_result=session.get('color', 'Not Tested'), contrast_result=session.get('contrast', 'Not Tested'), overall=f"You have {len(all_reports)} total reports", reports=all_reports)

@app.route("/tips")
def tips(): return render_template("tips.html")
@app.route("/about")
def about(): return render_template("about.html")
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)