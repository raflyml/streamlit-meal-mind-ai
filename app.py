import streamlit as st
import sqlite3
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import os
import requests
from ultralytics import YOLO
from io import BytesIO
import altair as alt
import random
import cv2

# ========= SETUP API URL =========
API_URL = "https://model-meat-mind-ai-production.up.railway.app"

# ========= DATABASE & AUTH =========

DATA_DIR = "data"
MODEL_DIR = "models"

def get_conn():
    db_path = os.path.join(DATA_DIR, "calorie_tracker.db")
    return sqlite3.connect(db_path, check_same_thread=False)

def create_tables():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS profiles (
        user_id INTEGER PRIMARY KEY,
        weight REAL,
        height REAL,
        age INTEGER,
        gender TEXT,
        activity TEXT,
        goal TEXT,
        tdee REAL,
        daily_limit REAL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        time TEXT,
        food TEXT,
        calories REAL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS water_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT,
        glasses INTEGER DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    conn.commit()
    conn.close()
create_tables()

def hash_pass(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register(username, password):
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?,?)", (username, hash_pass(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login(username, password):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, hash_pass(password)))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def save_profile(user_id, profile):
    conn = get_conn()
    c = conn.cursor()
    c.execute("REPLACE INTO profiles (user_id, weight, height, age, gender, activity, goal, tdee, daily_limit) VALUES (?,?,?,?,?,?,?,?,?)",
        (user_id, float(profile['weight']), float(profile['height']), int(profile['age']),
         profile['gender'], profile['activity'], profile['goal'],
         float(profile['tdee']), float(profile['daily_limit'])))
    conn.commit()
    conn.close()

def get_profile(user_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT weight, height, age, gender, activity, goal, tdee, daily_limit FROM profiles WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        keys = ['weight','height','age','gender','activity','goal','tdee','daily_limit']
        return dict(zip(keys, row))
    else:
        return None

def add_log(user_id, food, calories):
    try:
        calories = float(calories)
    except Exception:
        calories = 0
    conn = get_conn()
    c = conn.cursor()
    now = datetime.now()
    c.execute("INSERT INTO logs (user_id, date, time, food, calories) VALUES (?,?,?,?,?)",
              (user_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M"), food, calories))
    conn.commit()
    conn.close()

def get_today_logs(user_id):
    conn = get_conn()
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT time, food, calories FROM logs WHERE user_id=? AND date=? ORDER BY id DESC", (user_id, today))
    rows = c.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["time","food","calories"])
    if not df.empty:
        df["calories"] = pd.to_numeric(df["calories"], errors="coerce").fillna(0)
    return df

def get_all_logs(user_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT date, time, food, calories FROM logs WHERE user_id=? ORDER BY date DESC, time DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["date","time","food","calories"])
    if not df.empty:
        df["calories"] = pd.to_numeric(df["calories"], errors="coerce").fillna(0)
    return df

def get_weekly_summary(user_id):
    conn = get_conn()
    c = conn.cursor()
    today = datetime.now()
    last_week = (today - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    c.execute("""
        SELECT date, SUM(calories) FROM logs 
        WHERE user_id=? 
        AND date BETWEEN ? AND ?
        GROUP BY date
        ORDER BY date ASC
    """, (user_id, last_week, today_str))
    rows = c.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["date","calories"])
    if not df.empty:
        df["calories"] = pd.to_numeric(df["calories"], errors="coerce").fillna(0)
    return df

def add_water(user_id):
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT glasses FROM water_logs WHERE user_id=? AND date=?", (user_id, today))
    row = c.fetchone()
    if row:
        glasses = int(row[0]) + 1
        c.execute("UPDATE water_logs SET glasses=? WHERE user_id=? AND date=?", (glasses, user_id, today))
    else:
        glasses = 1
        c.execute("INSERT INTO water_logs (user_id, date, glasses) VALUES (?,?,?)", (user_id, today, glasses))
    conn.commit()
    conn.close()
    return glasses

def get_water(user_id):
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT glasses FROM water_logs WHERE user_id=? AND date=?", (user_id, today))
    row = c.fetchone()
    conn.close()
    return int(row[0]) if row else 0

def calculate_tdee(weight, height, age, gender, activity_level):
    weight = float(weight)
    height = float(height)
    age = int(age)
    if gender == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    activity_dict = {
        "Sedentary (rarely exercise)": 1.2,
        "Light (1-3 days/week)": 1.375,
        "Moderate (3-5 days/week)": 1.55,
        "Active (6-7 days/week)": 1.725,
        "Very Active (hard exercise & job)": 1.9
    }
    tdee = bmr * activity_dict[activity_level]
    return round(tdee)

def calculate_deficit_limit(tdee, goal):
    if goal == "Lose weight (deficit 500 kcal)":
        return tdee - 500
    elif goal == "Lose fast (deficit 750 kcal)":
        return tdee - 750
    elif goal == "Maintain":
        return tdee
    elif goal == "Gain weight (surplus 300 kcal)":
        return tdee + 300
    else:
        return tdee

# ========= FOOD CLASS NAMES & NUTRITION =========

@st.cache_data
def load_nutrition_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'nutrition_data.csv'))
    df = df.rename(columns={"label": "food_name"})
    for col in ['calories','weight','protein','carbohydrates','fat','fiber','sugar','sodium']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def get_nutrition_info(food_class, nutritional_data):
    row = nutritional_data[nutritional_data['food_name'].str.lower() == str(food_class).lower()]
    return row.iloc[0] if not row.empty else None

# ========= PREDICT API FASTAPI =========

def predict_food_api(image_bytes):
    files = {'file': image_bytes}
    try:
        res = requests.post(API_URL + "/predict/food", files=files, timeout=30)
        if res.status_code == 200:
            result = res.json()
            return result["class"], result["confidence"]
        else:
            return None, 0
    except Exception as e:
        print("Error Food API:", e)
        return None, 0

def predict_fruit_api(image_bytes):
    files = {'file': image_bytes}
    try:
        res = requests.post(API_URL + "/predict/fruit", files=files, timeout=30)
        if res.status_code == 200:
            result = res.json()
            return result["class"], result["confidence"]
        else:
            return None, 0
    except Exception as e:
        print("Error Fruit API:", e)
        return None, 0

def predict_yolo(image_bytes, yolo_model):
    img_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    results = yolo_model(img)
    return results, img

# ========= STREAMLIT MAIN PAGE =========

st.set_page_config(page_title="MealMind AI", page_icon="🧠🍽️", layout="centered")
st.title('🧠🍽️ MealMind AI')
st.markdown("""
Your smart assistant for tracking meals, calories, and water intake.  
Snap your food, log manually, and see your nutrition progress — all powered by AI!
""")

# === LOGIN/REGISTER ===
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.user_id:
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="loginuser")
        login_password = st.text_input("Password", type="password", key="loginpass")
        if st.button("Login"):
            user_id = login(login_username, login_password)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.username = login_username
                st.balloons()
                st.toast("Login successful, welcome!", icon="👋")
                st.rerun()
            else:
                st.error("Login failed! Please check your username or password.")
    with tab2:
        st.subheader("Register")
        reg_username = st.text_input("Username", key="reguser")
        reg_password = st.text_input("Password", type="password", key="regpass")
        if st.button("Register"):
            if register(reg_username, reg_password):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username already taken!")
    st.stop()

st.sidebar.markdown(f"Hello, **{st.session_state.username}**!")
if st.sidebar.button("Logout"):
    st.session_state.user_id = None
    st.session_state.username = None
    st.rerun()

# ========== PROFILE FORM ==========
profile = get_profile(st.session_state.user_id)
with st.sidebar.form("profile_form"):
    weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=int(profile['weight']) if profile else 70)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=int(profile['height']) if profile else 170)
    age = st.number_input("Age", min_value=10, max_value=100, value=int(profile['age']) if profile else 25)
    gender = st.selectbox("Gender", ["Male", "Female"], index=0 if (profile and profile['gender']=="Male") else 1)
    activity = st.selectbox("Activity Level", [
        "Sedentary (rarely exercise)", "Light (1-3 days/week)",
        "Moderate (3-5 days/week)", "Active (6-7 days/week)", "Very Active (hard exercise & job)"
    ], index=0 if not profile else [
        "Sedentary (rarely exercise)", "Light (1-3 days/week)",
        "Moderate (3-5 days/week)", "Active (6-7 days/week)", "Very Active (hard exercise & job)"
    ].index(profile['activity']))
    goal = st.selectbox("Goal", [
        "Lose weight (deficit 500 kcal)", "Lose fast (deficit 750 kcal)",
        "Maintain", "Gain weight (surplus 300 kcal)"
    ], index=0 if not profile else [
        "Lose weight (deficit 500 kcal)", "Lose fast (deficit 750 kcal)",
        "Maintain", "Gain weight (surplus 300 kcal)"
    ].index(profile['goal']))
    submit_profile = st.form_submit_button("Save Profile")

if submit_profile:
    tdee = calculate_tdee(weight, height, age, gender, activity)
    daily_limit = calculate_deficit_limit(tdee, goal)
    save_profile(st.session_state.user_id, {
        "weight": weight,
        "height": height,
        "age": age,
        "gender": gender,
        "activity": activity,
        "goal": goal,
        "tdee": tdee,
        "daily_limit": daily_limit
    })
    st.success("Profile saved!")
    st.rerun()

profile = get_profile(st.session_state.user_id)
if profile:
    st.sidebar.info(
        f"TDEE: **{profile['tdee']} kcal**\n\n"
        f"Target: **{profile['daily_limit']} kcal**\n\n"
        f"Goal: **{profile['goal']}**"
    )
else:
    st.sidebar.warning("Please complete your profile first!")

# ========== WATER TRACKER ==========
st.sidebar.header("💧 Water Tracker")
if st.sidebar.button("Add 1 Glass (250ml)"):
    add_water(st.session_state.user_id)
water_today = get_water(st.session_state.user_id)
water_target = 8
st.sidebar.progress(min(water_today / water_target, 1.0))
st.sidebar.markdown(f"**Today:** {water_today*250} ml / 2000 ml")
if water_today < water_target:
    st.sidebar.info("Drinking enough water supports your health and energy!")
else:
    st.sidebar.success("Water goal achieved, well done!")

# ========== FAST MANUAL LOG ==========
with st.expander("🍽️ Quick Food Logging (No Image Scan)"):
    nutritional_data = load_nutrition_data()
    manual_food = st.text_input("Type a food (e.g.: fried rice)")
    manual_gram = st.number_input("How many grams?", value=100, step=10)
    if st.button("Add to Log (Manual)"):
        matches = nutritional_data[nutritional_data["food_name"].str.contains(manual_food.lower(), na=False)]
        if not matches.empty:
            n = matches.iloc[0]
            cal = (manual_gram / float(n['weight'])) * float(n['calories'])
            st.success(f"Added: {n['food_name'].title()} ({manual_gram}g) = {cal:.0f} kcal")
            add_log(st.session_state.user_id, n['food_name'], cal)
        else:
            st.warning("Food not found in the database!")

# ========== AI FOOD SCAN ==========
@st.cache_resource
def load_yolo_model():
    return YOLO(os.path.join(MODEL_DIR, 'indo_yolo.pt'))

yolo_model = load_yolo_model()

input_method = st.radio("📸 Choose image input method:", ["📁 Upload File", "📷 Use Camera"])
image_data = None
if input_method == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_data = uploaded_file
elif input_method == "📷 Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image_data = camera_image

if image_data and profile:
    st.markdown("### 🔍 Scanning your food image...")
    image_bytes = image_data.read()
    nutritional_data = load_nutrition_data()
    results, img_cv = predict_yolo(image_bytes, yolo_model)
    result = results[0]
    boxes = result.boxes

    total_calories = 0
    food_logged = []

    if len(boxes) > 1:
        st.markdown("🔁 **Multiple foods detected. Using AI detection.**")
        st.image(img_cv, caption='Detected Multiple Foods', use_container_width=True)
        st.markdown("### 🍽️ Detected Foods:")
        for box in boxes:
            class_id = int(box.cls[0])
            food_class = result.names[class_id]
            nutrition = get_nutrition_info(food_class, nutritional_data)
            if nutrition is not None:
                st.markdown(f"- **{food_class}**")
                st.markdown(f"  - ⚖️ Weight: {nutrition['weight']} g")
                st.markdown(f"  - 🍔 Calories: {nutrition['calories']} kcal")
                st.markdown(f"  - 🍗 Protein: {nutrition['protein']} g")
                st.markdown(f"  - 🍞 Carbs: {nutrition['carbohydrates']} g")
                st.markdown(f"  - 🧈 Fat: {nutrition['fat']} g")
                st.markdown(f"  - 🌾 Fiber: {nutrition['fiber']} g")
                st.markdown(f"  - 🍬 Sugar: {nutrition['sugar']} g")
                st.markdown(f"  - 🧂 Sodium: {nutrition['sodium']} mg")
                total_calories += float(nutrition['calories'])
                food_logged.append((food_class, float(nutrition['calories'])))
            else:
                st.warning(f"Nutritional info not found for {food_class}")
    else:
        st.markdown("🔁 **Single food detected. Predicting with API...**")
        food_pred, conf_food = predict_food_api(image_bytes)
        fruit_pred, conf_fruit = predict_fruit_api(image_bytes)
        if conf_fruit > conf_food and conf_fruit > 0.5:
            st.info("🍏 The fruit model is more confident. Using fruit model prediction.")
            final_class = fruit_pred
            final_conf = conf_fruit
        else:
            final_class = food_pred
            final_conf = conf_food

        nutrition = get_nutrition_info(final_class, nutritional_data)
        st.image(Image.open(BytesIO(image_bytes)), caption='Your Food Image', use_container_width=True)
        st.markdown(f"### 🧠 Predicted Food: *{final_class}*")
        st.markdown(f"✅ Confidence: *{final_conf*100:.2f}%*")

        if nutrition is not None:
            st.markdown("### 🍴 Nutritional Information:")
            st.markdown(f"- ⚖️ Weight: {nutrition['weight']} g")
            st.markdown(f"- 🍔 Calories: {nutrition['calories']} kcal")
            st.markdown(f"- 🍗 Protein: {nutrition['protein']} g")
            st.markdown(f"- 🍞 Carbs: {nutrition['carbohydrates']} g")
            st.markdown(f"- 🧈 Fat: {nutrition['fat']} g")
            st.markdown(f"- 🌾 Fiber: {nutrition['fiber']} g")
            st.markdown(f"- 🍬 Sugar: {nutrition['sugar']} g")
            st.markdown(f"- 🧂 Sodium: {nutrition['sodium']} mg")
            total_calories += float(nutrition['calories'])
            food_logged.append((final_class, float(nutrition['calories'])))
        else:
            st.warning("Nutritional info not found for this food.")

    # === Log all detected foods automatically ===
    for food, cal in food_logged:
        add_log(st.session_state.user_id, food, cal)

    st.markdown(f"## 🧾 Total Estimated Calories This Log: **{total_calories:.0f} kcal**")

# ========== TODAY LOG TABLE ==========
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Daily Calorie Log")
df = get_today_logs(st.session_state.user_id)
if not df.empty:
    total_today = df["calories"].sum()
    try:
        total_today = float(total_today)
    except Exception:
        total_today = 0
    st.sidebar.dataframe(df.tail(10), use_container_width=True)
else:
    st.sidebar.info("No foods logged today yet.")
    total_today = 0

if profile:
    daily_limit = profile["daily_limit"]
    try:
        daily_limit = float(daily_limit)
    except Exception:
        daily_limit = 2000

    if daily_limit > 0:
        st.sidebar.progress(min(total_today / daily_limit, 1.0))
        remaining = daily_limit - total_today
    else:
        st.sidebar.progress(0)
        remaining = 0
    st.sidebar.markdown(f"**Today's total:** {total_today:.0f} kcal / {daily_limit:.0f} kcal")
    st.sidebar.markdown(f"**Remaining today:** {remaining:.0f} kcal")
    if total_today < daily_limit - 400:
        st.sidebar.warning("⚠️ Too much deficit! Consider eating a bit more for a healthy balance.")
    elif total_today > daily_limit:
        st.sidebar.error("⚠️ You have exceeded your daily target.")
    else:
        st.sidebar.success("✅ You are on track for your calorie goal!")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Weekly Summary")
    week_df = get_weekly_summary(st.session_state.user_id)
    if not week_df.empty:
        week_df['date'] = pd.to_datetime(week_df['date'])
        avg_deficit = daily_limit - week_df["calories"].mean()
        st.sidebar.markdown(f"**Average deficit/day this week:** {avg_deficit:.0f} kcal")
        if avg_deficit > 0:
            est_loss = avg_deficit * 7 / 7700
            st.sidebar.markdown(f"**Estimated fat loss/week:** {est_loss:.2f} kg")
        chart = alt.Chart(week_df).mark_line(point=True).encode(
            x='date:T', y='calories:Q'
        ).properties(title='Daily Calorie Trend (This Week)').interactive()
        st.sidebar.altair_chart(chart, use_container_width=True)
    else:
        avg_deficit = 0

    nutritional_data = load_nutrition_data()
    if remaining < 200 and remaining > 0:
        nutritional_data['calories'] = pd.to_numeric(nutritional_data['calories'], errors='coerce').fillna(0)
        snack_choices = nutritional_data[(nutritional_data['calories'] <= remaining) & (nutritional_data['calories'] > 0)]
        if not snack_choices.empty:
            snack = snack_choices.sample(1).iloc[0]
            st.sidebar.info(f"Snack Suggestion: {snack['food_name'].title()} ({snack['calories']} kcal)")

    st.sidebar.markdown("---")
    # Export CSV (today)
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("⬇️ Download Today's Log (CSV)", csv, "calorie_log_today.csv", "text/csv")
    # Export CSV (all log)
    all_log = get_all_logs(st.session_state.user_id)
    if not all_log.empty:
        csvall = all_log.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("⬇️ Download ALL Log", csvall, "calorie_log_all.csv", "text/csv")
    # Motivation
    quotes = [
        "Every healthy choice today brings you closer to your goals.",
        "Small progress is still progress!",
        "Consistency is the key to real results.",
        "Eat well, drink enough, move more.",
        "Don't give up—change takes time.",
        "Big changes come from small healthy habits."
    ]
    st.sidebar.markdown("---")
    st.sidebar.success(random.choice(quotes))
else:
    st.sidebar.info("Complete your profile to activate full calorie tracking and summary.")

st.sidebar.markdown("---")
st.sidebar.caption("Made with 💪 and AI for smarter, healthier living.")
