import streamlit as st
import sqlite3
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import os
import requests
from io import BytesIO
import altair as alt
import random
import time  # retry/backoff

# ========= CONFIG =========
API_URL = "https://raflyml-model-meal-mind-ai.hf.space"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)  # ensure data directory exists

# ========= COPY TEXT (ENGLISH ONLY) =========
L = {
    "welcome": "👋 Welcome to MealMind AI! Your friendly smart buddy for a healthier you. Track meals, water, and progress—easily & enjoyably!",
    "login": "Login",
    "register": "Sign Up",
    "username": "Username",
    "password": "Password",
    "login_btn": "Log In",
    "login_success": "Welcome back! Glad to see you again! 🎉",
    "login_fail": "Oops! Username or password is wrong. Try again, okay?",
    "register_btn": "Create Account",
    "register_success": "Account created! Please log in to get started.",
    "register_fail": "That username's already taken. Please try another!",
    "logout": "Log out",
    "profile": "👤 Profile",
    "weight": "Weight (kg)",
    "height": "Height (cm)",
    "age": "Age",
    "gender": "Gender",
    "activity": "Activity Level",
    "goal": "Goal",
    "save_profile": "Save My Profile",
    "profile_saved": "Profile updated! Let’s crush your goals! 👍",
    "water_tracker": "💧 Water Tracker",
    "add_water": "Add a glass (250ml)",
    "today_water": "You've had {0} ml of 2000 ml today!",
    "water_info": "Keep drinking water—hydration is key! 🚰",
    "water_done": "Awesome! Water goal reached! 💙",
    "quick_log": "🍽️ Quick Add (No Photo Needed)",
    "type_food": "Type food name...",
    "grams": "How many grams?",
    "add_to_log": "Add to Log",
    "food_not_found": "Sorry, I couldn't find that food in the database.",
    "food_added": "Logged: {food} ({gram}g) = {cal} kcal",
    "scan_image": "Scan your food (photo)",
    "upload_image": "Upload Image",
    "camera": "Take a Photo",
    "multiple_detected": "Found several foods! Let's see what you have:",
    "single_detected": "Food found! Here’s the result:",
    "ai_guess": "🤖 AI thinks this is: {name}",
    "ai_confidence": "Confidence: {conf:.1f}%",
    "nutrition_info": "🍴 Nutrition Info",
    "no_nutrition": "Sorry, I don't have nutrition info for this food yet.",
    "logged_total": "Today’s total: {total} kcal / {target} kcal",
    "remaining": "Left for today: {remain} kcal",
    "over_limit": "Oh no! You’ve gone over your calorie target today. Tomorrow is a new chance! 💪",
    "under_limit": "You still have room for more healthy food today! Don’t skip meals, okay?",
    "on_track": "Great job! You’re on track today!",
    "logout_success": "You’ve logged out. See you soon! 👋",
    "snack_suggestion": "How about a snack? Try {snack} ({cal} kcal)",
    "motivation": [
        "Every step counts. Keep going! 💪",
        "Small progress is still progress. Consistency matters.",
        "You’re doing amazing today!",
        "Healthy body, happy mind!",
        "Progress, not perfection. You got this!"
    ],
    # Friendlier scan flow
    "analyze_photo": "Analyze Photo",
    "adjust_portion": "Adjust portion (grams)",
    "add_to_diary": "Add to Diary",
    "analysis_running": "Analyzing your photo… models are working",
    "analysis_ready": "Analysis complete",
    "nothing_detected": "Hmm, I couldn't detect any food. Try a closer, brighter photo.",
    "preview": "Preview",
    "logged_ok": "Added to your diary 🎉",
}

# ========= DB & AUTH =========
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
        "Very Active (hard exercise & job)": 1.9,
    }
    tdee = bmr * activity_dict.get(activity_level, 1.2)
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
    return tdee

# ========= NUTRITION DATA =========
@st.cache_data
def load_nutrition_data():
    path = os.path.join(DATA_DIR, 'nutrition_data.csv')
    if not os.path.exists(path):
        st.error("File 'data/nutrition_data.csv' was not found. Please add it to proceed.")
        return pd.DataFrame(columns=["food_name","calories","weight","protein","carbohydrates","fat","fiber","sugar","sodium"])
    df = pd.read_csv(path)
    df = df.rename(columns={"label": "food_name"})
    for col in ['calories','weight','protein','carbohydrates','fat','fiber','sugar','sodium']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if 'food_name' in df:
        df['food_name'] = df['food_name'].astype(str)
    return df

def get_nutrition_info(food_class, nutritional_data):
    if not food_class or nutritional_data is None or nutritional_data.empty:
        return None
    row = nutritional_data[nutritional_data['food_name'].str.lower() == str(food_class).lower()]
    return row.iloc[0] if not row.empty else None

# ========= FASTAPI CLIENT (robust uploads, retries) =========
def _post_image(endpoint, image_bytes, timeout=120, retries=3):
    """
    Send multipart/form-data with field 'file' including filename and MIME.
    Retries with simple backoff and surfaces clear errors to the UI.
    """
    url = API_URL + endpoint
    files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
    last_err = None
    for i in range(retries):
        try:
            res = requests.post(url, files=files, timeout=timeout)
            if res.status_code == 200:
                return res.json()
            else:
                last_err = f"HTTP {res.status_code} - {res.text[:400]}"
        except requests.exceptions.RequestException as e:
            last_err = str(e)
        time.sleep(2 * (i + 1))  # 2s, 4s, 6s
    st.error(f"API call failed: {url} → {last_err}")
    return None

def predict_food_api(image_bytes):
    data = _post_image("/predict/food", image_bytes)
    if not data:
        return None, 0.0
    return data.get("class"), float(data.get("confidence", 0.0))

def predict_fruit_api(image_bytes):
    data = _post_image("/predict/fruit", image_bytes)
    if not data:
        return None, 0.0
    return data.get("class"), float(data.get("confidence", 0.0))

def predict_yolo_api(image_bytes):
    data = _post_image("/predict/yolo", image_bytes)
    if not data:
        return []
    return data.get("results", [])

# ========= APP =========
st.set_page_config(page_title="MealMind AI", page_icon="🧠🍽️", layout="centered")
st.title("🧠🍽️ MealMind AI")
st.markdown(L["welcome"])

# Optional health check (cached 60s)
@st.cache_data(ttl=60)
def check_api_root():
    try:
        r = requests.get(API_URL + "/", timeout=15)
        return r.ok
    except Exception:
        return False
if not check_api_root():
    st.warning("API isn’t responding yet. If the Space just woke up, models may still be loading.")

# ---- LOGIN / REGISTER ----
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.user_id:
    tab1, tab2 = st.tabs([L["login"], L["register"]])
    with tab1:
        st.subheader(L["login"])
        login_username = st.text_input(L["username"], key="loginuser")
        login_password = st.text_input(L["password"], type="password", key="loginpass")
        if st.button(L["login_btn"]):
            user_id = login(login_username, login_password)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.username = login_username
                st.balloons()
                st.success(L["login_success"])
                try:
                    st.toast("Logged in ✅")  # available in newer Streamlit
                except Exception:
                    pass
                st.rerun()
            else:
                st.error(L["login_fail"])
    with tab2:
        st.subheader(L["register"])
        reg_username = st.text_input(L["username"], key="reguser")
        reg_password = st.text_input(L["password"], type="password", key="regpass")
        if st.button(L["register_btn"]):
            if register(reg_username, reg_password):
                st.success(L["register_success"])
            else:
                st.error(L["register_fail"])
    st.stop()

# ---- SIDEBAR: SESSION ----
st.sidebar.markdown(f"👋 Hi, **{st.session_state.username}**!")
if st.sidebar.button(L["logout"]):
    st.session_state.user_id = None
    st.session_state.username = None
    st.success(L["logout_success"])
    st.rerun()

# ---- SIDEBAR: PROFILE ----
profile = get_profile(st.session_state.user_id)
with st.sidebar.form("profile_form"):
    weight = st.number_input(L["weight"], min_value=30, max_value=300, value=int(profile['weight']) if profile else 70)
    height = st.number_input(L["height"], min_value=100, max_value=250, value=int(profile['height']) if profile else 170)
    age = st.number_input(L["age"], min_value=10, max_value=100, value=int(profile['age']) if profile else 25)
    gender = st.selectbox(L["gender"], ["Male", "Female"], index=0 if (profile and profile['gender']=="Male") else 1)
    activity_list = [
        "Sedentary (rarely exercise)", "Light (1-3 days/week)",
        "Moderate (3-5 days/week)", "Active (6-7 days/week)", "Very Active (hard exercise & job)"
    ]
    activity = st.selectbox(L["activity"], activity_list, index=0 if not profile else activity_list.index(profile['activity']))
    goal_list = [
        "Lose weight (deficit 500 kcal)", "Lose fast (deficit 750 kcal)",
        "Maintain", "Gain weight (surplus 300 kcal)"
    ]
    goal = st.selectbox(L["goal"], goal_list, index=0 if not profile else goal_list.index(profile['goal']))
    submit_profile = st.form_submit_button(L["save_profile"])

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
    st.success(L["profile_saved"])
    try:
        st.toast("Profile saved ✅")
    except Exception:
        pass
    st.rerun()

profile = get_profile(st.session_state.user_id)
if profile:
    st.sidebar.info(
        f"TDEE: **{profile['tdee']} kcal**\n\n"
        f"Target: **{profile['daily_limit']} kcal**\n\n"
        f"Goal: **{profile['goal']}**"
    )
else:
    st.sidebar.warning("Complete your profile so MealMind can work best for you!")

# ---- SIDEBAR: WATER ----
st.sidebar.header(L["water_tracker"])
if st.sidebar.button(L["add_water"]):
    add_water(st.session_state.user_id)
water_today = get_water(st.session_state.user_id)
water_target = 8
st.sidebar.progress(min(water_today / water_target, 1.0))
st.sidebar.markdown(L["today_water"].format(water_today*250))
if water_today < water_target:
    st.sidebar.info(L["water_info"])
else:
    st.sidebar.success(L["water_done"])

# ---- QUICK MANUAL LOG ----
with st.expander(L["quick_log"]):
    nutritional_data = load_nutrition_data()
    manual_food = st.text_input(L["type_food"])
    manual_gram = st.number_input(L["grams"], value=100, step=10)
    if st.button(L["add_to_log"]):
        matches = nutritional_data[nutritional_data["food_name"].str.contains(manual_food, case=False, na=False, regex=False)]
        if not matches.empty:
            n = matches.iloc[0]
            base_w = float(n['weight']) if float(n['weight']) > 0 else 100.0
            cal = (manual_gram / base_w) * float(n['calories'])
            st.success(L["food_added"].format(food=n['food_name'].title(), gram=manual_gram, cal=int(cal)))
            add_log(st.session_state.user_id, n['food_name'], cal)
        else:
            st.warning(L["food_not_found"])

# ---- FOOD SCAN — FRIENDLIER FLOW ----
st.markdown("---")
st.subheader("📷 Food Scan")

input_method = st.radio(L["scan_image"], [L["upload_image"], L["camera"]], horizontal=True)
image_data = None
if input_method == L["upload_image"]:
    uploaded_file = st.file_uploader(L["upload_image"], type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_data = uploaded_file
elif input_method == L["camera"]:
    camera_image = st.camera_input(L["camera"])
    if camera_image:
        image_data = camera_image

# Keep adjustable items across reruns
if "pending_items" not in st.session_state:
    st.session_state.pending_items = []  # list of dicts

if image_data:
    image_bytes = image_data.read()
    st.image(Image.open(BytesIO(image_bytes)), caption=L["preview"], use_container_width=True)

    if profile is None:
        st.info("Tip: Save your profile in the sidebar to enable calorie targets and progress insights. Scanning still works without it 👍")

    cols = st.columns([1, 1])
    analyze_clicked = cols[0].button(f"🔎 {L['analyze_photo']}", use_container_width=True)

    if analyze_clicked:
        st.session_state.pending_items = []  # reset
        try:
            status_ctx = st.status(L["analysis_running"], expanded=True)
        except Exception:
            # fallback for older Streamlit
            status_ctx = st.empty()
            status_ctx.write(L["analysis_running"])

        with status_ctx:
            # Try multi-object detection (YOLO) first
            yolo_results = predict_yolo_api(image_bytes)
            nutritional_data = load_nutrition_data()

            if isinstance(yolo_results, list) and len(yolo_results) > 1:
                st.write("Detected multiple foods (YOLO).")
                found_any = False
                for item in yolo_results:
                    food_class = item.get("class")
                    conf = float(item.get("confidence", 0))
                    if not food_class:
                        continue
                    found_any = True
                    nutrition = get_nutrition_info(food_class, nutritional_data)
                    if nutrition is not None:
                        ref_w = float(nutrition["weight"]) if float(nutrition["weight"]) > 0 else 100.0
                        st.session_state.pending_items.append({
                            "name": food_class,
                            "conf": conf,
                            "grams": ref_w,                  # default portion equals ref weight
                            "ref_weight": ref_w,
                            "cal_ref": float(nutrition["calories"]),
                            "nutrition": nutrition.to_dict(),
                        })
                if not found_any:
                    st.session_state.pending_items = []
                    try:
                        status_ctx.update(label=L["analysis_ready"], state="error")
                    except Exception:
                        pass
                    st.error(L["nothing_detected"])
                else:
                    try:
                        status_ctx.update(label=L["analysis_ready"], state="complete")
                    except Exception:
                        pass

            else:
                # Single-object classification: food vs fruit
                food_pred, conf_food = predict_food_api(image_bytes)
                fruit_pred, conf_fruit = predict_fruit_api(image_bytes)

                final_class, final_conf = None, 0.0
                if (conf_fruit or 0) > (conf_food or 0) and (conf_fruit or 0) > 0.5:
                    st.write("Fruit model selected (higher confidence).")
                    final_class, final_conf = fruit_pred, conf_fruit
                else:
                    st.write("Food model selected.")
                    final_class, final_conf = food_pred, conf_food

                if not final_class:
                    try:
                        status_ctx.update(label=L["analysis_ready"], state="error")
                    except Exception:
                        pass
                    st.error("No prediction received from API. Please try again or check connectivity.")
                else:
                    nutrition = get_nutrition_info(final_class, nutritional_data)
                    if nutrition is None:
                        try:
                            status_ctx.update(label=L["analysis_ready"], state="error")
                        except Exception:
                            pass
                        st.warning(f"{L['no_nutrition']} ({final_class})")
                    else:
                        ref_w = float(nutrition["weight"]) if float(nutrition["weight"]) > 0 else 100.0
                        st.session_state.pending_items = [{
                            "name": final_class,
                            "conf": float(final_conf or 0),
                            "grams": ref_w,
                            "ref_weight": ref_w,
                            "cal_ref": float(nutrition["calories"]),
                            "nutrition": nutrition.to_dict(),
                        }]
                        try:
                            status_ctx.update(label=L["analysis_ready"], state="complete")
                        except Exception:
                            pass

    # Render adjustable items + total and Add to Diary
    if st.session_state.pending_items:
        st.markdown("### 🍽️ Detected items")
        total_preview_cals = 0.0
        for i, it in enumerate(st.session_state.pending_items):
            with st.container(border=True):
                c1, c2 = st.columns([2, 1])
                c1.markdown(f"**{it['name']}** · {L['ai_confidence'].format(conf=it['conf']*100)}")
                new_grams = c2.number_input(
                    L["adjust_portion"], min_value=10.0, max_value=1000.0, step=10.0,
                    value=float(it["grams"]), key=f"grams_{i}"
                )
                it["grams"] = float(new_grams)

                ref_w = it["ref_weight"] if it["ref_weight"] > 0 else 100.0
                cal_scaled = (it["grams"] / ref_w) * it["cal_ref"]
                total_preview_cals += cal_scaled

                nut = it["nutrition"]
                c1.markdown(
                    f"- ⚖️ Reference: {nut['weight']} g  \n"
                    f"- 🍔 Calories (ref): {nut['calories']} kcal  \n"
                    f"- 🍗 Protein: {nut['protein']} g · 🍞 Carbs: {nut['carbohydrates']} g · 🧈 Fat: {nut['fat']} g"
                )

        st.markdown(f"### 🧾 This scan (adjusted): **{total_preview_cals:.0f} kcal**")

        if st.button(f"✅ {L['add_to_diary']}", use_container_width=True):
            for it in st.session_state.pending_items:
                ref_w = it["ref_weight"] if it["ref_weight"] > 0 else 100.0
                cal_scaled = (it["grams"] / ref_w) * it["cal_ref"]
                add_log(st.session_state.user_id, it["name"], cal_scaled)
            st.session_state.pending_items = []
            st.success(L["logged_ok"])
            try:
                st.toast("Logged to diary ✅")
            except Exception:
                pass

# ---- TODAY LOG ----
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Today's Log")
df = get_today_logs(st.session_state.user_id)
if not df.empty:
    total_today = float(df["calories"].sum())
    st.sidebar.dataframe(df.tail(10), use_container_width=True)
    cA, cB = st.sidebar.columns(2)
    cA.metric("Today", f"{int(total_today)} kcal")
    cB.metric("Entries", str(len(df)))
else:
    st.sidebar.info("No foods logged yet today. Let's start!")
    total_today = 0.0

# ---- GOAL / PROGRESS ----
if profile:
    try:
        daily_limit = float(profile["daily_limit"])
    except Exception:
        daily_limit = 2000.0

    remaining = max(daily_limit - total_today, 0.0)
    st.sidebar.progress(min(total_today / max(daily_limit, 1.0), 1.0))
    st.sidebar.markdown(L["logged_total"].format(total=int(total_today), target=int(daily_limit)))
    st.sidebar.markdown(L["remaining"].format(remain=int(remaining)))

    if total_today < daily_limit - 400:
        st.sidebar.warning(L["under_limit"])
    elif total_today > daily_limit:
        st.sidebar.error(L["over_limit"])
    else:
        st.sidebar.success(L["on_track"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Weekly Summary")
    week_df = get_weekly_summary(st.session_state.user_id)
    if not week_df.empty:
        week_df['date'] = pd.to_datetime(week_df['date'])
        avg_deficit = daily_limit - week_df["calories"].mean()
        st.sidebar.markdown(f"Average deficit/day: {avg_deficit:.0f} kcal")
        if avg_deficit > 0:
            est_loss = avg_deficit * 7 / 7700
            st.sidebar.markdown(f"Estimated fat loss/week: {est_loss:.2f} kg")
        chart = alt.Chart(week_df).mark_line(point=True).encode(
            x='date:T', y='calories:Q'
        ).properties(title='Daily Calorie Trend (This Week)').interactive()
        st.sidebar.altair_chart(chart, use_container_width=True)

    # Snack suggestion quick-add
    nutritional_data = load_nutrition_data()
    if remaining < 200 and remaining > 0 and nutritional_data is not None and not nutritional_data.empty:
        nutritional_data['calories'] = pd.to_numeric(nutritional_data['calories'], errors='coerce').fillna(0)
        snack_choices = nutritional_data[(nutritional_data['calories'] <= remaining) & (nutritional_data['calories'] > 0)]
        if not snack_choices.empty:
            snack = snack_choices.sample(1).iloc[0]
            if st.sidebar.button(f"Try: {snack['food_name'].title()} ({int(snack['calories'])} kcal)"):
                add_log(st.session_state.user_id, snack['food_name'], float(snack['calories']))
                st.sidebar.success("Added snack to your diary 🎉")

    st.sidebar.markdown("---")
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("⬇️ Download Today's Log (CSV)", csv, "calorie_log_today.csv", "text/csv")
    all_log = get_all_logs(st.session_state.user_id)
    if not all_log.empty:
        csvall = all_log.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("⬇️ Download ALL Log", csvall, "calorie_log_all.csv", "text/csv")
    st.sidebar.markdown("---")
    st.sidebar.success(random.choice(L["motivation"]))
else:
    st.sidebar.info("Complete your profile to activate full tracking and see your progress!")

st.sidebar.markdown("---")
st.sidebar.caption("Made with 💪 and AI for smarter, healthier living.")
