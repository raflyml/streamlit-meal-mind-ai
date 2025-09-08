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
import time

# ========= CONFIG =========
API_URL = "https://raflyml-model-meal-mind-ai.hf.space"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CONF_THRESHOLD = 0.4  # 40% confidence threshold

# ========= UI STRINGS =========
L = {
    "welcome": "👋 Welcome to MealMind AI! Track meals, water, and progress—easily & enjoyably!",
    "login": "Login",
    "register": "Sign Up",
    "username": "Username",
    "password": "Password",
    "login_btn": "Log In",
    "login_success": "Welcome back! 🎉",
    "login_fail": "Oops! Username or password is wrong.",
    "register_btn": "Create Account",
    "register_success": "Account created! Please log in.",
    "register_fail": "That username's already taken.",
    "logout": "Log out",
    "profile": "👤 Profile",
    "weight": "Weight (kg)",
    "height": "Height (cm)",
    "age": "Age",
    "gender": "Gender",
    "activity": "Activity Level",
    "goal": "Goal",
    "save_profile": "Save My Profile",
    "profile_saved": "Profile updated! 👍",
    "water_tracker": "💧 Water Tracker",
    "add_water": "Add a glass (250ml)",
    "today_water": "You've had {0} ml of 2000 ml today!",
    "water_info": "Keep drinking water—hydration is key! 🚰",
    "water_done": "Awesome! Water goal reached! 💙",
    "quick_log": "🍽️ Quick Add (No Photo Needed)",
    "type_food": "Type food name...",
    "grams": "How many grams?",
    "add_to_log": "Add to Log",
    "food_not_found": "Food not found in the database.",
    "food_added": "Logged: {food} ({gram}g) = {cal} kcal",
    "scan_image": "Scan your food (photo)",
    "upload_image": "Upload Image",
    "camera": "Take a Photo",
    "multiple_detected": "Found several foods:",
    "single_detected": "Food found:",
    "ai_guess": "🤖 AI thinks this is: {name}",
    "ai_confidence": "Confidence: {conf:.1f}%",
    "nutrition_info": "🍴 Nutrition Info",
    "no_nutrition": "No nutrition info available.",
    "logged_total": "Today’s total: {total} kcal / {target} kcal",
    "remaining": "Remaining today: {remain} kcal",
    "over_limit": "Over your calorie target today.",
    "under_limit": "You still have room for more food today.",
    "on_track": "Great job! You’re on track today!",
    "logout_success": "You’ve logged out. 👋",
    "snack_suggestion": "How about a snack? Try {snack} ({cal} kcal)",
    "motivation": [
        "Every step counts. Keep going! 💪",
        "Small progress is still progress.",
        "You’re doing amazing today!",
        "Healthy body, happy mind!",
        "Progress, not perfection. You got this!"
    ],
    "analyze_photo": "Analyze Photo",
    "adjust_portion": "Adjust portion (grams)",
    "add_to_diary": "Add to Diary",
    "analysis_running": "Analyzing your photo…",
    "analysis_ready": "Analysis complete",
    "nothing_detected": "🚫 This doesn’t look like food. Try again with a clear food photo.",
    "preview": "Preview",
    "logged_ok": "Added to your diary 🎉",
}

# ========= DATABASE =========
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

def hash_pass(password): return hashlib.sha256(password.encode()).hexdigest()
def register(u,p):
    conn=get_conn(); c=conn.cursor()
    try: c.execute("INSERT INTO users (username,password) VALUES (?,?)",(u,hash_pass(p))); conn.commit(); return True
    except sqlite3.IntegrityError: return False
    finally: conn.close()
def login(u,p):
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?",(u,hash_pass(p))); row=c.fetchone(); conn.close()
    return row[0] if row else None
def save_profile(uid,p):
    conn=get_conn(); c=conn.cursor()
    c.execute("REPLACE INTO profiles (user_id, weight, height, age, gender, activity, goal, tdee, daily_limit) VALUES (?,?,?,?,?,?,?,?,?)",
              (uid,p['weight'],p['height'],p['age'],p['gender'],p['activity'],p['goal'],p['tdee'],p['daily_limit']))
    conn.commit(); conn.close()
def get_profile(uid):
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT weight,height,age,gender,activity,goal,tdee,daily_limit FROM profiles WHERE user_id=?",(uid,))
    row=c.fetchone(); conn.close()
    return dict(zip(['weight','height','age','gender','activity','goal','tdee','daily_limit'],row)) if row else None
def add_log(uid,food,cal):
    conn=get_conn(); c=conn.cursor(); now=datetime.now()
    c.execute("INSERT INTO logs (user_id,date,time,food,calories) VALUES (?,?,?,?,?)",
              (uid,now.strftime("%Y-%m-%d"),now.strftime("%H:%M"),food,float(cal))); conn.commit(); conn.close()
def get_today_logs(uid):
    conn=get_conn(); c=conn.cursor(); today=datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT time,food,calories FROM logs WHERE user_id=? AND date=? ORDER BY id DESC",(uid,today))
    df=pd.DataFrame(c.fetchall(),columns=["time","food","calories"]); conn.close()
    if not df.empty: df["calories"]=pd.to_numeric(df["calories"],errors="coerce").fillna(0)
    return df
def get_all_logs(uid):
    conn=get_conn(); c=conn.cursor()
    c.execute("SELECT date,time,food,calories FROM logs WHERE user_id=? ORDER BY date DESC,time DESC",(uid,))
    df=pd.DataFrame(c.fetchall(),columns=["date","time","food","calories"]); conn.close()
    if not df.empty: df["calories"]=pd.to_numeric(df["calories"],errors="coerce").fillna(0)
    return df
def get_weekly_summary(uid):
    conn=get_conn(); c=conn.cursor()
    today=datetime.now(); last_week=(today-pd.Timedelta(days=6)).strftime("%Y-%m-%d"); today_str=today.strftime("%Y-%m-%d")
    c.execute("SELECT date,SUM(calories) FROM logs WHERE user_id=? AND date BETWEEN ? AND ? GROUP BY date ORDER BY date ASC",(uid,last_week,today_str))
    df=pd.DataFrame(c.fetchall(),columns=["date","calories"]); conn.close()
    if not df.empty: df["calories"]=pd.to_numeric(df["calories"],errors="coerce").fillna(0)
    return df
def add_water(uid):
    today=datetime.now().strftime("%Y-%m-%d"); conn=get_conn(); c=conn.cursor()
    c.execute("SELECT glasses FROM water_logs WHERE user_id=? AND date=?",(uid,today)); row=c.fetchone()
    if row: g=row[0]+1; c.execute("UPDATE water_logs SET glasses=? WHERE user_id=? AND date=?",(g,uid,today))
    else: g=1; c.execute("INSERT INTO water_logs (user_id,date,glasses) VALUES (?,?,?)",(uid,today,g))
    conn.commit(); conn.close(); return g
def get_water(uid):
    today=datetime.now().strftime("%Y-%m-%d"); conn=get_conn(); c=conn.cursor()
    c.execute("SELECT glasses FROM water_logs WHERE user_id=? AND date=?",(uid,today)); row=c.fetchone(); conn.close()
    return row[0] if row else 0

def calculate_tdee(w,h,a,g,act):
    bmr = 10*w + 6.25*h - 5*a + (5 if g=="Male" else -161)
    mult={"Sedentary (rarely exercise)":1.2,"Light (1-3 days/week)":1.375,"Moderate (3-5 days/week)":1.55,"Active (6-7 days/week)":1.725,"Very Active (hard exercise & job)":1.9}
    return round(bmr*mult.get(act,1.2))
def calculate_deficit_limit(tdee,goal):
    if goal=="Lose weight (deficit 500 kcal)": return tdee-500
    if goal=="Lose fast (deficit 750 kcal)": return tdee-750
    if goal=="Maintain": return tdee
    if goal=="Gain weight (surplus 300 kcal)": return tdee+300
    return tdee

# ========= NUTRITION DATA =========
@st.cache_data
def load_nutrition_data():
    path=os.path.join(DATA_DIR,'nutrition_data.csv')
    if not os.path.exists(path):
        st.error("File data/nutrition_data.csv missing.")
        return pd.DataFrame(columns=["food_name","weight","calories","protein","carbohydrates","fat","fiber","sugar","sodium"])
    df=pd.read_csv(path)
    df=df.rename(columns={"label":"food_name"})
    for col in ['calories','weight','protein','carbohydrates','fat','fiber','sugar','sodium']:
        if col in df: df[col]=pd.to_numeric(df[col],errors='coerce').fillna(0)
    if 'food_name' in df: df['food_name']=df['food_name'].astype(str)
    return df

def get_nutrition_info(name,data):
    if not name or data is None or data.empty: return None
    row=data[data['food_name'].str.lower()==str(name).lower()]
    return row.iloc[0] if not row.empty else None

# ---------- Pretty nutrition block with icons ----------
def render_nutrition(nut: dict):
    """Render a compact bullet list with icons (Weight, Calories, Protein, etc.)."""
    st.markdown(
        """
        <style>
        ul.ul-compact { margin: 0 0 0.5rem 0; padding-left: 1.2rem; }
        ul.ul-compact li { margin: 0.15rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    items = [
        ("⚖️", "Weight (g)", f"{nut.get('weight', 0)} g"),
        ("🍔", "Calories", f"{nut.get('calories', 0)} kcal"),
        ("🍗", "Protein", f"{nut.get('protein', 0)} g"),
        ("🍞", "Carbs", f"{nut.get('carbohydrates', 0)} g"),
        ("🧈", "Fat", f"{nut.get('fat', 0)} g"),
        ("🌾", "Fiber", f"{nut.get('fiber', 0)} g"),
        ("🍬", "Sugar", f"{nut.get('sugar', 0)} g"),
        ("🧂", "Sodium", f"{nut.get('sodium', 0)} mg"),
    ]
    html = "<ul class='ul-compact'>"
    for icon, label, value in items:
        html += f"<li>{icon} <strong>{label}:</strong> {value}</li>"
    html += "</ul>"
    st.markdown(html, unsafe_allow_html=True)

# ========= API CLIENT =========
def _post_image(endpoint,bytes,timeout=120,retries=3):
    url=API_URL+endpoint; files={'file':('image.jpg',bytes,'image/jpeg')}
    last=None
    for i in range(retries):
        try:
            r=requests.post(url,files=files,timeout=timeout)
            if r.status_code==200: return r.json()
            else: last=f"HTTP {r.status_code} {r.text[:200]}"
        except Exception as e: last=str(e)
        time.sleep(2*(i+1))
    st.error(f"API failed: {last}"); return None
def predict_food_api(b): d=_post_image("/predict/food",b); return (d.get("class"),float(d.get("confidence",0))) if d else (None,0)
def predict_fruit_api(b): d=_post_image("/predict/fruit",b); return (d.get("class"),float(d.get("confidence",0))) if d else (None,0)
def predict_yolo_api(b): d=_post_image("/predict/yolo",b); return d.get("results",[]) if d else []

# ========= APP =========
st.set_page_config(page_title="MealMind AI", page_icon="🧠🍽️", layout="centered")
st.title("🧠🍽️ MealMind AI")
st.markdown(L["welcome"])

# ---- LOGIN ----
if "user_id" not in st.session_state: st.session_state.user_id=None
if "username" not in st.session_state: st.session_state.username=None
if not st.session_state.user_id:
    tab1,tab2=st.tabs([L["login"],L["register"]])
    with tab1:
        u=st.text_input(L["username"],key="lu"); p=st.text_input(L["password"],type="password",key="lp")
        if st.button(L["login_btn"]):
            uid=login(u,p)
            if uid: st.session_state.user_id=uid; st.session_state.username=u; st.success(L["login_success"]); st.rerun()
            else: st.error(L["login_fail"])
    with tab2:
        u=st.text_input(L["username"],key="ru"); p=st.text_input(L["password"],type="password",key="rp")
        if st.button(L["register_btn"]):
            if register(u,p): st.success(L["register_success"])
            else: st.error(L["register_fail"])
    st.stop()

st.sidebar.markdown(f"👋 Hi, **{st.session_state.username}**!")
if st.sidebar.button(L["logout"]): st.session_state.user_id=None; st.session_state.username=None; st.success(L["logout_success"]); st.rerun()

# ---- PROFILE ----
profile=get_profile(st.session_state.user_id)
with st.sidebar.form("profile"):
    w=st.number_input(L["weight"],30,300,int(profile['weight']) if profile else 70)
    h=st.number_input(L["height"],100,250,int(profile['height']) if profile else 170)
    a=st.number_input(L["age"],10,100,int(profile['age']) if profile else 25)
    g=st.selectbox(L["gender"],["Male","Female"],0 if not profile or profile['gender']=="Male" else 1)
    acts=["Sedentary (rarely exercise)","Light (1-3 days/week)","Moderate (3-5 days/week)","Active (6-7 days/week)","Very Active (hard exercise & job)"]
    act=st.selectbox(L["activity"],acts,acts.index(profile['activity']) if profile else 0)
    goals=["Lose weight (deficit 500 kcal)","Lose fast (deficit 750 kcal)","Maintain","Gain weight (surplus 300 kcal)"]
    goal=st.selectbox(L["goal"],goals,goals.index(profile['goal']) if profile else 0)
    if st.form_submit_button(L["save_profile"]):
        tdee=calculate_tdee(w,h,a,g,act); limit=calculate_deficit_limit(tdee,goal)
        save_profile(st.session_state.user_id,{"weight":w,"height":h,"age":a,"gender":g,"activity":act,"goal":goal,"tdee":tdee,"daily_limit":limit})
        st.success(L["profile_saved"]); st.rerun()
profile=get_profile(st.session_state.user_id)
if profile: st.sidebar.info(f"TDEE: **{profile['tdee']} kcal**\nTarget: **{profile['daily_limit']} kcal**\nGoal: **{profile['goal']}**")
else: st.sidebar.warning("Complete your profile for better tracking.")

# ---- WATER ----
st.sidebar.header(L["water_tracker"])
if st.sidebar.button(L["add_water"]): add_water(st.session_state.user_id)
water=get_water(st.session_state.user_id); st.sidebar.progress(min(water/8,1.0))
st.sidebar.markdown(L["today_water"].format(water*250))
st.sidebar.info(L["water_info"] if water<8 else L["water_done"])

# ---- QUICK LOG ----
with st.expander(L["quick_log"]):
    data=load_nutrition_data()
    mf=st.text_input(L["type_food"]); mg=st.number_input(L["grams"],10,1000,100,10)
    if st.button(L["add_to_log"]):
        m=data[data["food_name"].str.contains(mf,case=False,na=False,regex=False)]
        if not m.empty:
            n=m.iloc[0]; base=n['weight'] if n['weight']>0 else 100; cal=(mg/base)*n['calories']
            st.success(L["food_added"].format(food=n['food_name'],gram=mg,cal=int(cal)))
            add_log(st.session_state.user_id,n['food_name'],cal)
        else: st.warning(L["food_not_found"])

# ---- FOOD SCAN ----
st.markdown("---"); st.subheader("📷 Food Scan")
im=None
method=st.radio(L["scan_image"],[L["upload_image"],L["camera"]],horizontal=True)
if method==L["upload_image"]: f=st.file_uploader(L["upload_image"],type=["jpg","jpeg","png"]); im=f if f else None
else: f=st.camera_input(L["camera"]); im=f if f else None

if "pending_items" not in st.session_state: st.session_state.pending_items=[]
if im:
    img=im.read(); st.image(Image.open(BytesIO(img)),caption=L["preview"],use_container_width=True)
    if st.button(f"🔎 {L['analyze_photo']}",use_container_width=True):
        st.session_state.pending_items=[]; data=load_nutrition_data()
        yres=predict_yolo_api(img)
        if isinstance(yres,list) and len(yres)>1:
            for it in yres:
                name=it.get("class"); conf=float(it.get("confidence",0))
                nut=get_nutrition_info(name,data)
                if name and conf>=CONF_THRESHOLD and nut is not None:
                    ref=nut['weight'] if nut['weight']>0 else 100
                    st.session_state.pending_items.append({"name":name,"conf":conf,"grams":ref,"ref_weight":ref,"cal_ref":nut['calories'],"nutrition":nut.to_dict()})
            if not st.session_state.pending_items: st.error(L["nothing_detected"])
        else:
            f_pred,c_food=predict_food_api(img); fr_pred,c_fruit=predict_fruit_api(img)
            if (c_fruit>c_food and c_fruit>0.5): name,conf=fr_pred,c_fruit
            else: name,conf=f_pred,c_food
            nut=get_nutrition_info(name,data) if name and conf>=CONF_THRESHOLD else None
            if name and nut is not None:
                ref=nut['weight'] if nut['weight']>0 else 100
                st.session_state.pending_items=[{"name":name,"conf":conf,"grams":ref,"ref_weight":ref,"cal_ref":nut['calories'],"nutrition":nut.to_dict()}]
            else: st.error(L["nothing_detected"])

    if st.session_state.pending_items:
        st.markdown("### 🍽️ Detected Items")
        total=0
        for i,it in enumerate(st.session_state.pending_items):
            with st.container(border=True):
                c1, c2 = st.columns([2,1])
                c1.markdown(f"**{it['name']}** · {L['ai_confidence'].format(conf=it['conf']*100)}")
                grams=c2.number_input(L["adjust_portion"],10.0,1000.0,step=10.0,value=float(it["grams"]),key=f"g{i}")
                it["grams"]=grams
                ref=it["ref_weight"] if it["ref_weight"]>0 else 100
                cal=(grams/ref)*it["cal_ref"]; total+=cal
                c1.markdown(f"#### {L['nutrition_info']}")
                render_nutrition(it["nutrition"])
        st.markdown(f"### 🧾 This scan: **{total:.0f} kcal**")
        if st.button(f"✅ {L['add_to_diary']}",use_container_width=True):
            for it in st.session_state.pending_items:
                ref=it["ref_weight"] if it["ref_weight"]>0 else 100; cal=(it["grams"]/ref)*it["cal_ref"]
                add_log(st.session_state.user_id,it["name"],cal)
            st.session_state.pending_items=[]; st.success(L["logged_ok"])

# ---- TODAY LOG ----
st.sidebar.markdown("---"); st.sidebar.markdown("### 📅 Today's Log")
df=get_today_logs(st.session_state.user_id)
if not df.empty:
    total=float(df["calories"].sum()); st.sidebar.dataframe(df.tail(10),use_container_width=True)
else: total=0; st.sidebar.info("No foods logged yet today.")

# ---- GOAL / PROGRESS ----
if profile:
    limit=float(profile["daily_limit"]); remain=limit-total
    st.sidebar.progress(min(total/max(limit,1),1.0))
    st.sidebar.markdown(L["logged_total"].format(total=int(total),target=int(limit)))
    st.sidebar.markdown(L["remaining"].format(remain=int(remain)))
    if total<limit-400: st.sidebar.warning(L["under_limit"])
    elif total>limit: st.sidebar.error(L["over_limit"])
    else: st.sidebar.success(L["on_track"])

    st.sidebar.markdown("---"); st.sidebar.markdown("### 📈 Weekly Summary")
    week=get_weekly_summary(st.session_state.user_id)
    if not week.empty:
        week['date']=pd.to_datetime(week['date']); avg=limit-week["calories"].mean()
        st.sidebar.markdown(f"Avg deficit/day: {avg:.0f} kcal")
        if avg>0: st.sidebar.markdown(f"Est fat loss/week: {avg*7/7700:.2f} kg")
        chart=alt.Chart(week).mark_line(point=True).encode(x='date:T',y='calories:Q').properties(title="Daily Calorie Trend").interactive()
        st.sidebar.altair_chart(chart,use_container_width=True)

    # snack quick add
    data=load_nutrition_data()
    if remain<200 and remain>0 and not data.empty:
        choices=data[(data['calories']<=remain)&(data['calories']>0)]
        if not choices.empty:
            snack=choices.sample(1).iloc[0]
            if st.sidebar.button(f"Snack: {snack['food_name']} ({int(snack['calories'])} kcal)"):
                add_log(st.session_state.user_id,snack['food_name'],snack['calories']); st.sidebar.success("Added snack!")

    if not df.empty:
        csv=df.to_csv(index=False).encode('utf-8'); st.sidebar.download_button("⬇️ Download Today",csv,"today.csv","text/csv")
    all_log=get_all_logs(st.session_state.user_id)
    if not all_log.empty:
        csv=all_log.to_csv(index=False).encode('utf-8'); st.sidebar.download_button("⬇️ Download All",csv,"all.csv","text/csv")
    st.sidebar.markdown("---"); st.sidebar.success(random.choice(L["motivation"]))
else:
    st.sidebar.info("Complete your profile for full tracking.")

st.sidebar.markdown("---"); st.sidebar.caption("Made with 💪 and AI for smarter living.")
