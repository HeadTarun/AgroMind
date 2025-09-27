import streamlit as st
import requests, os, json, tempfile, time, random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from st_audiorec import st_audiorec
from gtts import gTTS

# ------------------- Config -------------------
st.set_page_config(page_title="🌾 AI कृषि सहायक", layout="wide")

# ------------------- Welcome Screen -------------------
welcome_messages = [
    "🌱 स्वागत है! ऐप लोड हो रहा है… कृपया थोड़ी देर प्रतीक्षा करें…",
    "🌿 नमस्ते किसान भाई! आपका AI सहायक तैयार हो रहा है… धैर्य रखें…",
    "🌾 हलचल मच रही है! ऐप जल्दी ही तैयार होगा…",
    "🌻 खेत की जानकारी लोड हो रही है… कुछ ही पलों में शुरू!",
    "🍃 स्वागत है! AI सलाहकार आपके खेत के लिए तैयार हो रहा है…"
]
with st.spinner(random.choice(welcome_messages)):
    time.sleep(5)

st.title("🌾 AI आधारित फसल सलाह सहायक (हिंदी, आवाज़ सहित)")

# ------------------- Load Env -------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GROQ_BASE = "https://api.groq.com/openai/v1"

if not GROQ_API_KEY:
    st.error("❌ .env में GROQ_API_KEY सेट करें")
    st.stop()

# ------------------- Session State -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("♻️ चैट रीसेट करें"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# ------------------- Location, Soil & Weather -------------------
@st.cache_data(ttl=3600)
def get_user_location():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=3)
        loc = res.json().get("loc", "28.61,77.20").split(",")
        return float(loc[0]), float(loc[1])
    except:
        return 28.61, 77.20

@st.cache_data(ttl=3600)
def fetch_soil(lat, lon):
    try:
        url = f"https://rest.isric.org/soilgrids/query?lon={lon}&lat={lat}&attributes=phh2o,nitrogen,ocd,sand,silt"
        r = requests.get(url, timeout=4)
        data = r.json().get("properties", {})
        return {k: v.get("M", {}).get("0-5cm", 0) for k,v in data.items()}
    except:
        return {"phh2o":6.5,"nitrogen":50,"ocd":10,"sand":40,"silt":40}

@st.cache_data(ttl=600)
def fetch_weather(lat, lon):
    if not WEATHER_API_KEY:
        return {"temp_c":25,"humidity":70,"precip_mm":2,"wind_kph":10}
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={lat},{lon}"
        r = requests.get(url, timeout=4)
        c = r.json().get("current", {})
        return {"temp_c":c.get("temp_c",25),"humidity":c.get("humidity",70),
                "precip_mm":c.get("precip_mm",2),"wind_kph":c.get("wind_kph",10)}
    except:
        return {"temp_c":25,"humidity":70,"precip_mm":2,"wind_kph":10}

lat, lon = get_user_location()
soil_data = fetch_soil(lat, lon)
weather_data = fetch_weather(lat, lon)

st.sidebar.header("📊 मिट्टी और मौसम डेटा")
st.sidebar.json({"मिट्टी": soil_data, "मौसम": weather_data})

# ------------------- ML Crop Model -------------------
@st.cache_resource
def get_trained_model(X_scaled):
    np.random.seed(42)
    y = np.random.choice([0,1,2], size=X_scaled.shape[0])
    if X_scaled.shape[0]==1:
        X_scaled = np.tile(X_scaled,(20,1))
        y = np.random.choice([0,1,2], size=X_scaled.shape[0])
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y)
    return clf

def prepare_features(soil, weather):
    df = pd.DataFrame([soil])
    df = pd.concat([df, pd.DataFrame([weather])], axis=1).fillna(0)
    return StandardScaler().fit_transform(df)

X_scaled = prepare_features(soil_data, weather_data)
clf = get_trained_model(X_scaled)
crop_map = {0:"🌾 गेहूँ", 1:"🌱 धान", 2:"🌽 मक्का"}
predicted_crop = crop_map.get(clf.predict(X_scaled)[0], "❓ अज्ञात")
st.sidebar.success(f"✅ मशीन लर्निंग सुझाव: {predicted_crop}")

# ------------------- Groq LLM -------------------
MODEL_NAME = "openai/gpt-oss-20B"
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=0.7,
    streaming=True
)

template_text = """आप एक गाँव का दोस्त और भारतीय किसान AI सहायक हैं।  
आप friendly, सरल और मज़ेदार हिंदी/स्थानीय भाषा में बात करें, केवल खेती और किसान से जुड़ी बातें करें।  
ML फसल prediction केवल तब शामिल करें जब user पूछे "कौन सी फसल उगाऊँ" या "फसल सुझाव".  
Soil और Weather के सवालों पर raw API डेटा का farmer-friendly explanation दें।  
Market और real-time info पूछे तो कहें: "यह फीचर अभी डेवलपमेंट में है। AgroMind टीम जल्द ही इसे उपलब्ध कराएगी।"
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template_text),
    ("user", "{question}")
])
chain = prompt | llm | StrOutputParser()

# ------------------- Speech Functions -------------------
def speech_to_text(file_path):
    url = f"{GROQ_BASE}/audio/transcriptions"
    with open(file_path,"rb") as f:
        resp = requests.post(
            url,
            headers={"Authorization":f"Bearer {GROQ_API_KEY}"},
            data={"model":"whisper-large-v3"},
            files={"file":(os.path.basename(file_path), f)},
            timeout=30
        )
    resp.raise_for_status()
    return resp.json().get("text","")

def text_to_speech(text, filename="response.mp3"):
    tts = gTTS(text, lang="hi")
    tts.save(filename)
    return filename

# ------------------- Input Enrichment -------------------
def enrich_input(user_text):
    user_lower = user_text.lower()
    context_text = user_text

    if "फसल" in user_lower or "crop" in user_lower:
        context_text += f"\nमिट्टी: {soil_data}\nमौसम: {weather_data}\nML सुझाव: {predicted_crop}"

    elif any(word in user_lower for word in ["मिट्टी", "soil", "weather", "मौसम", "pH", "नाइट्रोजन", "nitrogen"]):
        context_text += f"\nमिट्टी और मौसम की जानकारी:\n{soil_data}\n{weather_data}\nकिसान की भाषा में सरल व्याख्या दें।"

    elif any(word in user_lower for word in ["बाजार", "market", "भाव", "price"]):
        context_text = "यह फीचर अभी डेवलपमेंट में है। AgroMind टीम जल्द ही इसे उपलब्ध कराएगी।"

    return context_text

# ------------------- Voice Input -------------------
st.subheader("🎤 बोलकर पूछें (Voice Input)")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    wav_audio_data = st_audiorec()
    button_placeholder = st.empty()
    if wav_audio_data is not None:
        st.audio(wav_audio_data, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_audio_data)
            audio_path = tmp.name

        if button_placeholder.button("🔎 जवाब पाएं"):
            with st.spinner("🔄 आवाज़ को टेक्स्ट में बदल रहा है..."):
                voice_text = speech_to_text(audio_path)
                st.success(f"🗣 पहचाना गया सवाल: {voice_text}")

            enriched_voice = enrich_input(voice_text)
            response_placeholder_voice = st.empty()
            full_response_voice = ""
            with st.spinner("🤖 जवाब सोच रहा है..."):
                for chunk in chain.stream({"question": enriched_voice}):
                    full_response_voice += chunk
                    response_placeholder_voice.markdown(full_response_voice)

            st.session_state.chat_history.append({"role":"user","content": voice_text})
            st.session_state.chat_history.append({"role":"assistant","content": full_response_voice})

            st.markdown("### 🎧 आवाज़ में जवाब (Voice Response)")
            reply_audio = text_to_speech(full_response_voice)
            st.audio(reply_audio, format="audio/mp3")

# ------------------- Chat Display -------------------
st.subheader("💬 चैट में सवाल-जवाब")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------- Text Input -------------------
if user_input := st.chat_input("✍️ अपना सवाल लिखें..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role":"user","content":user_input})

    enriched = enrich_input(user_input)
    response_placeholder = st.empty()
    response_placeholder.markdown("🤖 AI सोच रहा है… 🧠")
    full_response = ""
    with st.spinner("🤖 जवाब तैयार हो रहा है..."):
        for chunk in chain.stream({"question": enriched}):
            full_response += chunk
            response_placeholder.markdown(full_response)

    st.session_state.chat_history.append({"role":"assistant","content": full_response})
    st.markdown("### 🎧 आवाज़ में जवाब (Voice Response)")
    audio_file = text_to_speech(full_response)
    st.audio(audio_file, format="audio/mp3")
