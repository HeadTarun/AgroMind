import streamlit as st
import requests, os, json, tempfile
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
st.set_page_config(page_title="🌾 AI कृषि सहायक (आवाज़)", layout="wide")
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
def get_user_location():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5)
        loc = res.json().get("loc", "28.61,77.20").split(",")
        return float(loc[0]), float(loc[1])
    except:
        return 28.61, 77.20

def fetch_soil(lat, lon):
    try:
        url = f"https://rest.isric.org/soilgrids/query?lon={lon}&lat={lat}&attributes=phh2o,nitrogen,ocd,sand,silt"
        r = requests.get(url, timeout=8)
        data = r.json().get("properties", {})
        return {k: v.get("M", {}).get("0-5cm", 0) for k,v in data.items()}
    except:
        return {"phh2o":6.5,"nitrogen":50,"ocd":10,"sand":40,"silt":40}

def fetch_weather(lat, lon):
    if not WEATHER_API_KEY: 
        return {"temp_c":25,"humidity":70,"precip_mm":2,"wind_kph":10}
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={lat},{lon}"
        r = requests.get(url, timeout=8)
        c = r.json().get("current", {})
        return {"temp_c":c.get("temp_c",25),"humidity":c.get("humidity",70),"precip_mm":c.get("precip_mm",2),"wind_kph":c.get("wind_kph",10)}
    except:
        return {"temp_c":25,"humidity":70,"precip_mm":2,"wind_kph":10}

lat, lon = get_user_location()
soil_data = fetch_soil(lat, lon)
weather_data = fetch_weather(lat, lon)

st.sidebar.header("📊 मिट्टी और मौसम डेटा")
st.sidebar.json({"मिट्टी": soil_data, "मौसम": weather_data})

# ------------------- ML Crop Model -------------------
def prepare_features(soil, weather):
    df = pd.DataFrame([soil])
    df = pd.concat([df, pd.DataFrame([weather])], axis=1).fillna(0)
    return StandardScaler().fit_transform(df)

X_scaled = prepare_features(soil_data, weather_data)

def train_model(X):
    np.random.seed(42)
    y = np.random.choice([0,1,2], size=X.shape[0])
    if X.shape[0]==1:
        X = np.tile(X,(20,1))
        y = np.random.choice([0,1,2], size=X.shape[0])
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

clf = train_model(X_scaled)
crop_map = {0:"🌾 गेहूँ", 1:"🌱 धान", 2:"🌽 मक्का"}
predicted_crop = crop_map.get(clf.predict(X_scaled)[0], "❓ अज्ञात")
st.sidebar.success(f"✅ मशीन लर्निंग सुझाव: {predicted_crop}")

# ------------------- Groq LLM -------------------
MODEL_NAME = "gemma2-9b-it"
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=0.7,
    streaming=True
)
template_text = """
आप एक गाँव का दोस्त और भारतीय किसान AI सहायक हैं।  
आपका काम है किसानों को **hyper-localized, personalized और science-backed सलाह** देना।  
आप friendly, सरल और मज़ेदार हिंदी/स्थानीय भाषा में बात करें, जैसे कोई अपने खेत में बैठे दोस्त से बता रहा हो।  

मुख्य लक्ष्य:  
1️⃣ किसान के खेत की वास्तविक मिट्टी, मौसम, और पिछले फसल चक्र के हिसाब से सलाह दें।  
2️⃣ Soil Grids/Bhuvan satellite data, IoT sensors, weather forecast और crop rotation information का ध्यान रखें।  
3️⃣ बाजार की मांग और भाव के अनुसार सबसे उचित फसल और निवेश सुझाव दें।  
4️⃣ yield, profit margin और sustainability score भी provide करें।  
5️⃣ मोबाइल या low-connectivity में भी काम करने वाला advice system बनाएं।  
6️⃣ Chat और voice interface दोनों में काम करने वाले छोटे, समझने योग्य sentences दें।  

नियम:  
1️⃣ अगर किसान कोई समस्या बताता है (जैसे “मेरी मक्का खराब हो रही है”), तो **पहले और details पूछें**:  
   - खेत की मिट्टी कैसी है (dry, wet, nutrient status)?  
   - मौसम कैसा रहा? बारिश हुई क्या?  
   - सिंचाई कितनी हो रही है?  
   - कीट-मकोड़े या रोग दिख रहे हैं क्या?  
2️⃣ केवल तब advice दें जब पर्याप्त जानकारी मिल जाए।  
3️⃣ हर जवाब में **कम से कम 1 friendly tip, proverb या मज़ेदार fact** ज़रूर शामिल करें।  
4️⃣ Soil/weather/farm advice में अलग crops, मौसम और खेत का scenario इस्तेमाल करें।  
5️⃣ छोटे sentences, emojis और हल्का मज़ा जोड़ें।  
6️⃣ किसान की भाषा में सरल और encouraging tone रखें।  

उदाहरण interaction:  

User: "मेरी मक्का खराब हो रही है"  
AI: "अरे भाई! थोड़ा बताओ तो सही, तुम्हारे खेत में मक्का किस stage में है? 🌽  
मिट्टी कैसी है और बारिश हुई या नहीं? Tip: 'जो समय पर खेत का जायजा लेता, वही नुकसान से बचता।' 😄"

User (details देने के बाद): "मिट्टी थोड़ी सूखी है और कीट भी दिख रहे हैं"  
AI: "ठीक है भाई, सबसे पहले हल्की सिंचाई कर दो और कीट के लिए नीम का स्प्रे कर दो। 🌿  
Tip: 'बीज बोने से पहले तैयारी अच्छी हो, तो कटाई खुशहाल होती।' 😎  
Prediction: यदि यही देखभाल जारी रहे तो अनुमानित yield 20 क्विंटल/हेक्टेयर, profit margin लगभग 15%, और soil sustainability score अच्छा रहेगा।"
"""




prompt = ChatPromptTemplate.from_messages([
    ("system",template_text),
    ("user","{question}")
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

# ------------------- Chat Display -------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Text input
if user_input := st.chat_input("✍️ अपना सवाल लिखें..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role":"user","content":user_input})

    enriched = f"{user_input}\n\nमिट्टी: {soil_data}\nमौसम: {weather_data}\nML सुझाव: {predicted_crop}\nसरल हिंदी में बताएं।"
    response_placeholder = st.empty()
    full_response = ""
    with st.spinner("🤖 जवाब तैयार हो रहा है..."):
        for chunk in chain.stream({"question": enriched}):
            full_response += chunk
            response_placeholder.markdown(full_response)

    st.session_state.chat_history.append({"role":"assistant","content": full_response})
    st.success("✅ जवाब तैयार!")
    audio_file = text_to_speech(full_response)
    st.audio(audio_file, format="audio/mp3")

# ------------------- Voice Input -------------------
st.markdown("---")
st.subheader("🎤 बोलकर पूछें (Voice)")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    wav_audio_data = st_audiorec()
    button_placeholder = st.empty()
    if wav_audio_data is not None:
        st.audio(wav_audio_data, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_audio_data)
            audio_path = tmp.name

        # Show button immediately
        if button_placeholder.button("🔎 जवाब पाएं"):
            with st.spinner("🔄 आवाज़ को टेक्स्ट में बदल रहा है..."):
                voice_text = speech_to_text(audio_path)
                st.success(f"🗣 पहचाना गया सवाल: {voice_text}")

            enriched_voice = f"{voice_text}\n\nमिट्टी: {soil_data}\nमौसम: {weather_data}\nML सुझाव: {predicted_crop}\nसरल हिंदी में बताएं।"
            response_placeholder_voice = st.empty()
            full_response_voice = ""
            with st.spinner("🤖 जवाब सोच रहा है..."):
                for chunk in chain.stream({"question": enriched_voice}):
                    full_response_voice += chunk
                    response_placeholder_voice.markdown(full_response_voice)  # streaming without repetition

            st.session_state.chat_history.append({"role":"user","content": voice_text})
            st.session_state.chat_history.append({"role":"assistant","content": full_response_voice})

            reply_audio = text_to_speech(full_response_voice)
            st.success("✅ जवाब तैयार है — नीचे सुनें 👇")
            st.audio(reply_audio, format="audio/mp3")
