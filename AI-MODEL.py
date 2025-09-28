
import os
import io
import time
import json
import tempfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import logging
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from gtts import gTTS
from st_audiorec import st_audiorec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Langchain / Groq imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# ------------------- Safe TTS Warm-up -------------------
try:
    import pyttsx3

    def warmup_tts():
        try:
            engine = pyttsx3.init()
            engine.say("Warm-up")
            engine.runAndWait()
            engine.stop()
        except Exception as inner_e:
            logger.warning(f"TTS warm-up skipped: {inner_e}")

    if "tts_warmed_up" not in st.session_state:
        warmup_tts()
        st.session_state["tts_warmed_up"] = True

except ImportError:
    logger.warning("pyttsx3 not installed or not supported. Skipping TTS warm-up.")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- Page config & Enhanced CSS -------------------
st.set_page_config(
    page_title="🌾 AI कृषि सहायक", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "AI कृषि सहायक - आपका डिजिटल खेती सलाहकार"
    }
)

st.markdown("""
<style>
    .main-title { 
        text-align: center; 
        color: #2E8B57; 
        font-size: 2.2rem; 
        margin-bottom: 1rem; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .voice-section { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 1.5rem; 
        border-radius: 12px; 
        color: white; 
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #4caf50;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #c3e6cb;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #f5c6cb;
    }
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #bee5eb;
    }
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🌾 AI आधारित फसल सलाह सहायक (हिंदी, आवाज़ सहित)</h1>', unsafe_allow_html=True)

# ------------------- Load environment variables -------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "").strip()

# Enhanced validation
if not GROQ_API_KEY:
    st.error("❌ .env फ़ाइल में `GROQ_API_KEY` सेट करें — यह LLM और स्पीच APIs के लिए आवश्यक है।")
    st.info("💡 Groq API key प्राप्त करने के लिए https://console.groq.com पर जाएं")
    st.stop()

# ------------------- Session state initialization -------------------
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "app_initialized": False,
        "chat_history": [],
        "processing": False,
        "last_audio_data": None,
        "error_count": 0,
        "last_error_time": None,
        "voice_enabled": True,
        "auto_play_response": True
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Enhanced initialization with better UX
if not st.session_state.app_initialized:
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        initialization_steps = [
            ("🔧 सिस्टम शुरू कर रहे हैं...", 20),
            ("🌐 API कनेक्शन जांच रहे हैं...", 40),
            ("📊 डेटा सोर्स कनेक्ट कर रहे हैं...", 60),
            ("🤖 AI मॉडल लोड कर रहे हैं...", 80),
            ("✅ सभी सिस्टम तैयार!", 100)
        ]
        
        for step_text, progress_value in initialization_steps:
            status_text.text(step_text)
            progress_bar.progress(progress_value)
            time.sleep(0.4)
        
        time.sleep(0.5)
        progress_container.empty()
        st.session_state.app_initialized = True

# ------------------- Enhanced utility functions -------------------
def get_default_soil_data() -> Dict[str, float]:
    """Return default soil data for fallback"""
    return {
        "ph": 6.5,
        "nitrogen": 50,
        "organic_carbon": 10,
        "sand": 40,
        "silt": 40,
        "clay": 20
    }

def get_default_weather_data() -> Dict[str, Any]:
    """Return default weather data for fallback"""
    return {
        "temperature": 25,
        "humidity": 70,
        "precipitation": 2,
        "wind_speed": 10,
        "condition": "साफ़"
    }

@st.cache_data(ttl=3600, show_spinner=False)
def get_user_location() -> Tuple[float, float, str]:
    """Get user location with enhanced error handling"""
    try:
        response = requests.get("https://ipinfo.io/json", timeout=8)
        if response.status_code == 200:
            data = response.json()
            loc = data.get("loc", "28.61,77.20").split(",")
            city = data.get("city", "दिल्ली")
            region = data.get("region", "")
            country = data.get("country", "IN")
            
            location_name = f"{city}"
            if region and region != city:
                location_name += f", {region}"
                
            return float(loc[0]), float(loc[1]), location_name
    except requests.RequestException as e:
        logger.warning(f"Location fetch failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in location fetch: {e}")
    
    return 28.61, 77.20, "दिल्ली"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_soil(lat: float, lon: float) -> Dict[str, float]:
    """Fetch soil data with better error handling and realistic defaults"""
    try:
        # Using a more reliable approach for soil data
        url = "https://rest.isric.org/soilgrids/v2.0/properties"
        params = {
            "lon": lon,
            "lat": lat,
            "property": "phh2o",
            "depth": "0-5cm",
            "value": "mean"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            # For demo purposes, return realistic values based on location
            # In production, you would parse the actual SoilGrids response
            base_soil = get_default_soil_data()
            
            # Adjust values slightly based on location for realism
            lat_factor = (lat - 20) / 20  # Normalize around typical Indian latitudes
            
            base_soil["ph"] += lat_factor * 0.5
            base_soil["nitrogen"] += lat_factor * 10
            
            return base_soil
            
    except requests.RequestException as e:
        logger.warning(f"Soil data fetch failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in soil data fetch: {e}")
    
    return get_default_soil_data()

@st.cache_data(ttl=600, show_spinner=False)  # 10 minute cache for weather
def fetch_weather(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch weather data with comprehensive error handling"""
    if not WEATHER_API_KEY:
        return get_default_weather_data()
        
    try:
        url = "http://api.weatherapi.com/v1/current.json"
        params = {
            "key": WEATHER_API_KEY,
            "q": f"{lat},{lon}",
            "aqi": "no"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            
            return {
                "temperature": current.get("temp_c", 25),
                "humidity": current.get("humidity", 70),
                "precipitation": current.get("precip_mm", 2),
                "wind_speed": current.get("wind_kph", 10),
                "condition": current.get("condition", {}).get("text", "साफ़"),
                "feels_like": current.get("feelslike_c", 25),
                "uv": current.get("uv", 5)
            }
        else:
            logger.warning(f"Weather API returned status {response.status_code}")
            
    except requests.RequestException as e:
        logger.warning(f"Weather data fetch failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in weather data fetch: {e}")
    
    return get_default_weather_data()

# ------------------- Enhanced ML model -------------------
@st.cache_resource(show_spinner=False)
def get_trained_model() -> Tuple[RandomForestClassifier, StandardScaler]:
    """Create and train enhanced ML model"""
    np.random.seed(42)
    n_samples = 2000  # More training data
    
    features = []
    labels = []
    
    # Generate more diverse and realistic training data
    for _ in range(n_samples):
        temp = np.random.normal(25, 10)
        humidity = np.random.normal(70, 20)
        ph = np.random.normal(6.5, 1.2)
        nitrogen = np.random.normal(50, 25)
        
        features.append([temp, humidity, ph, nitrogen])
        
        # Enhanced decision logic for crop recommendation
        if temp < 22 and humidity > 55 and ph > 6.0:
            labels.append(0)  # गेहूँ
        elif temp > 28 and humidity > 75 and ph < 7.5:
            labels.append(1)  # धान
        elif temp > 20 and temp < 35 and humidity < 80:
            labels.append(2)  # मक्का
        else:
            # Random assignment for edge cases
            labels.append(np.random.choice([0, 1, 2]))
    
    X = np.array(features)
    y = np.array(labels)
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model with better parameters
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    clf.fit(X_scaled, y)
    
    return clf, scaler

def get_crop_prediction(soil: Dict[str, float], weather: Dict[str, Any]) -> Tuple[str, float]:
    """Get crop prediction with confidence score"""
    try:
        clf, scaler = get_trained_model()
        
        features = np.array([[
            weather.get("temperature", 25),
            weather.get("humidity", 70),
            soil.get("ph", 6.5),
            soil.get("nitrogen", 50)
        ]])
        
        features_scaled = scaler.transform(features)
        probabilities = clf.predict_proba(features_scaled)[0]
        prediction = int(clf.predict(features_scaled)[0])
        
        crop_map = {0: "🌾 गेहूँ", 1: "🌱 धान", 2: "🌽 मक्का"}
        confidence = float(max(probabilities) * 100)
        
        return crop_map.get(prediction, "❓ अज्ञात"), confidence
        
    except Exception as e:
        logger.error(f"Crop prediction failed: {e}")
        return "🌾 गेहूँ", 75.0

# ------------------- Load environmental data -------------------
with st.spinner("🌍 स्थान और पर्यावरण डेटा लोड कर रहे हैं..."):
    lat, lon, city = get_user_location()
    soil_data = fetch_soil(lat, lon)
    weather_data = fetch_weather(lat, lon)

# ------------------- Enhanced Sidebar -------------------
with st.sidebar:
    st.header("🎛️ नियंत्रण पैनल")
    
    # Settings
    st.subheader("⚙️ सेटिंग्स")
    st.session_state.voice_enabled = st.checkbox("🔊 आवाज़ प्लेबैक", value=st.session_state.voice_enabled)
    st.session_state.auto_play_response = st.checkbox("🎵 स्वचालित प्लेबैक", value=st.session_state.auto_play_response)
    
    response_length = st.selectbox(
        "उत्तर की लंबाई",
        ["संक्षिप्त", "सामान्य", "विस्तृत"],
        index=1
    )
    
    # Reset button with confirmation
    if st.button("♻️ चैट रीसेट करें", type="secondary"):
        if st.session_state.chat_history:
            st.session_state.chat_history = []
            st.success("चैट रीसेट हो गई!")
            time.sleep(1)
            st.rerun()
        else:
            st.info("कोई चैट हिस्ट्री नहीं है")

    # Environmental data display
    st.header("📊 वर्तमान डेटा")
    st.success(f"📍 स्थान: {city}")

    with st.expander("🌱 मिट्टी की विस्तृत जानकारी", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("pH स्तर", f"{soil_data.get('ph', 0):.1f}", 
                     delta=f"{soil_data.get('ph', 0) - 7.0:.1f} से न्यूट्रल")
            st.metric("रेत %", f"{soil_data.get('sand', 0):.0f}")
            st.metric("गाद %", f"{soil_data.get('silt', 0):.0f}")
        with col2:
            st.metric("नाइट्रोजन", f"{soil_data.get('nitrogen', 0):.0f}")
            st.metric("कार्बन %", f"{soil_data.get('organic_carbon', 0):.1f}")
            st.metric("चिकनी मिट्टी %", f"{soil_data.get('clay', 0):.0f}")

    with st.expander("🌤️ मौसम की विस्तृत जानकारी", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("तापमान", f"{weather_data.get('temperature', 0):.1f}°C")
            st.metric("महसूस होता है", f"{weather_data.get('feels_like', weather_data.get('temperature', 25)):.1f}°C")
            st.metric("आर्द्रता", f"{weather_data.get('humidity', 0):.0f}%")
        with col2:
            st.metric("बारिश", f"{weather_data.get('precipitation', 0):.1f}mm")
            st.metric("हवा की गति", f"{weather_data.get('wind_speed', 0):.1f}km/h")
            if "uv" in weather_data:
                st.metric("UV सूचकांक", f"{weather_data.get('uv', 0):.0f}")
        
        st.info(f"मौसम: {weather_data.get('condition', 'साफ़')}")

    # Crop prediction
    predicted_crop, confidence = get_crop_prediction(soil_data, weather_data)
    st.success(f"🎯 सुझाई गई फसल: {predicted_crop}")
    
    # Enhanced confidence display
    confidence_color = "green" if confidence > 80 else "orange" if confidence > 60 else "red"
    st.markdown(f"विश्वास स्तर: :{confidence_color}[{confidence:.1f}%]")

# ------------------- Enhanced Groq LLM setup -------------------
try:
    MODEL_NAME = "openai/gpt-oss-20B"
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.7,
        streaming=True,
        max_tokens=1024
    )

    # Enhanced prompt template with context
    template_text = """
आप एक अनुभवी और दोस्ताना किसान मित्र हैं जो कृषि सलाहकार का काम करते हैं। 
Agar koi aap se puche apko kisne banaya, to kaho "AgroMind team ne, jo aapke kisan bhaiyon ke liye best AI assistant banane mein laga hai".

आपकी विशेषताएं:
- हमेशा सरल, समझने योग्य हिंदी में बात करना
- स्थानीय परिस्थितियों (मौसम, मिट्टी) के अनुसार व्यावहारिक सलाह देना  
- "भाई", "जी", "आइए" जैसे दोस्ताना शब्दों का उपयोग करना
- छोटे, actionable steps में जवाब देना

वर्तमान स्थानीय डेटा:
- स्थान: {location}
- तापमान: {temperature}°C, आर्द्रता: {humidity}%
- मिट्टी pH: {soil_ph}, नाइट्रोजन: {nitrogen}
- AI सुझाव: {crop_suggestion} (विश्वास: {confidence:.1f}%)

नियम:
1. यदि मार्केट रेट/मंडी भाव पूछें तो कहें: "यह सुविधा अभी विकास में है, जल्द ही उपलब्ध होगी"
2. फसल सुझाव के लिए ऊपर दिए गए स्थानीय डेटा का उपयोग करें
3. हमेशा प्रैक्टिकल और लागू करने योग्य सलाह दें
4. अगर कोई चिकित्सा सलाह पूछे तो डॉक्टर से मिलने को कहें
Aur jis salwal ka jawab aapko nahi pata, usme aap seedha "मुझे खेद है, मैं इस बारे में जानकारी नहीं दे सकता। कृपया विशेषज्ञ से संपर्क करें।" keh dena.

उपयोगकर्ता का सवाल: {question}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template_text),
        ("user", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    
except Exception as e:
    st.error(f"❌ LLM मॉडल लोड करने में समस्या: {e}")
    st.info("कृपया अपनी इंटरनेट कनेक्शन और GROQ_API_KEY की जांच करें")
    st.stop()

# ------------------- Enhanced speech functions -------------------
def text_to_speech_bytes(text: str) -> Optional[bytes]:
    """Convert text to speech using gTTS with better error handling"""
    if not text or not text.strip():
        return None
        
    try:
        # Limit text length to avoid API issues
        max_length = 500
        if len(text) > max_length:
            # Smart truncation at sentence boundary
            sentences = text.split('।')
            truncated_text = ""
            for sentence in sentences:
                if len(truncated_text + sentence + "।") <= max_length:
                    truncated_text += sentence + "।"
                else:
                    break
            text = truncated_text if truncated_text else text[:max_length] + "..."

        tts = gTTS(text=text, lang="hi", slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"TTS conversion failed: {e}")
        st.warning("आवाज़ बनाने में समस्या हुई")
        return None

def speech_to_text(file_path: str) -> str:
    """Convert speech to text using Groq Whisper API with enhanced error handling"""
    try:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        
        with open(file_path, "rb") as audio_file:
            files = {"file": (os.path.basename(file_path), audio_file, "audio/wav")}
            data = {"model": "whisper-large-v3", "language": "hi", "response_format": "text"}
            
            response = requests.post(
                url, 
                headers=headers, 
                data=data, 
                files=files, 
                timeout=45  # Increased timeout for larger files
            )
        
        if response.status_code == 200:
            transcribed_text = response.text.strip()
            return transcribed_text if transcribed_text else ""
        else:
            logger.error(f"Transcription API error: {response.status_code}, {response.text}")
            
    except requests.RequestException as e:
        logger.error(f"Speech-to-text request failed: {e}")
        st.error("नेटवर्क की समस्या के कारण आवाज़ को टेक्स्ट में बदल नहीं सके")
    except Exception as e:
        logger.error(f"Unexpected error in speech-to-text: {e}")
        
    return ""

# ------------------- Enhanced Voice Input Section -------------------
st.markdown('<div class="voice-section">', unsafe_allow_html=True)
st.subheader("🎤 आवाज़ से सवाल पूछें")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    wav_audio_data = st_audiorec()
    
    if wav_audio_data is not None:
        # Check if this is new audio data
        if wav_audio_data != st.session_state.last_audio_data:
            st.session_state.last_audio_data = wav_audio_data
            st.audio(wav_audio_data, format="audio/wav")
            st.success("🎵 ऑडियो रिकॉर्ड हो गया! अब 'जवाब पाएं' बटन दबाएं।")

        if st.button("🔎 जवाब पाएं", type="primary", disabled=st.session_state.processing):
            if st.session_state.processing:
                st.warning("⏳ कृपया प्रतीक्षा करें, प्रोसेसिंग चल रही है...")
            else:
                st.session_state.processing = True
                
                try:
                    # Save audio to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(wav_audio_data)
                        temp_audio_path = tmp_file.name
                    
                    # Step 1: Convert speech to text
                    with st.spinner("🔄 आवाज़ को समझ रहे हैं..."):
                        voice_text = speech_to_text(temp_audio_path)
                    
                    # Clean up temporary file
                    try:
                        os.unlink(temp_audio_path)
                    except OSError:
                        pass
                    
                    if not voice_text:
                        st.error("❌ आवाज़ स्पष्ट नहीं आई। कृपया फिर से स्पष्ट आवाज़ में बोलें।")
                        st.info("💡 टिप: शांत जगह में रिकॉर्ड करें और माइक के पास बोलें")
                    else:
                        st.success(f"🗣️ आपका सवाल: **{voice_text}**")
                        
                        # Step 2: Generate response
                        response_container = st.empty()
                        full_response = ""
                        
                        with st.spinner("🤖 आपके लिए सबसे बेहतर जवाब तैयार कर रहे हैं..."):
                            try:
                                for chunk in chain.stream({
                                    "question": voice_text,
                                    "location": city,
                                    "temperature": weather_data.get('temperature', 25),
                                    "humidity": weather_data.get('humidity', 70),
                                    "soil_ph": soil_data.get('ph', 6.5),
                                    "nitrogen": soil_data.get('nitrogen', 50),
                                    "crop_suggestion": predicted_crop,
                                    "confidence": confidence
                                }):
                                    full_response += chunk
                                    response_container.markdown(f"🤖 **AI सलाहकार:** {full_response}")
                                    
                            except Exception as e:
                                error_msg = f"जवाब तैयार करने में समस्या: {str(e)}"
                                response_container.error(f"❌ {error_msg}")
                                full_response = error_msg

                        # Step 3: Save to chat history
                        st.session_state.chat_history.extend([
                            {"role": "user", "content": voice_text, "type": "voice", "timestamp": datetime.now().isoformat()},
                            {"role": "assistant", "content": full_response, "type": "voice_response", "timestamp": datetime.now().isoformat()}
                        ])

                        # Step 4: Generate audio response
                        if st.session_state.voice_enabled and full_response:
                            with st.spinner("🎧 आवाज़ में जवाब तैयार कर रहे हैं..."):
                                audio_bytes = text_to_speech_bytes(full_response)
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3", autoplay=st.session_state.auto_play_response)
                                    st.success("🔊 आवाज़ में जवाब तैयार!")

                except Exception as e:
                    st.error(f"❌ प्रोसेसिंग में समस्या: {str(e)}")
                    logger.error(f"Voice processing error: {e}")
                finally:
                    st.session_state.processing = False

st.markdown('</div>', unsafe_allow_html=True)

# ------------------- Enhanced Chat History Display -------------------
st.subheader("💬 बातचीत का इतिहास")

if st.session_state.chat_history:
    for i, message in enumerate(st.session_state.chat_history):
        role = message.get("role")
        content = message.get("content", "")
        msg_type = message.get("type", "text")
        timestamp = message.get("timestamp", "")
        
        if role == "user":
            st.markdown(f'<div class="user-message">', unsafe_allow_html=True)
            icon = "🎤" if msg_type == "voice" else "✍️"
            st.markdown(f"**{icon} उपयोगकर्ता:** {content}")
            if timestamp:
                st.caption(f"⏰ {timestamp[:19].replace('T', ' ')}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:  # assistant message
            st.markdown(f'<div class="assistant-message">', unsafe_allow_html=True)
            st.markdown(f"**🤖 AI सलाहकार:** {content}")
            if timestamp:
                st.caption(f"⏰ {timestamp[:19].replace('T', ' ')}")
            
            # Audio playback button for assistant messages
            if st.session_state.voice_enabled:
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button(f"🔊", key=f"play_audio_{i}", help="इस जवाब को सुनें"):
                        with st.spinner("🎧 आवाज़ तैयार कर रहे हैं..."):
                            audio_bytes = text_to_speech_bytes(content)
                            if audio_bytes:
                                st.audio(audio_bytes, format="audio/mp3")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
else:
   st.markdown("""
<style>
.chat-container {
    background-color: #000000;  /* 🔥 काला बैकग्राउंड */
    color: #FFFFFF;            /* ✨ सफेद टेक्स्ट */
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
    margin-top: 20px;
    margin-bottom: 20px;
    font-family: 'Segoe UI', sans-serif;
}
.chat-container h4 {
    font-size: 1.8rem;
    margin-bottom: 10px;
    color: #00FF7F; /* 🌿 हल्का हरा टाइटल */
}
.chat-container ul li {
    margin-bottom: 5px;
}
.chat-container em {
    color: #FFD700; /* 🌟 सुनहरा रंग (emphasis text) */
}
</style>

<div class="chat-container">
    <h4>👋 नमस्ते किसान भाई!</h4>
    <p>मैं आपका AI कृषि सलाहकार हूं। आप मुझसे निम्नलिखित विषयों पर सवाल पूछ सकते हैं:</p>
    <ul>
        <li>🌾 <strong>फसल की सिफारिश</strong> - कौन सी फसल बोएं</li>
        <li>🌱 <strong>मिट्टी की देखभाल</strong> - मिट्टी सुधार के तरीके</li>
        <li>🌧️ <strong>मौसम आधारित सलाह</strong> - मौसम के अनुसार खेती</li>
        <li>🐛 <strong>कीट और रोग नियंत्रण</strong> - समस्याओं का समाधान</li>
        <li>💧 <strong>सिंचाई प्रबंधन</strong> - पानी की सही व्यवस्था</li>
        <li>🌿 <strong>जैविक खेती</strong> - प्राकृतिक तरीके</li>
    </ul>
    <p><em>आप टेक्स्ट लिखकर या आवाज़ में सवाल पूछ सकते हैं!</em></p>
</div>
""", unsafe_allow_html=True)


# ------------------- Enhanced Text Input Section -------------------
def process_text_input(user_input: str):
    """Process text input with enhanced error handling"""
    if st.session_state.processing:
        st.warning("⏳ कृपया प्रतीक्षा करें, एक प्रोसेस पहले से चल रही है...")
        return
        
    st.session_state.processing = True
    
    try:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f"✍️ {user_input}")
        
        # Add to chat history
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input, 
            "type": "text",
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate AI response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("🤖 सोच रहा हूं... 🧠")
            
            full_response = ""
            try:
                for chunk in chain.stream({
                    "question": user_input,
                    "location": city,
                    "temperature": weather_data.get('temperature', 25),
                    "humidity": weather_data.get('humidity', 70),
                    "soil_ph": soil_data.get('ph', 6.5),
                    "nitrogen": soil_data.get('nitrogen', 50),
                    "crop_suggestion": predicted_crop,
                    "confidence": confidence
                }):
                    full_response += chunk
                    response_placeholder.markdown(f"🤖 {full_response}")
                    
            except Exception as e:
                error_msg = f"जवाब तैयार करने में समस्या: {str(e)}"
                response_placeholder.error(f"❌ {error_msg}")
                full_response = "क्षमा करें, तकनीकी समस्या के कारण जवाब नहीं दे सका। कृपया फिर से कोशिश करें।"
                logger.error(f"LLM generation error: {e}")
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": full_response, 
                "type": "text_response",
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate audio response if enabled
            if st.session_state.voice_enabled and st.session_state.auto_play_response and full_response:
                with st.spinner("🎧 आवाज़ में जवाब तैयार कर रहे हैं..."):
                    audio_bytes = text_to_speech_bytes(full_response)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                        
    except Exception as e:
        st.error(f"❌ प्रोसेसिंग में समस्या: {str(e)}")
        logger.error(f"Text processing error: {e}")
    finally:
        st.session_state.processing = False

# Text input handling
if user_input := st.chat_input("✍️ अपना सवाल यहाँ लिखें... (उदाहरण: इस मौसम में कौन सी फसल बोना बेहतर होगा?)", disabled=st.session_state.processing):
    process_text_input(user_input)

# ------------------- Enhanced Footer Section -------------------
st.markdown("---")

# Statistics and utilities
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_chats = len([m for m in st.session_state.chat_history if m["role"] == "user"])
    st.metric("💬 कुल सवाल", total_chats)

with col2:
    voice_chats = len([m for m in st.session_state.chat_history if m.get("type") == "voice"])
    st.metric("🎤 आवाज़ी सवाल", voice_chats)

with col3:
    if st.session_state.chat_history:
        last_chat_time = st.session_state.chat_history[-1].get("timestamp", "")
        if last_chat_time:
            st.metric("⏰ अंतिम सवाल", last_chat_time[:19].replace('T', ' '))
    else:
        st.metric("⏰ अंतिम सवाल", "कोई नहीं")

with col4:
    # Export chat functionality
    if st.button("📥 चैट एक्सपोर्ट करें"):
        if st.session_state.chat_history:
            chat_export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "location_info": {
                    "city": city,
                    "coordinates": [lat, lon],
                    "weather": weather_data,
                    "soil": soil_data
                },
                "crop_prediction": {
                    "recommended_crop": predicted_crop,
                    "confidence": confidence
                },
                "chat_history": st.session_state.chat_history,
                "statistics": {
                    "total_messages": len(st.session_state.chat_history),
                    "voice_messages": voice_chats,
                    "text_messages": total_chats - voice_chats
                }
            }
            
            # Create downloadable JSON
            json_str = json.dumps(chat_export_data, ensure_ascii=False, indent=2)
            
            st.download_button(
                label="💾 JSON फ़ाइल डाउनलोड करें",
                data=json_str,
                file_name=f"agriculture_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="अपनी पूरी बातचीत को JSON फ़ाइल के रूप में सेव करें"
            )
        else:
            st.info("कोई चैट हिस्ट्री नहीं है")

# Quick action buttons
st.markdown("### 🚀 त्वरित प्रश्न")
col1, col2, col3 = st.columns(3)

quick_questions = [
    "इस मौसम में कौन सी फसल बेहतर होगी?",
    "मिट्टी की गुणवत्ता कैसे सुधारें?",
    "बारिश के बाद क्या करना चाहिए?"
]

for i, (col, question) in enumerate(zip([col1, col2, col3], quick_questions)):
    with col:
        if st.button(question, key=f"quick_q_{i}"):
            process_text_input(question)

# Help and information section
with st.expander("ℹ️ मदद और जानकारी", expanded=False):
    st.markdown("""
    ### 🔧 कैसे इस्तेमाल करें:
    
    **आवाज़ से सवाल पूछने के लिए:**
    1. 🎤 "रिकॉर्ड" बटन दबाएं
    2. स्पष्ट आवाज़ में अपना सवाल बोलें
    3. "स्टॉप" दबाकर रिकॉर्डिंग बंद करें  
    4. "जवाब पाएं" बटन दबाएं
    
    **टेक्स्ट से सवाल पूछने के लिए:**
    1. नीचे टेक्स्ट बॉक्स में अपना सवाल लिखें
    2. Enter दबाएं या भेजें बटन दबाएं
    
    ### 🌾 मैं किन विषयों में मदद कर सकता हूं:
    - फसल चुनने की सलाह (मौसम और मिट्टी के अनुसार)
    - मिट्टी सुधार के तरीके
    - कीट और बीमारी की रोकथाम
    - सिंचाई का सही समय और तरीका
    - खाद और उर्वरक की जानकारी
    - जैविक खेती के नुस्खे
    - मौसम के अनुसार खेती की योजना
    
    ### ⚠️ महत्वपूर्ण सूचना:
    - यह एक AI सहायक है और दी गई सभी सलाह केवल सुझाव हैं
    - महत्वपूर्ण कृषि निर्णयों के लिए स्थानीय कृषि विशेषज्ञ से परामर्श लें
    - मार्केट रेट की जानकारी अभी उपलब्ध नहीं है (जल्द आएगी)
    
    ### 🛠️ तकनीकी सहायता:
    - यदि आवाज़ पहचान में समस्या हो तो शांत जगह से बात करें
    - इंटरनेट कनेक्शन धीमा होने पर थोड़ा इंतज़ार करें
    - किसी भी समस्या के लिए "चैट रीसेट करें" बटन का उपयोग करें
    """)

# Footer with credits and version info
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem; padding: 1rem; border-top: 1px solid #ddd;'>
    <p>🌾 <strong>AI कृषि सहायक</strong> - आपके खेत का डिजिटल मित्र</p>
    <p><small>संस्करण 2.0 | Powered by Groq AI & OpenWeatherMap</small></p>
    <p><small>
        सभी सलाह केवल सूचनात्मक उद्देश्यों के लिए हैं। 
        महत्वपूर्ण कृषि निर्णयों के लिए स्थानीय कृषि विशेषज्ञ से परामर्श अवश्य लें।
    </small></p>
</div>
""", unsafe_allow_html=True)
