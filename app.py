"""Watt & Wait dashboard: original workplace PCDR plus EV predictive tools."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import DEFAULT_DATA_PATH, load_ev_sessions
from src.feature_engineering import DRIVING_TELEMETRY_FEATURES, HEALTH_FEATURES, MODEL_FEATURES, RANGE_FEATURES, build_live_features, classify_driving_mode
from src.nudge_logic import NUDGE_FEATURES, nudge_priority_score, risk_from_pcdr, risk_series, smart_nudge_score
from src.strategy_engine import recommend_target_soc
from src.train_multitarget_models import DEGRADATION_MODEL_PATH, RANGE_MODEL_PATH
from src.train_nudge_model import DEFAULT_NUDGE_MODEL_PATH
from src.train_pcdr_model import DEFAULT_MODEL_PATH

ROOT = Path(__file__).resolve().parent
TELEMETRY_PATH = ROOT / "data" / "ieee_vehicle_telemetry.csv"
RISK_COLORS = {"Low": "#16a34a", "Medium": "#d97706", "High": "#dc2626"}
st.set_page_config(page_title="Watt & Wait | EV Intelligence", page_icon="⚡", layout="wide")
st.markdown("""<style>
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg,#f8fafc,#ecfdf5); color:#172033;}
[data-testid="stSidebar"] {background:#062c2b;} [data-testid="stSidebar"] * {color:#e7f8f4 !important;}
.hero {padding:1.8rem 2rem;border-radius:22px;background:linear-gradient(110deg,#082f49,#0f766e);color:white;margin-bottom:1.25rem;box-shadow:0 16px 38px #0f172a2b;}
.hero h1 {color:#fff;margin:0;font-size:2.5rem}.hero p {color:#ccfbf1;margin:.45rem 0 0}.eyebrow{color:#99f6e4;font-weight:700;letter-spacing:.1em;font-size:.75rem;text-transform:uppercase}
[data-testid="stMetric"] {background:#fff;border:1px solid #dbe7e3;border-radius:15px;padding:1rem;box-shadow:0 5px 18px #0f172a0d}
.card {background:#fff;border:1px solid #dbe7e3;border-radius:16px;padding:1.25rem;margin:.5rem 0;box-shadow:0 5px 18px #0f172a0d}
.risk {padding:1rem;border-radius:12px;color:white;font-weight:700;font-size:1.05rem}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all persisted model bundles once per Streamlit process."""
    return (joblib.load(DEFAULT_MODEL_PATH), joblib.load(DEFAULT_NUDGE_MODEL_PATH), joblib.load(RANGE_MODEL_PATH), joblib.load(DEGRADATION_MODEL_PATH))

@st.cache_data
def sessions(): return load_ev_sessions(DEFAULT_DATA_PATH)

@st.cache_data
def driving_telemetry():
    """Load IEEE-style trip telemetry, with a deterministic synthetic fallback."""
    if TELEMETRY_PATH.exists():
        frame = pd.read_csv(TELEMETRY_PATH)
        if set(DRIVING_TELEMETRY_FEATURES).issubset(frame.columns):
            frame[DRIVING_TELEMETRY_FEATURES] = frame[DRIVING_TELEMETRY_FEATURES].apply(pd.to_numeric, errors="coerce")
            frame = frame.dropna(subset=DRIVING_TELEMETRY_FEATURES)
            if not frame.empty:
                return frame.reset_index(drop=True)
        st.warning("Telemetry schema is incomplete; using synthetic IEEE-style sample data.")
    rng = np.random.default_rng(42); n = 100
    return pd.DataFrame({"session_id":[f"SYN-{i+1:04d}" for i in range(n)], "user_id":[f"USR-{i%20+1:04d}" for i in range(n)], "vehicle_id":[f"VEH-{i%20+1:04d}" for i in range(n)], "trip_distance_km":rng.uniform(12,220,n), "avg_speed_kmph":rng.uniform(25,105,n), "max_acceleration_ms2":rng.uniform(.6,3.4,n), "hard_braking_count":rng.poisson(1.2,n), "idle_time_percentage":rng.uniform(0,22,n), "rpm_or_throttle_variance":rng.uniform(.05,.8,n)})

needed = [DEFAULT_MODEL_PATH, DEFAULT_NUDGE_MODEL_PATH, RANGE_MODEL_PATH, DEGRADATION_MODEL_PATH]
if not all(path.exists() for path in needed):
    st.error("Model artifacts are not available. Run the training commands in README.md."); st.stop()
pcdr_bundle, nudge_bundle, range_bundle, health_bundle = load_models()

with st.sidebar:
    st.markdown("## ⚡ Watt & Wait")
    st.caption("EV intelligence workspace")
    page = st.radio("Navigate", ["Workplace station management", "Range & endurance", "Battery health", "Smart recommendation"], label_visibility="collapsed")
    st.divider(); st.caption("All predictions are estimates for operational planning, not safety-critical vehicle controls.")

st.markdown("""<div class="hero"><div class="eyebrow">EV charging operations</div><h1>Watt &amp; Wait</h1><p>Make every charge session more useful—from workplace dwell decisions to battery-aware travel planning.</p></div>""", unsafe_allow_html=True)

if page == "Workplace station management":
    st.subheader("Live decision")
    st.caption("Original PCDR prediction and congestion-aware smart nudging workflow.")
    users, stations = sorted(pcdr_bundle["user_profiles"]), sorted(pcdr_bundle["station_rates"])
    with st.container(border=True):
        a,b,c,d = st.columns(4)
        user_id = a.selectbox("User ID", users); station_id = b.selectbox("Station ID", stations)
        energy_kwh = c.slider("Energy required (kWh)", 2., 80., 24., .5); occupancy = d.slider("Current station occupancy (%)", 0, 100, 72)
        e,f = st.columns(2); arrival_date = e.date_input("Arrival date", datetime.now().date()); arrival_time = f.time_input("Arrival time", datetime.now().replace(second=0,microsecond=0).time())
    arrival = pd.Timestamp(datetime.combine(arrival_date, arrival_time))
    live = build_live_features(user_id=user_id, station_id=station_id, arrival_time=arrival, energy_kwh=energy_kwh, user_type="Employee", department="Operations", user_profiles=pcdr_bundle["user_profiles"], station_rates=pcdr_bundle["station_rates"], global_pcdr=pcdr_bundle["global_pcdr"], global_rate_kw=pcdr_bundle["global_rate_kw"])
    pcdr = float(np.clip(pcdr_bundle["pipeline"].predict(live[MODEL_FEATURES])[0],0,1)); risk = risk_from_pcdr(pcdr); congestion = occupancy/100; priority = nudge_priority_score(pcdr, congestion)
    nf = live[[c for c in NUDGE_FEATURES if c in live]].copy(); nf.insert(0,"predicted_pcdr",pcdr); nf.insert(1,"station_congestion_factor",congestion); nf=nf[NUDGE_FEATURES]
    classifier = nudge_bundle["responsiveness_classifier"]; positive = list(classifier.named_steps["classifier"].classes_).index(1); response = float(classifier.predict_proba(nf)[0,positive]); smart = smart_nudge_score(priority,response)
    rate=float(live.at[0,"avg_charging_rate_kw"]); active=energy_kwh/max(rate,.1); idle=active*min(pcdr,.95)/max(1-pcdr,.05); completion=arrival+pd.to_timedelta(active,unit="h")
    a,b,c,d=st.columns(4); a.metric("Predicted PCDR",f"{pcdr:.1%}"); b.metric("Estimated idle time",f"{idle:.1f} hr"); c.metric("Nudge priority",f"{priority:.0%}"); d.metric("Response probability",f"{response:.0%}")
    left,right=st.columns([1,2])
    with left:
        st.markdown(f'<div class="risk" style="background:{RISK_COLORS[risk]}">{risk} dwell risk · {pcdr:.1%} PCDR</div>',unsafe_allow_html=True)
        st.progress(priority,text=f"Base priority {priority:.0%}"); st.progress(smart,text=f"Responsiveness-adjusted {smart:.0%}")
    with right:
        if risk=="High" and congestion>.70: st.error(f"Schedule an automated reminder at {completion.strftime('%I:%M %p').lstrip('0')}. High dwell risk meets the congestion alert threshold.")
        elif risk=="High": st.warning("High dwell risk: add this driver to the watchlist; congestion has not crossed 70%.")
        else: st.success("No immediate automated nudge. Continue monitoring occupancy and charge completion.")
        st.markdown(f'<div class="card"><b>Estimated charge completion:</b> {completion.strftime("%d %b %Y, %I:%M %p")}<br><b>Station rate assumption:</b> {rate:.1f} kW<br><b>Estimated total connection:</b> {active+idle:.1f} hours</div>',unsafe_allow_html=True)
    st.divider(); st.subheader("Fleet context")
    historic=sessions().copy(); historic["risk_level"]=risk_series(historic.pcdr)
    fig=px.histogram(historic,x="pcdr",color="risk_level",nbins=30,color_discrete_map=RISK_COLORS,category_orders={"risk_level":["Low","Medium","High"]},title="Historical PCDR distribution")
    fig.add_vline(x=pcdr,line_dash="dash",annotation_text="Live session"); fig.update_layout(template="plotly_white",xaxis_tickformat=".0%",height=360); st.plotly_chart(fig,width="stretch")

elif page == "Range & endurance":
    st.subheader("Range & endurance predictor")
    st.caption("Estimate available driving distance and operating time using driving behavior and current vehicle conditions.")

    st.markdown("#### Automated driving mode")
    st.caption("Adjust the trip physics inputs to see the mode classification update immediately.")
    a,b,c,d=st.columns(4)
    acceleration=a.slider("Maximum acceleration (m/s^2)",0.5,4.0,1.8,0.1)
    braking=b.slider("Hard braking events",0,10,1)
    speed=c.slider("Average speed (km/h)",20,120,50)
    variance=d.slider("Throttle variance",0.05,0.80,0.20,0.01)
    mode=classify_driving_mode({"avg_speed_kmph":speed,"max_acceleration_ms2":acceleration,"hard_braking_count":braking,"idle_time_percentage":0.0,"rpm_or_throttle_variance":variance})
    mode_color={"Eco":"#16a34a","Normal":"#2563eb","Sport":"#ea580c"}[mode]
    st.markdown(f'<div style="display:inline-flex;align-items:center;gap:.6rem;padding:.45rem .75rem;border:1px solid #dbe7e3;border-radius:12px;background:#fff;margin:.15rem 0 .5rem"><span style="font-size:.72rem;color:#64748b;letter-spacing:.06em">AUTOMATIC MODE</span><b style="color:{mode_color};font-size:.95rem">{mode}</b></div>',unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Range & endurance estimate")
    a,b,c=st.columns(3)
    soc=a.slider("Current SOC (%)",5,100,80)
    capacity=b.slider("Battery capacity (kWh)",30.,120.,75.)
    temp=c.slider("Ambient temperature (deg C)",-10,50,25)
    mode_efficiency={"Eco":.9,"Normal":1.0,"Sport":1.2}[mode]
    efficiency=.165*mode_efficiency*(1+abs(temp-22)*.004)
    row=pd.DataFrame([[soc,capacity,efficiency,temp,speed]],columns=RANGE_FEATURES)
    distance=max(0.0,float(range_bundle["pipeline"].predict(row)[0]))
    endurance=distance/max(speed,1)
    a,b=st.columns(2)
    a.metric("Predicted driving range",f"{distance:.0f} km")
    b.metric("Operating endurance",f"{endurance:.1f} hours")

    if mode == "Sport":
        eco_row=row.copy(); eco_row["efficiency_kwh_per_km"]=.165*.9*(1+abs(temp-22)*.004)
        eco_distance=max(0.0,float(range_bundle["pipeline"].predict(eco_row)[0]))
        uplift=max(0.0,(eco_distance-distance)/max(distance,1)*100)
        st.info(f"Try smoother acceleration and fewer hard braking events to work toward Eco mode. At these conditions, the model estimates up to {uplift:.0f}% more range in Eco mode.")
    elif mode == "Eco":
        st.info("Your current inputs classify as Eco. Keep acceleration and braking smooth to preserve this range-friendly driving pattern.")
    else:
        st.info("Your inputs classify as Normal. Lower acceleration peaks and fewer hard braking events may move the trip toward Eco mode and improve range.")
    st.info("For routine daily charging, consider an 80% target to reduce time spent at high state of charge; charge higher when your trip requires the extra range.")

elif page == "Battery health":
    st.subheader("Battery health & degradation analytics"); st.caption("Estimate State of Health from charge cycles, fast charging, and thermal exposure.")
    a,b,c,d=st.columns(4); cycles=a.slider("Charge cycles",0,2500,600); fast=b.slider("Fast charging (%)",0,100,35)/100; temperature=c.slider("Average battery temperature (°C)",10,60,30); avg_soc=d.slider("Average maintained SOC (%)",20,100,70)
    row=pd.DataFrame([[cycles,fast,temperature,avg_soc,temperature*fast]],columns=HEALTH_FEATURES); soh=float(np.clip(health_bundle["pipeline"].predict(row)[0],0,100)); annual=max(.1,(100-soh)/max(cycles/250,1)*.7+fast*1.1+max(temperature-30,0)*.05)
    a,b=st.columns(2); a.metric("Predicted remaining SoH",f"{soh:.1f}%"); b.metric("Estimated annual degradation",f"{annual:.2f}%")
    (st.warning if fast>=.5 else st.success)("High fast-charge usage increases thermal stress and battery ageing." if fast>=.5 else "Fast-charge exposure is in the lower-impact range.")

else:
    st.subheader("Unified smart recommendation"); st.caption("Balance daily range needs, workplace dwell risk, and battery protection.")
    a,b,c,d=st.columns(4); need=a.slider("Expected daily driving (km)",10,700,180); range_80=b.slider("Estimated range at 80% (km)",50,600,260); dwell=c.selectbox("Workplace dwell risk",["Low","Medium","High"]); soh=d.slider("Battery SoH (%)",50,100,90)
    fast=st.slider("Fast charging usage (%)",0,100,35)/100; cutoff,reason=recommend_target_soc(need,range_80,dwell,soh,fast)
    st.metric("Recommended charge cutoff",f"{cutoff}% SOC"); st.markdown(f'<div class="card"><b>Recommendation:</b> {reason}<br><br>80% range buffer: <b>{range_80} km</b> · Daily need: <b>{need} km</b> · Dwell risk: <b>{dwell}</b></div>',unsafe_allow_html=True)
