
import streamlit as st
import streamlit.components.v1 as components
import base64
import re

# NOTE: This app requires Praat-Parselmouth for F0/Intensity/Pitch-range extraction.
# Streamlit Cloud will raise ModuleNotFoundError unless you add it to requirements.txt:
#   praat-parselmouth
try:
    import parselmouth
    from parselmouth.praat import call
except ModuleNotFoundError as e:
    st.error(
        "필수 패키지(parselmouth)가 설치되지 않아 실행할 수 없습니다.\n\n"
        "Streamlit Cloud를 사용 중이면, GitHub 레포에 requirements.txt를 만들고 아래 한 줄을 추가한 뒤 재배포하세요:\n"
        "- praat-parselmouth\n\n"
        "(이미 requirements.txt가 있다면, 거기에 추가하면 됩니다.)"
    )
    st.stop()
import numpy as np
import math
import pandas as pd
import os
import datetime
import io
import html
from pathlib import Path

# Optional (cloud + email)
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

try:
    import gspread
    from google.oauth2 import service_account
    HAS_GSPREAD = True
except Exception:
    HAS_GSPREAD = False

# =========================
# Page config
# =========================
st.set_page_config(page_title="PD 음성 평가(평가판)", layout="wide")

# Anchor for programmatic scrolling
st.markdown('<div id="top"></div>', unsafe_allow_html=True)
if st.session_state.get("scroll_to_top", False):
    components.html("""<script>
const el = window.parent.document.getElementById("top");
if(el){ el.scrollIntoView({behavior: "auto", block: "start"}); }
else{ window.parent.scrollTo(0,0); }
</script>""", height=0)
    st.session_state.scroll_to_top = False


# --- Prevent duplicate submissions in the same browser session ---
def make_submission_key(wav_path: str, patient_info: dict) -> str:
    """Create a stable-ish key for the current recording to prevent duplicate sends."""
    try:
        mtime = os.path.getmtime(wav_path) if wav_path and os.path.exists(wav_path) else 0.0
        size = os.path.getsize(wav_path) if wav_path and os.path.exists(wav_path) else 0
    except Exception:
        mtime, size = 0.0, 0
    p = patient_info or {}
    name = str(p.get("name", "")).strip()
    age = str(p.get("age", "")).strip()
    gender = str(p.get("gender", "")).strip()
    return f"{os.path.basename(wav_path)}|{mtime:.3f}|{size}|{name}|{age}|{gender}"

if "sent_submission_keys" not in st.session_state:
    st.session_state["sent_submission_keys"] = set()

def reset_for_new_evaluation():
    """Reset state for a brand-new participant/evaluation (keeps app running without refreshing the page)."""
    keys_to_clear = [
        "enrolled", "show_instructions", "patient_info",
        "wav_path", "analysis",
        "vhi_total", "vhi_f", "vhi_p", "vhi_e",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    # Clear VHI responses
    for i in range(1, 11):
        kk = f"vhi_q{i}"
        if kk in st.session_state:
            del st.session_state[kk]
    # Allow re-send keys to remain; they are recording-specific.
    st.session_state["enrolled"] = False
    st.session_state["show_instructions"] = False
    st.rerun()

st.title("🧠 파킨슨병(PD) 음성 평가(평가판)")

# =========================
# Fixed reading paragraph (84 syllables)
# =========================
READING_TITLE = "조음 정밀 (84음절)"
READING_TEXT = """바닷가 부둣가 바닥에 비둘기 바둑이 본다, 다시 걷는다.
달·딸·탈, 바·빠·파, 가·까·카를 같은 박자로 끊지 말고 잇는다.
사과를 싸서 씻고, 조심히 찾아 차분히 웃는다.
노란 물 멀리 두고 말로 마무리하며 느리게 내려놓는다."""
TOTAL_SYLLABLES_FIXED = 84

def styled_text(text: str, size: int) -> str:
    safe = html.escape(str(text)).replace("\n", "<br>")
    return f"""
    <div style="font-size:{int(size)}px; line-height: 1.6; padding: 12px; border-radius: 8px;
                background-color: #f9f9f9; color:#333; border: 1px solid #eee;">
        {safe}
    </div>
    """

# =========================
# Training data load (Step2 model: intensity + SPS only)
# =========================
TRAINING_DATA_CSV_EMBED = """환자ID,성별,F0,Range,강도(dB),SPS,음도(청지각),음도범위(청지각),강도(청지각),말속도(청지각),조음정확도(청지각),VHI총점,VHI_신체,VHI_기능,VHI_정서,진단결과 (Label)
PD1,여,193.6,137.78,56.57,4.56,52.78,48.11,35.56,49.22,65.33,90,30,30,30,PD_Intensity
PD2,여,198.38,75.89,49.51,4.24,21.78,22.78,5.78,63.78,23.33,70,23,22,25,PD_Intensity
PD3,남,137.07,93.55,62.66,4.63,41.89,47.33,54.22,69.22,47.78,51,19,18,14,PD_Rate
PD4,여,155.45,56.67,53.1,3.1,26.22,28.22,26.78,43.44,28.89,109,36,36,37,PD_Intensity
PD5,남,125.52,106.84,60.02,3.29,42.78,49.22,45.44,44.56,36.78,48,18,15,15,PD_Articulation
PD6,남,179.69,151.93,67.91,3.97,53,63.33,69.89,55.44,26.44,58,23,19,16,PD_Articulation
PD7,여,126.97,69.55,51.78,3.26,22.78,17,12.78,40.78,20.33,116,36,40,40,PD_Intensity
PD8,여,169.32,105.57,56.26,4.42,46.89,47.78,34.78,50.56,61.11,68,23,22,23,PD_Intensity
PD9,남,114.93,54.89,55.03,4.58,24.56,18.11,19.44,66.44,23.78,37,14,13,10,PD_Rate
PD10,남,122.54,78.4,58.81,3.36,33.89,37.89,31.78,39.56,60.56,36,16,10,10,PD_Intensity
PD11,남,113.83,92.63,59.85,3.93,43.56,33.11,63.22,58.67,45.11,55,23,19,13,PD_Articulation
PD12,남,124.23,88.15,57.35,3.26,43.56,53.56,49.89,47.56,60.56,66,23,24,19,PD_Articulation
PD13,남,138.56,102.52,63.63,7.03,48.22,34.67,60.22,92.33,37.11,96,24,35,37,PD_Rate
PD14,여,198.33,68.22,50.58,3.97,29.44,13,6.78,40.44,24.89,87,27,30,30,PD_Intensity
PD15,남,131.23,91.37,58.75,3.55,52.67,56.33,58.11,68.33,71.89,33,15,11,7,PD_Intensity
PD16,여,189.72,111.82,62.57,2.64,55,35.44,61.44,59,61.56,57,21,18,18,PD_Intensity
PD17,여,165.65,139.99,51.45,3.78,61.56,63.11,46.44,58.67,77.67,30,9,11,10,PD_Articulation
PD18,여,154.43,103.33,52.59,3.59,41.33,35.78,27.67,52.11,49.67,60,20,20,20,PD_Intensity
PD19,남,154.52,112.58,60.23,2.97,36,40,22,55,19,86,26,29,31,PD_Articulation
PD20,여,198.38,120.44,60.32,4.85,52.78,48.89,36.33,66.22,59.67,58,20,16,22,PD_Intensity
"""

def get_training_csv_path() -> Path | None:
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidates = [
        base / "training_data.csv",
        Path.cwd() / "training_data.csv",
        Path("/mount/src/pd-voice-diagnosis/training_data.csv"),
        Path("/mnt/data/training_data.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def load_training_df() -> pd.DataFrame:
    p = get_training_csv_path()
    if p is None:
        return pd.read_csv(io.StringIO(TRAINING_DATA_CSV_EMBED))
    return pd.read_csv(p)

def train_step2_intensity_sps(df: pd.DataFrame):
    df = df.copy()
    df = df[df["진단결과 (Label)"].astype(str).str.startswith("PD_")]

    X = df[["강도(dB)", "SPS"]].astype(float).values
    y = df["진단결과 (Label)"].astype(str).values

    classes = np.unique(y)
    means = {c: X[y == c].mean(axis=0) for c in classes}

    Xc = X - np.vstack([means[yy] for yy in y])
    cov = np.cov(Xc.T, bias=False)
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(cov)

    priors = {c: float(np.mean(y == c)) for c in classes}
    return {"classes": classes, "means": means, "inv_cov": inv_cov, "priors": priors}

def predict_step2(model, intensity_db: float, sps: float):
    if model is None or (not np.isfinite(intensity_db)) or (not np.isfinite(sps)):
        return None, None

    x = np.array([float(intensity_db), float(sps)], dtype=float)
    classes = model["classes"]
    inv_cov = model["inv_cov"]
    priors = model["priors"]
    means = model["means"]

    scores = []
    for c in classes:
        mu = means[c]
        s = float(x @ inv_cov @ mu - 0.5 * (mu @ inv_cov @ mu) + np.log(max(priors.get(c, 1e-9), 1e-9)))
        scores.append(s)
    scores = np.array(scores, dtype=float)

    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / max(np.sum(probs), 1e-12)

    pairs = sorted(zip(classes, probs), key=lambda z: float(z[1]), reverse=True)
    top1, p1 = pairs[0][0], float(pairs[0][1])
    top2, p2 = (pairs[1][0], float(pairs[1][1])) if len(pairs) > 1 else (None, 0.0)

    mixed = (top2 is not None) and ((p1 - p2) < 0.20) and (p2 >= 0.25)
    final = f"{top1}+{top2}" if mixed else top1
    return final, dict(pairs)

try:
    _df_train = load_training_df()
    STEP2_MODEL = train_step2_intensity_sps(_df_train)
except Exception as e:
    STEP2_MODEL = None
    st.warning(f"⚠️ Step2(집단) 모델 로드 실패: {type(e).__name__}: {e}")

# =========================
# Google Sheet + Email logging
# =========================
SHEET_NAME = st.secrets.get("sheet", {}).get("name", None)


def _json_safe_value(v):
    """Convert values to JSON/GSheets-safe primitives (avoid NaN/Inf)."""
    if v is None:
        return ""
    # numpy scalars
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, (float,)):
        if (not math.isfinite(v)) or math.isnan(v):
            return ""
        return float(v)
    if isinstance(v, (int, bool)):
        return v
    # allow short strings as-is
    return str(v)

def _json_safe_row(row):
    return [_json_safe_value(v) for v in row]

def send_email_and_log_sheet(wav_path: str, patient_info: dict, analysis: dict, final_diag: str):
    """Send wav to research email and append a row to Google Sheet.
    Returns: (log_filename, sheet_ok, sheet_msg, email_ok, email_msg)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Build a safe filename label for logging/email
    raw_name = str(patient_info.get("name", "participant"))
    safe_name = re.sub(r"[^0-9A-Za-z가-힣_\-]+", "", raw_name.replace(" ", "")) or "participant"
    log_prefix = "TEST_" if patient_info.get("is_test") else ""
    log_filename = f"{log_prefix}{safe_name}_{patient_info.get('age','')}_{patient_info.get('gender','')}_{timestamp}.wav"

    # --- Google Sheet ---
    sheet_ok = False
    sheet_msg = ""
    if HAS_GSPREAD and ("gcp_service_account" in st.secrets) and (SHEET_NAME is not None):
        try:
            # Streamlit secrets may store newlines as literal "\n". Google auth expects real newlines.
            svc_info = dict(st.secrets["gcp_service_account"])
            if "private_key" in svc_info and isinstance(svc_info["private_key"], str):
                svc_info["private_key"] = svc_info["private_key"].replace("\\n", "\n")

            creds = service_account.Credentials.from_service_account_info(
                svc_info,
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive",
                ],
            )
            gc = gspread.authorize(creds)
            sh = gc.open(SHEET_NAME)

            # Use first worksheet by default (or configured name)
            worksheet_name = st.secrets.get("sheet", {}).get("worksheet", None)
            worksheet = sh.worksheet(worksheet_name) if worksheet_name else sh.sheet1

            header = [
                "timestamp", "filename",
                "name", "age", "gender",
                "diag_years", "dopa_meds", "hearing_issue", "device",
                "F0", "range", "intensity_dB", "SPS",
                "VHI-total", "VHI_F", "VHI_P", "VHI_E",
                "Final diagnosis",
            ]

            existing = worksheet.row_values(1)
            if existing != header:
                # Overwrite row1 to keep header consistent (avoid multiple header rows).
                worksheet.update("A1", [header])

            row = [
                timestamp,
                log_filename,
                patient_info.get("name", ""),
                patient_info.get("age", ""),
                patient_info.get("gender", ""),
                patient_info.get("diag_years", ""),
                patient_info.get("dopa_meds", ""),
                patient_info.get("hearing_issue", ""),
                patient_info.get("device", ""),
                analysis.get("f0", ""),
                analysis.get("range", ""),
                analysis.get("intensity_db", ""),
                analysis.get("sps", ""),
                analysis.get("vhi_total", ""),
                analysis.get("vhi_f", ""),
                analysis.get("vhi_p", ""),
                analysis.get("vhi_e", ""),
                final_diag or "",
            ]
            row = _json_safe_row(row)
            worksheet.append_row(row)
            sheet_ok = True
            sheet_msg = "구글시트 저장 성공"
        except Exception as e:
            sheet_ok = False
            sheet_msg = f"구글시트 저장 실패: {type(e).__name__}: {e}"
    else:
        sheet_ok = False
        sheet_msg = "구글시트 저장 생략(Secrets 미설정)"

    # --- Email ---
    email_ok = False
    email_msg = ""
    try:
        sender = st.secrets["email"]["sender"]
        password = st.secrets["email"]["password"]
        receiver = st.secrets["email"]["receiver"]

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = receiver
        msg["Subject"] = f"[PD Pilot] {log_filename}"

        body = f"""[PD Pilot - New Sample]
timestamp: {timestamp}
filename: {log_filename}

name: {patient_info.get('name','')}
age: {patient_info.get('age','')}
gender: {patient_info.get('gender','')}

diag_years: {patient_info.get('diag_years','')}
dopamine_meds: {patient_info.get('dopa_meds','')}
hearing_issue: {patient_info.get('hearing_issue','')}

device: {patient_info.get('device','')}
mic: {patient_info.get('mic','')}
distance_30cm_confirmed: {patient_info.get('distance_ok','')}

F0: {analysis.get('f0','')}
range: {analysis.get('range','')}
intensity_dB: {analysis.get('intensity_db','')}
SPS: {analysis.get('sps','')}

VHI_total: {analysis.get('vhi_total','')}
VHI_F: {analysis.get('vhi_f','')}
VHI_P: {analysis.get('vhi_p','')}
VHI_E: {analysis.get('vhi_e','')}

final_diagnosis(model): {final_diag}
""".strip()
        msg.attach(MIMEText(body, "plain"))

        with open(wav_path, "rb") as f:
            part = MIMEBase("audio", "wav")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={log_filename}")
        msg.attach(part)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()

        email_ok = True
        email_msg = "이메일 전송 성공"
    except KeyError:
        email_ok = False
        email_msg = "이메일 전송 생략(Secrets 미설정)"
    except Exception as e:
        email_ok = False
        email_msg = f"이메일 전송 실패: {type(e).__name__}: {e}"

    return log_filename, sheet_ok, sheet_msg, email_ok, email_msg

# -------------------------
# Duplicate participation guard (Google Sheet-based, best-effort)
# -------------------------
KST = datetime.timezone(datetime.timedelta(hours=9))

def _kst_now() -> datetime.datetime:
    return datetime.datetime.now(tz=KST)

def _get_sheet_worksheet():
    """Return a gspread worksheet object if configured; otherwise raise."""
    if not (HAS_GSPREAD and ("gcp_service_account" in st.secrets) and (SHEET_NAME is not None)):
        raise RuntimeError("Sheets secrets not configured")
    svc_info = dict(st.secrets["gcp_service_account"])
    if "private_key" in svc_info and isinstance(svc_info["private_key"], str):
        svc_info["private_key"] = svc_info["private_key"].replace("\\n", "\n")
    creds = service_account.Credentials.from_service_account_info(
        svc_info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    gc = gspread.authorize(creds)
    sh = gc.open(SHEET_NAME)
    worksheet_name = st.secrets.get("sheet", {}).get("worksheet", None)
    return sh.worksheet(worksheet_name) if worksheet_name else sh.sheet1

def check_duplicate_participation(name: str, age: int, gender: str):
    """Block if same (name,age,gender) already submitted today (KST). Returns (is_duplicate, message)."""
    try:
        ws = _get_sheet_worksheet()
        today = _kst_now().strftime("%Y%m%d")
        # Read minimal columns: timestamp, filename, name, age, gender
        rows = ws.get("A2:E")  # list[list[str]]
        name0 = str(name).strip()
        age0 = str(age).strip()
        gender0 = str(gender).strip()
        for r in rows:
            if len(r) < 5:
                continue
            ts, _fn, nm, ag, gd = r[0], r[1], r[2], r[3], r[4]
            if str(ts).strip()[:8] != today:
                continue
            if str(nm).strip() == name0 and str(ag).strip() == age0 and str(gd).strip() == gender0:
                return True, f"동일한 참여자 정보로 오늘({today}) 이미 제출 기록이 있어 **중복 참여가 제한**됩니다."
        return False, "중복 참여 없음"
    except Exception as e:
        # Best-effort: if check can't run, do not block.
        return False, f"중복 참여 확인 생략: {type(e).__name__}: {e}"

# =========================
# Consent gate (required)
# =========================
if "enrolled" not in st.session_state:
    st.session_state.enrolled = False
if "show_instructions" not in st.session_state:
    st.session_state.show_instructions = False

def consent_block():
    st.subheader("연구 참여 동의 및 기본정보 입력")
    st.caption("아래 항목은 필수입니다. 동의하지 않으면 평가를 진행할 수 없습니다.")

    with st.form("consent_form", clear_on_submit=False):
        name = st.text_input("이름(실명 또는 연구ID) *", value="")
        age = st.number_input("나이 *", min_value=1, max_value=120, value=60, step=1)
        gender = st.selectbox("성별 *", ["남", "여"])
        diag_years = st.number_input("진단연차(진단 후 경과년수) *", min_value=0, max_value=60, value=0, step=1)
        dopa_meds = st.selectbox("도파민 약(레보도파 등) 복용 여부 *", ["예", "아니오", "모름"])
        hearing_issue = st.selectbox("청각 문제(난청/보청기/이명 등) 여부 *", ["없음", "있음", "모름"])
        device = st.selectbox("녹음 기기 *", ["노트북", "핸드폰", "태블릿", "외장 마이크/레코더", "기타"])
        mic = st.text_input("마이크 정보(선택)", value="")
        # --- Research team test mode (bypass duplicate guard) ---
        with st.expander("연구팀 테스트(중복 참여 허용)", expanded=False):
            tester_mode = st.checkbox("연구팀/테스트 모드로 진행", value=False)
            tester_code = st.text_input("테스트 코드(관리자용)", type="password", value="")
        # --- Required confirmations (bigger / bold) ---
        st.markdown(
            """<style>
            .consent-check{font-size:18px;font-weight:800;line-height:1.45;margin-top:2px;margin-bottom:8px;}
            </style>""", unsafe_allow_html=True
        )

        c1, c2 = st.columns([0.07, 0.93], vertical_alignment="center")
        with c1:
            dist_ok = st.checkbox(" ", key="dist_ok_chk", label_visibility="collapsed")
        with c2:
            st.markdown("""<div class="consent-check"><b>녹음 기기(마이크)와의 거리가 약 30cm임을 확인했습니다. (필수)</b></div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([0.07, 0.93], vertical_alignment="center")
        with c1:
            read_ok = st.checkbox(" ", key="read_ok_chk", label_visibility="collapsed")
        with c2:
            st.markdown("""<div class="consent-check"><b>사용 방법 안내를 읽고 이해했습니다. (필수)</b></div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([0.07, 0.93], vertical_alignment="center")
        with c1:
            consent = st.checkbox(" ", key="consent_chk", label_visibility="collapsed")
        with c2:
            st.markdown("""<div class="consent-check"><b>본 연구(온라인 음성 평가) 참여에 동의합니다. (필수)</b></div>""", unsafe_allow_html=True)
        submitted = st.form_submit_button("✅ 동의하고 시작하기")

    if submitted:
        problems = []
        if not consent:
            problems.append("연구 참여 동의가 필요합니다.")
        if not str(name).strip():
            problems.append("이름(또는 연구ID)을 입력해주세요.")
        if not dist_ok:
            problems.append("거리(약 30cm) 확인이 필요합니다.")
        if not read_ok:
            problems.append("사용 방법 안내 확인이 필요합니다.")
        if problems:
            st.error(" / ".join(problems))
            st.stop()

        # Validate test mode code if enabled (research team)
        is_tester = False
        if tester_mode:
            admin_code = None
            try:
                if "admin" in st.secrets and "bypass_code" in st.secrets["admin"]:
                    admin_code = str(st.secrets["admin"]["bypass_code"]).strip()
            except Exception:
                admin_code = None

            if not admin_code:
                st.error("연구팀 테스트 모드를 사용하려면 관리자 코드가 설정되어 있어야 합니다. (Streamlit Secrets의 [admin].bypass_code)")
                st.stop()
            if str(tester_code).strip() != admin_code:
                st.error("테스트 코드가 올바르지 않습니다.")
                st.stop()
            is_tester = True

        # Duplicate participation guard (best-effort; blocks when a duplicate is detected)
        if is_tester:
            st.info("🧪 **연구팀 테스트 모드**: 중복 참여 제한을 적용하지 않습니다.")
        else:
            is_dup, dup_msg = check_duplicate_participation(str(name).strip(), int(age), gender)
            if is_dup:
                st.error(f"⚠️ {dup_msg}")
                st.stop()
            else:
                # Show non-blocking status only if we had to skip the check due to config
                if str(dup_msg).startswith("중복 참여 확인 생략"):
                    st.warning(f"ℹ️ {dup_msg}")

        st.session_state.enrolled = True
        st.session_state.show_instructions = True
        st.session_state.patient_info = {
            "name": str(name).strip(),
            "is_test": bool(is_tester),
            "age": int(age),
            "gender": gender,
            "diag_years": int(diag_years),
            "dopa_meds": dopa_meds,
            "hearing_issue": hearing_issue,
            "device": device,
            "mic": str(mic).strip(),
            "distance_ok": bool(dist_ok),
        }
        st.rerun()

if not st.session_state.enrolled:
    st.info("""📌 연구 목적(요약)

안녕하세요. 본 연구는 **대림대학교 언어치료학과**에서 **파킨슨병(PD)** 진단을 받은 분들의 **낭독 음성**을 수집하여, **음향학적 지표(평균 음도, 음도 범위, 평균 강도, 말속도)**와 **자가지각 설문(VHI-10)**이 어떤 양상으로 나타나는지 분석하고, 이를 바탕으로 향후 **평가 도구** 및 **중재(훈련/디지털 치료)** 개발에 활용하기 위해 진행됩니다.

연구에 참여하실 경우,

평가 과정에서 입력하신 **이름/나이/성별**과 **녹음된 음성**, **설문 결과**는 연구 목적에 한해 사용되며, 연구팀이 자료를 검토할 수 있도록 **안전한 방식으로 저장**됩니다. 연구 참여는 자발적이며, 원하실 경우 언제든 중단하실 수 있습니다.

📌 사용 방법(요약)

1) 글자 크기를 조절하면 낭독 문단의 글자 크기가 변경됩니다.  
2) 녹음 기기(마이크)와의 거리는 **약 30cm**를 유지해주세요.  
3) 너무 잘 읽으려고 하지도, 일부러 안 좋게 읽으려고 하지도 말고 **‘편안하게’** 읽어주세요.  
4) **[녹음 시작] → 낭독 → [정지] → [녹음된 음성 분석]** 순서로 진행합니다.  
5) 마지막으로 **VHI-10**을 작성하고 **[결과 저장/전송]**을 눌러주세요.  
6) 본 연구는 동일 참여자의 **중복 참여가 제한**될 수 있어, 이미 참여하신 경우 **재참여가 어려울 수 있습니다.**
""")
    consent_block()
    st.stop()

with st.sidebar:
    st.header("👤 대상자 정보")
    pinfo = st.session_state.get("patient_info", {})
    st.write(f"- 이름: **{pinfo.get('name','')}**")
    st.write(f"- 나이: **{pinfo.get('age','')}**")
    st.write(f"- 성별: **{pinfo.get('gender','')}**")
    st.write(f"- 녹음기기: **{pinfo.get('device','')}**")
    if pinfo.get("mic"):
        st.write(f"- 마이크: **{pinfo.get('mic')}**")

    st.markdown("---")
    if st.button("🆕 새 평가 시작", help="현재 입력/녹음/설문 내용을 초기화합니다."):
        reset_for_new_evaluation()

def _instructions_body():
    st.markdown("### 📌 평가 사용방법")
    st.markdown(
        "- 글자 크기를 수정하면 낭독 문단의 글자 크기가 변경됩니다.\n"
        "- 녹음 기기(마이크)와의 거리는 **약 30cm**를 유지해주세요.\n"
        "- 너무 잘 읽으려고 하지도, 일부러 안 좋게 읽으려고 하지도 말고 **편안하게** 읽어주세요.\n"
        "- **[녹음 시작] → 낭독 → [정지] → [녹음된 음성 분석]** 순서로 진행합니다.\n"
        "- 분석 후 **VHI-10 작성 → [결과 저장/전송]**을 눌러주세요.\n"
        "- 본 연구는 동일 참여자의 **중복 참여가 제한**될 수 있어, 이미 참여하신 경우 **재참여가 어려울 수 있습니다.**"
    )
    if st.button("닫기"):
        st.session_state.show_instructions = False
        st.session_state.scroll_to_top = True
        st.rerun()

if st.session_state.get("show_instructions", False):
    if hasattr(st, "dialog"):
        @st.dialog("평가 사용방법 안내")
        def _dlg():
            _instructions_body()
        _dlg()
    else:
        st.warning("📌 평가 사용방법 안내 (팝업 대체)")
        _instructions_body()

# =========================
# Section 1: Recording
# =========================
st.header("1. 음성 데이터 수집(마이크 녹음)")
font_size = st.slider("🔍 글자 크기", 15, 50, 27, key="fs_read_eval")

st.markdown(f"**낭독 문단:** {READING_TITLE}  |  **전체 음절 수:** {TOTAL_SYLLABLES_FIXED}음절")
st.markdown(styled_text(READING_TEXT, font_size), unsafe_allow_html=True)
st.markdown(
    "<div style='font-size: 13px; color: #555; margin-top: 6px;'>"
    "조음 위치 전환(양순–치조–연구) · 경음/평음/기식음 대조 · 마찰/파찰(ㅅ·ㅆ·ㅈ·ㅊ) 정밀도 · 유음/비음(ㄹ·ㄴ·ㅁ) 안정성"
    "</div>",
    unsafe_allow_html=True
)

# --- One-button audio recorder (start/stop) ---
def _audio_recorder_one_button(key: str = "recorder") -> dict | None:
    """Return dict {'wav_base64': str, 'sample_rate': int} when recording stops."""
    html = r'''
<div style="display:flex; flex-direction:column; gap:10px; font-family: sans-serif;">
  <button id="recBtn"
    style="font-size:20px; font-weight:700; padding:12px 16px; border-radius:12px; border:1px solid rgba(0,0,0,0.2); cursor:pointer;">
    🔴 녹음 시작
  </button>
  <div id="status" style="font-size:14px; opacity:0.85;">버튼을 눌러 녹음을 시작하세요.</div>
</div>

<script>
(function() {
  const sendValue = (value) => {
    const msg = {isStreamlitMessage: true, type: "streamlit:setComponentValue", value: value};
    window.parent.postMessage(msg, "*");
  };
  const setHeight = (height) => {
    const msg = {isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: height};
    window.parent.postMessage(msg, "*");
  };
  setHeight(170);

  const recBtn = document.getElementById("recBtn");
  const status = document.getElementById("status");

  let recording = false;
  let audioContext = null;
  let processor = null;
  let source = null;
  let stream = null;
  let chunks = [];
  let sampleRate = 44100;
  let t0 = null;
  let timer = null;

  function mergeChunks(chunks) {
    let length = 0;
    for (const c of chunks) length += c.length;
    const result = new Float32Array(length);
    let offset = 0;
    for (const c of chunks) { result.set(c, offset); offset += c.length; }
    return result;
  }

  function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);

    return new Uint8Array(buffer);
  }

  async function startRecording() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      status.textContent = "마이크 권한이 필요합니다. 브라우저에서 마이크 접근을 허용해주세요.";
      return;
    }
    chunks = [];
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sampleRate = audioContext.sampleRate;

    source = audioContext.createMediaStreamSource(stream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);

    const gain = audioContext.createGain();
    gain.gain.value = 0.0;

    processor.onaudioprocess = (e) => {
      if (!recording) return;
      const input = e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(input));
    };

    source.connect(processor);
    processor.connect(gain);
    gain.connect(audioContext.destination);

    recording = true;
    recBtn.textContent = "⏹️ 녹음 정지";
    t0 = Date.now();
    timer = setInterval(() => {
      const sec = Math.floor((Date.now() - t0) / 1000);
      status.textContent = `녹음 중... ${sec}초 (다시 누르면 정지합니다)`;
    }, 250);
  }

  async function stopRecording() {
    recording = false;
    recBtn.textContent = "🔴 녹음 시작";
    clearInterval(timer);
    status.textContent = "녹음을 처리 중입니다...";

    try {
      if (processor) processor.disconnect();
      if (source) source.disconnect();
      if (stream) stream.getTracks().forEach(t => t.stop());
      if (audioContext) await audioContext.close();
    } catch (e) {}

    const samples = mergeChunks(chunks);
    const wavBytes = encodeWAV(samples, sampleRate);

    // base64 encode
    let binary = '';
    const chunkSize = 0x8000;
    for (let i = 0; i < wavBytes.length; i += chunkSize) {
      binary += String.fromCharCode.apply(null, wavBytes.subarray(i, i + chunkSize));
    }
    const b64 = btoa(binary);

    status.textContent = "✅ 녹음이 완료되었습니다. 아래에서 재생 후 [녹음된 음성 분석]을 진행하세요.";
    sendValue({wav_base64: b64, sample_rate: sampleRate});
  }

  recBtn.addEventListener("click", async () => {
    if (!recording) {
      await startRecording();
    } else {
      await stopRecording();
    }
  });
})();
</script>
'''
    return components.html(html, height=180, key=key)

rec = _audio_recorder_one_button(key="one_button_recorder")

TEMP_WAV = "temp_eval.wav"
if rec and isinstance(rec, dict) and rec.get("wav_base64"):
    try:
        data = base64.b64decode(rec["wav_base64"])
        with open(TEMP_WAV, "wb") as f:
            f.write(data)
        st.session_state["wav_path"] = str(Path(TEMP_WAV).resolve())
        st.session_state["wav_bytes"] = data
    except Exception as e:
        st.error(f"녹음 데이터 처리 중 오류가 발생했습니다: {e}")

if st.session_state.get("wav_bytes"):
    st.audio(st.session_state["wav_bytes"], format="audio/wav")
st.markdown("---")

# =========================
# Analysis helpers
# =========================
def compute_pitch_stats(sound: parselmouth.Sound, gender: str):
    if gender == "여":
        f0_min, f0_max = 100, 500
    else:
        f0_min, f0_max = 70, 500

    pitch = sound.to_pitch(time_step=0.01, pitch_floor=f0_min, pitch_ceiling=f0_max)
    freq = pitch.selected_array["frequency"]
    t = pitch.xs()

    mask = np.isfinite(freq) & (freq > 0)
    f = freq[mask]
    tt = t[mask]

    if f.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    lo, hi = np.percentile(f, [2, 98])
    f2 = f[(f >= lo) & (f <= hi)]
    if f2.size == 0:
        f2 = f

    f0_mean = float(np.mean(f2))
    f0_range = float(np.max(f2) - np.min(f2))

    start_t = float(tt[0])
    end_t = float(tt[-1])
    return f0_mean, f0_range, start_t, end_t

def analyze_wav(path: str, gender: str):
    sound = parselmouth.Sound(path)
    f0_mean, f0_range, start_t, end_t = compute_pitch_stats(sound, gender)

    intensity = sound.to_intensity()
    if np.isfinite(start_t) and np.isfinite(end_t) and (end_t > start_t):
        mean_db = float(call(intensity, "Get mean", start_t, end_t, "energy"))
        dur = max(0.1, end_t - start_t)
    else:
        mean_db = float(call(intensity, "Get mean", 0, 0, "energy"))
        dur = max(0.1, float(sound.duration))

    sps = float(TOTAL_SYLLABLES_FIXED) / dur
    return {"f0": f0_mean, "range": f0_range, "intensity_db": mean_db, "sps": sps}

# =========================
# Section 2: Analysis results (table only)
# =========================
st.header("2. 음향학적 분석 결과")
if st.button("📈 녹음된 음성 분석"):
    wav_path = st.session_state.get("wav_path")
    if not wav_path or not os.path.exists(wav_path):
        st.error("녹음 파일이 없습니다. 먼저 녹음을 진행해주세요.")
    else:
        g = st.session_state.patient_info.get("gender", "남")
        a = analyze_wav(wav_path, g)
        st.session_state["analysis"] = a

analysis = st.session_state.get("analysis")
if analysis:
    df = pd.DataFrame({
        "항목": ["평균 음도(Hz)", "음도 범위(Hz)", "평균 강도(dB)", "말속도(SPS)"],
        "수치": [
            f"{analysis['f0']:.2f}" if np.isfinite(analysis['f0']) else "",
            f"{analysis['range']:.2f}" if np.isfinite(analysis['range']) else "",
            f"{analysis['intensity_db']:.2f}" if np.isfinite(analysis['intensity_db']) else "",
            f"{analysis['sps']:.2f}" if np.isfinite(analysis['sps']) else "",
        ]
    })
    st.dataframe(df, hide_index=True)

st.markdown("---")

# =========================
# Section 3: VHI-10 only
# =========================
st.header("3. VHI-10 입력")
st.caption("파킨슨을 진단 받은 후, 본인의 목소리에 대해 느끼는 대로 설문지를 작성해주세요.")

# 문항 글자 크기(사용자 조절)
vhi_q_fs = st.slider("🔠 VHI 문항 글자 크기", 14, 30, 18, key="vhi_q_fs")

vhi_opts = [0, 1, 2, 3, 4]
VHI_LABELS = {
    0: "전혀 그렇지 않다",
    1: "거의 그렇지 않다",
    2: "가끔 그렇다",
    3: "자주 그렇다",
    4: "항상 그렇다",
}


# --- VHI item display (bigger question text) ---
st.markdown(
    f"""
    <style>
      .vhi-q{{
        font-size: {int(vhi_q_fs)}px;
        font-weight: 600;
        line-height: 1.35;
        margin: 14px 0 6px 0;
      }}
      .vhi-help{{
        font-size: 13px;
        color: #666;
        margin: 0 0 8px 0;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

def vhi_item(num: int, text: str, key: str) -> int:
    st.markdown(f"<div class='vhi-q'>{num}. {html.escape(text)}</div>", unsafe_allow_html=True)
    return int(
        st.radio(
            label=f"vhi_{num}",
            options=vhi_opts,
            index=0,
            format_func=lambda x: f"{x} - {VHI_LABELS[x]}",
            key=key,
            label_visibility="collapsed",
        )
    )

with st.expander("VHI-10 문항 입력 (클릭해서 펼치기)", expanded=True):
    q1 = vhi_item(1, "목소리 때문에 상대방이 내 말을 알아듣기 힘들어한다.", "vhi_q1")
    q2 = vhi_item(2, "시끄러운 곳에서는 사람들이 내 말을 이해하기 어려워한다.", "vhi_q2")
    q3 = vhi_item(3, "사람들이 나에게 목소리가 왜 그러냐고 묻는다.", "vhi_q3")
    q4 = vhi_item(4, "목소리를 내려면 힘을 주어야 나오는 것 같다.", "vhi_q4")
    q5 = vhi_item(5, "음성문제로 개인 생활과 사회생활에 제한을 받는다.", "vhi_q5")
    q6 = vhi_item(6, "목소리가 언제쯤 맑게 잘 나올지 알 수가 없다(예측이 어렵다).", "vhi_q6")
    q7 = vhi_item(7, "내 목소리 때문에 대화에 끼지 못하여 소외감을 느낀다.", "vhi_q7")
    q8 = vhi_item(8, "음성 문제로 인해 소득(수입)에 감소가 생긴다.", "vhi_q8")
    q9 = vhi_item(9, "내 목소리 문제로 속이 상한다.", "vhi_q9")
    q10 = vhi_item(10, "음성 문제가 장애로(핸디캡으로) 여겨진다.", "vhi_q10")
vhi_f = int(q1 + q2 + q5 + q7 + q8)
vhi_p = int(q3 + q4 + q6)
vhi_e = int(q9 + q10)
vhi_total = int(vhi_f + vhi_p + vhi_e)

st.session_state["vhi_total"] = vhi_total
st.session_state["vhi_f"] = vhi_f
st.session_state["vhi_p"] = vhi_p
st.session_state["vhi_e"] = vhi_e

c1, c2, c3, c4 = st.columns(4)
c1.metric("총점", f"{vhi_total}점")
c2.metric("기능(F)", f"{vhi_f}점")
c3.metric("신체(P)", f"{vhi_p}점")
c4.metric("정서(E)", f"{vhi_e}점")

st.markdown("---")

# =========================
# Section 4: Save/Send
# =========================
st.header("4. 결과 저장/전송(연구팀 수집)")
st.caption("※ 이 단계에서는 환자에게 하위집단 진단 결과를 표시하지 않고, 연구팀에게 음성파일과 측정치가 전송됩니다.")

# Duplicate-send guard (same recording within the same session)
wav_path_now = st.session_state.get("wav_path")
analysis_now = st.session_state.get("analysis")
sub_key = make_submission_key(wav_path_now, st.session_state.get("patient_info", {})) if wav_path_now else ""
already_sent = bool(sub_key) and (sub_key in st.session_state["sent_submission_keys"])
if already_sent:
    st.info("✅ 이 녹음 건은 이미 전송이 완료되었습니다. (중복 전송 방지)\n\n새로 녹음한 뒤 [📈 녹음된 음성 분석]을 다시 누르면 전송 버튼이 다시 활성화됩니다.")

if st.button("📤 결과 저장/전송", type="primary", disabled=already_sent):
    wav_path = st.session_state.get("wav_path")
    analysis = st.session_state.get("analysis")

    if not wav_path or not os.path.exists(wav_path):
        st.error("녹음 파일이 없습니다. 먼저 녹음을 진행해주세요.")
    elif not analysis:
        st.error("분석 결과가 없습니다. [📈 녹음된 음성 분석]을 먼저 눌러주세요.")
    else:
        analysis = dict(analysis)
        analysis["vhi_total"] = st.session_state.get("vhi_total", "")
        analysis["vhi_f"] = st.session_state.get("vhi_f", "")
        analysis["vhi_p"] = st.session_state.get("vhi_p", "")
        analysis["vhi_e"] = st.session_state.get("vhi_e", "")

        # Internal label for research logging (not shown to participant)
        final_diag, _probs = predict_step2(
            STEP2_MODEL,
            float(analysis.get("intensity_db", np.nan)),
            float(analysis.get("sps", np.nan)),
        )

        log_filename, sheet_ok, sheet_msg, email_ok, email_msg = send_email_and_log_sheet(
            wav_path,
            st.session_state.get("patient_info", {}),
            analysis,
            final_diag or ""
        )

        # Mark as sent only when BOTH email + sheet succeeded (prevents accidental duplicates)
        if sheet_ok and email_ok and sub_key:
            st.session_state["sent_submission_keys"].add(sub_key)

        if sheet_ok and email_ok:
            st.success("✅ 저장/전송을 완료했습니다.\n\n**향후 연구에 도움이 될 수 있도록 참여해주셔서 감사합니다.**")
        elif email_ok and (not sheet_ok):
            st.warning("⚠️ 이메일 전송은 성공했지만, 구글시트 저장은 실패했습니다.")
        elif sheet_ok and (not email_ok):
            st.warning("⚠️ 구글시트 저장은 성공했지만, 이메일 전송은 실패했습니다.")
        else:
            st.error("❌ 저장/전송에 실패했습니다. 아래 로그를 확인하세요.")

        st.write(f"- 저장 파일명: `{log_filename}`")
        st.write(f"- 구글시트: {'성공' if sheet_ok else '실패/생략'} · {sheet_msg}")
        st.write(f"- 이메일: {'성공' if email_ok else '실패/생략'} · {email_msg}")
