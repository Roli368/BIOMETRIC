from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os, time, base64
import cv2

from .services import session as session_svc
from .services import alignment, quality, liveness, verification
from .config import (
    UPLOAD_FOLDER,
    SESSION_TIMEOUT,
    MIN_BUFFER,
    TRUST_THRESHOLD,
    MATCH_THRESHOLD,
    SWAP_THRESHOLD
)
from .utils.io import imdecode_bytes


app = FastAPI(title="Face Verification API")

# ===============================
# ENV CONFIG
# ===============================
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

UPLOAD_DIR = UPLOAD_FOLDER or "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===============================
# STATIC FILES
# ===============================
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ===============================
# CORS (NO LOCALHOST HARD-CODE)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# BASIC ROUTES
# ===============================
@app.get("/", include_in_schema=False)
def root():
    return {
        "status": "ok",
        "message": "Face Verification API",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.get("/api/health")
def api_health():
    return health()

# ===============================
# SESSION
# ===============================
@app.post("/start-session")
def start_session():
    sess = session_svc.create_session()
    return {"status": "OK", "session_id": sess["session_id"]}

@app.post("/api/start-session")
def api_start_session():
    return start_session()

# ===============================
# DOCUMENT UPLOAD
# ===============================
@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    try:
        sess = session_svc.get_session(session_id)
        if not sess:
            return JSONResponse(
                {"status": "FAILED", "message": "Invalid session"},
                status_code=400
            )

        data = await file.read()

        doc_path = os.path.join(UPLOAD_DIR, f"{session_id}_doc.jpg")
        with open(doc_path, "wb") as f:
            f.write(data)

        out_path = os.path.join(UPLOAD_DIR, f"{session_id}_id_face.jpg")
        res = alignment.extract_face_from_id(doc_path, out_path)
        if res is None:
            return {"status": "FAILED", "message": "No face detected"}

        id_face = cv2.imread(out_path)
        emb = verification.infer_embedding(id_face)

        session_svc.set_session(session_id, {
            "id_embedding": emb,
            "doc_face_path": out_path,
            "liveness_passed": False,
            "locked_live_embedding": None,
            "locked_live_image": None,
            "attempts": 0
        })

        return {
            "status": "DOCUMENT_READY",
            "message": "ID face extracted",
            "preview": f"/uploads/{session_id}_id_face.jpg"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"status": "ERROR", "message": str(e)},
            status_code=500
        )

@app.post("/api/upload-document")
async def api_upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    return await upload_document(file, session_id)

# ===============================
# PROCESS FRAME
# ===============================
@app.post("/process-frame")
async def process_frame(
    frame: UploadFile = File(...),
    session_id: str = Form(...)
):
    sess = session_svc.get_session(session_id)
    if not sess:
        return JSONResponse(
            {"status": "FAILED", "message": "Invalid session"},
            status_code=400
        )

    try:
        if time.time() - sess["start_ts"] > SESSION_TIMEOUT:
            return {"status": "EXPIRED", "message": "Session expired"}

        data = await frame.read()
        img = imdecode_bytes(data)
        if img is None:
            return JSONResponse(
                {"status": "FAILED", "message": "Invalid frame"},
                status_code=400
            )

        ok, qres = quality.pre_check_quality(img)
        if not ok:
            session_svc.set_session(session_id, {"paused": True})
            return {
                "status": "PAUSED",
                "message": qres.get("message", "Quality issue"),
                "metrics": qres.get("metrics", {})
            }

        if not sess.get("liveness_instance"):
            lv = liveness.UltimateLiveness10()
            session_svc.set_session(session_id, {"liveness_instance": lv})
            sess = session_svc.get_session(session_id)

        lv = sess["liveness_instance"]

        if not sess.get("liveness_passed"):
            result = lv.verify(img)
            trust = result.get("trust", 0.0)

            if result.get("status") == "LIVE HUMAN" and trust >= TRUST_THRESHOLD:
                emb = verification.infer_embedding(img)
                _, buffer = cv2.imencode(".jpg", img)
                b64_img = base64.b64encode(buffer).decode()

                session_svc.set_session(session_id, {
                    "liveness_passed": True,
                    "locked_live_embedding": emb,
                    "locked_live_image": b64_img
                })

            return {
                "status": "PROCESSING",
                "message": "Checking liveness...",
                "trust": trust
            }

        sim = verification.cosine_sim(
            sess["locked_live_embedding"],
            sess["id_embedding"]
        )

        if sim > MATCH_THRESHOLD:
            return {
                "status": "COMPLETE",
                "message": "Verified",
                "metrics": {"similarity": sim},
                "live_image": f"data:image/jpeg;base64,{sess['locked_live_image']}"
            }

        sess["attempts"] += 1
        session_svc.set_session(session_id, {"attempts": sess["attempts"]})

        return {
            "status": "FAILED" if sess["attempts"] >= 2 else "PROCESSING",
            "message": "Face mismatch",
            "attempts": sess["attempts"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"status": "ERROR", "message": str(e)},
            status_code=500
        )

@app.post("/api/process-frame")
async def api_process_frame(
    frame: UploadFile = File(...),
    session_id: str = Form(...)
):
    return await process_frame(frame, session_id)

# ===============================
# CLEANUP
# ===============================
@app.post("/cleanup-session")
def cleanup_session(session_id: str = Form(...)):
    try:
        for suffix in ["_doc.jpg", "_id_face.jpg"]:
            p = os.path.join(UPLOAD_DIR, f"{session_id}{suffix}")
            if os.path.exists(p):
                os.remove(p)

        session_svc.delete_session(session_id)
        return {"status": "OK", "message": "Session cleaned"}

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

@app.post("/api/cleanup-session")
def api_cleanup_session(session_id: str = Form(...)):
    return cleanup_session(session_id)
