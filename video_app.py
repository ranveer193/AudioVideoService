import os, tempfile, json, cv2, math
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
import mediapipe as mp

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                          refine_landmarks=True,
                                          max_num_faces=1)

app = FastAPI(title="Interview Video-Metrics API")

@app.get("/")
def root():
    return {"status": "ok"}

def angle_between(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return math.degrees(math.acos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)))

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    if file.content_type not in {"video/mp4", "video/webm"}:
        raise HTTPException(415, "Upload MP4/WebM")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read()); vpath = tmp.name

    cap = cv2.VideoCapture(vpath)
    frame_cnt, eye_contact, yaw_sum, nods = 0, 0, 0.0, 0
    prev_pitch = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_cnt += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        if not res.multi_face_landmarks: continue
        lm = res.multi_face_landmarks[0].landmark

        # eye contact: iris centre ~ image centre
        iris = lm[468]  # right iris centre
        h, w = frame.shape[:2]
        if abs(iris.x - 0.5) < 0.08 and abs(iris.y - 0.5) < 0.08:
            eye_contact += 1

        # head yaw (left/right)
        left  = np.array([lm[33].x, lm[33].y, lm[33].z])
        right = np.array([lm[263].x, lm[263].y, lm[263].z])
        mid   = (left + right) / 2
        nose  = np.array([lm[1].x, lm[1].y, lm[1].z])
        yaw   = angle_between(right-left, [1,0,0])  # rough yaw
        yaw_sum += abs(yaw)

        # nod detection via pitch
        chin = np.array([lm[152].x, lm[152].y, lm[152].z])
        pitch = angle_between(chin-mid, [0,1,0])
        if prev_pitch and (prev_pitch - pitch) > 10:  # downward nod
            nods += 1
        prev_pitch = pitch

    cap.release(); os.remove(vpath)
    if frame_cnt == 0:
        raise HTTPException(400, "Empty/invalid video")

    metrics = {
        "frames"            : frame_cnt,
        "eye_contact_percent": round(100*eye_contact/frame_cnt, 1),
        "avg_head_yaw_deg"  : round(yaw_sum/frame_cnt, 1),
        "nod_count"         : nods
    }
    return metrics
