from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from app.box_detector import Detector
import numpy as np
import cv2
from backend import db_utils

app = FastAPI()
detector = Detector()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("Khởi động hệ thống nhận diện khuôn mặt...")
    db_utils.init_face_recognition()

@app.post("/process_frame")
def process_frame(file: UploadFile = File(...)):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)
    
    face_boxes_for_response = []
    
    for (coords, conf, face_name) in face_boxes:
        
        similar_list = [face_name] if face_name != "Không xác định" else []
        
        face_boxes_for_response.append({
            "coords": coords,
            "confidence": conf,
            "similar_faces": similar_list,
        })
            
    return {
        "persons": person_count,
        "faces": face_count,
        "person_boxes": [
            {"coords": coords, "confidence": conf}
            for (coords, conf) in person_boxes
        ],
        "face_boxes": face_boxes_for_response,
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))