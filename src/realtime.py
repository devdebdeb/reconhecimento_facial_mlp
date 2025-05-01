import cv2
import torch
from predict import FacePredictor
import mediapipe as mp

class RealTimeRecognizer:
    def __init__(self, predictor_config):
        self.predictor = FacePredictor(**predictor_config)
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.7
        )
        
    def process_frame(self, frame):
        # Converter para RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar rostos
        results = self.face_detection.process(rgb)
        if not results.detections:
            return frame
            
        # Processar cada rosto detectado
        for detection in results.detections:
            # Extrair coordenadas
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            
            # Recortar rosto
            face = frame[y:y+height, x:x+width]
            if face.size == 0:
                continue
                
            # Fazer predição
            label, confidence = self.predictor.predict(face)
            
            # Desenhar retângulo e label
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (36,255,12), 2)
                       
        return frame
        
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro ao abrir câmera")
            return
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = self.process_frame(frame)
                cv2.imshow('Reconhecimento Facial', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    config = {
        'model_path': './trained_models/best_mlp_model.pth',
        'idx_to_class': {
            0: "Andre",
            1: "Cesar", 
            2: "Enzo",
            3: "Will"
        },
        'confidence_threshold': 0.7
    }
    
    recognizer = RealTimeRecognizer(config)
    recognizer.run()