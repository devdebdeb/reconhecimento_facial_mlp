import cv2
import numpy as np
import mediapipe as mp
from model import MLP  # Você precisaria criar um modelo específico para expressões

class ExpressionRecognizer:
    def __init__(self, model_path):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)
            
        self.model = MLP(input_size=468*3, hidden_sizes=[256, 128], num_classes=4)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.expressions = {
            0: "Feliz",
            1: "Neutro",
            2: "Bravo",
            3: "Surpreso"
        }
        
    def get_landmarks(self, image):
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
            
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
            
        return torch.FloatTensor(landmarks)
        
    def predict_expression(self, image):
        landmarks = self.get_landmarks(image)
        if landmarks is None:
            return "Nenhum rosto detectado"
            
        with torch.no_grad():
            outputs = self.model(landmarks.unsqueeze(0))
            _, predicted = torch.max(outputs, 1)
            return self.expressions[predicted.item()]