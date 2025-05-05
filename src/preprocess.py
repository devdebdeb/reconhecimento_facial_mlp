import os
import cv2
import mediapipe as mp
from tqdm import tqdm

class FacePreprocessor:
    def __init__(self, input_dir, output_dir, img_size=48, min_confidence=0.6):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_size = (img_size, img_size)
        self.min_confidence = min_confidence
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_confidence
        )

    def validate_dirs(self):
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Diretório de entrada não encontrado: {self.input_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_and_crop(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if not results.detections:
            return None
        best_detection = max(results.detections, key=lambda d: d.score[0])
        bbox = best_detection.location_data.relative_bounding_box
        h, w = image.shape[:2]
        margin = 0.2
        x = int(bbox.xmin * w - margin * bbox.width * w)
        y = int(bbox.ymin * h - margin * bbox.height * h)
        width = int(bbox.width * w * (1 + 2 * margin))
        height = int(bbox.height * h * (1 + 2 * margin))
        x, y = max(0, x), max(0, y)
        face = image[y:y+height, x:x+width]
        if face.size == 0:
            return None
        face_resized = cv2.resize(face, self.img_size)
        return cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

    def process_image(self, input_path, output_path):
        try:
            image = cv2.imread(input_path)
            if image is None:
                return False
            face = self.detect_and_crop(image)
            if face is None:
                return False
            return cv2.imwrite(output_path, face)
        except Exception as e:
            print(f"Erro ao processar {input_path}: {str(e)}")
            return False

    def process_all(self):
        self.validate_dirs()
        processed = 0
        skipped = 0
        for person in tqdm(os.listdir(self.input_dir), desc="Processando pessoas"):
            person_input = os.path.join(self.input_dir, person)
            person_output = os.path.join(self.output_dir, person)
            if not os.path.isdir(person_input):
                continue
            os.makedirs(person_output, exist_ok=True)
            for img_name in os.listdir(person_input):
                input_path = os.path.join(person_input, img_name)
                output_path = os.path.join(person_output, img_name)
                if self.process_image(input_path, output_path):
                    processed += 1
                else:
                    skipped += 1
        print(f"\nPré-processamento concluído. Processadas: {processed}, Ignoradas: {skipped}")

if __name__ == "__main__":
    preprocessor = FacePreprocessor(
        input_dir="./dataset_raw",
        output_dir="./processed_dataset",
        img_size=48
    )
    preprocessor.process_all()