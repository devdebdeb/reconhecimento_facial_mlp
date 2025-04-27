import os
import cv2
import mediapipe as mp

input_dir = "../dataset"
output_dir = "../processed_dataset"
image_size = 48

mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

def process_image(image_path, save_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.process(image_rgb)

    if result.detections:
        for detection in result.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape

            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            box_width = int(bbox.width * w)
            box_height = int(bbox.height * h)

            margin_x = int(box_width * 0.2)
            margin_y = int(box_height * 0.2)

            x1 = max(0, x_min - margin_x)
            y1 = max(0, y_min - margin_y)
            x2 = min(w, x_min + box_width + margin_x)
            y2 = min(h, y_min + box_height + margin_y)

            face = image[y1:y2, x1:x2]
            face_resized = cv2.resize(face, (image_size, image_size))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(save_path, face_gray)
            return

    print(f"Nenhum rosto detectado em: {image_path}")

def preprocess_all():
    os.makedirs(output_dir, exist_ok=True)

    for person in os.listdir(input_dir):
        person_input_path = os.path.join(input_dir, person)
        person_output_path = os.path.join(output_dir, person)
        os.makedirs(person_output_path, exist_ok=True)

        for img_file in os.listdir(person_input_path):
            input_path = os.path.join(person_input_path, img_file)
            output_path = os.path.join(person_output_path, img_file)
            process_image(input_path, output_path)

    print("Pré-processamento concluído.")

if __name__ == "__main__":
    preprocess_all()
