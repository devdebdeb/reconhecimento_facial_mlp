import cv2
import os
import time
from tqdm import tqdm

class FrameExtractor:
    def __init__(self, video_folder, output_folder, frame_interval=7, image_format='.png'):
        self.video_folder = video_folder
        self.output_folder = output_folder
        self.frame_interval = frame_interval
        self.image_format = image_format
        os.makedirs(self.output_folder, exist_ok=True)

    def validate_paths(self):
        if not os.path.exists(self.video_folder):
            raise FileNotFoundError(f"Pasta de vídeos não encontrada: {self.video_folder}")
        video_files = [f for f in os.listdir(self.video_folder) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            raise ValueError(f"Nenhum vídeo encontrado em {self.video_folder}")
        return video_files

    def extract_frames(self, video_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro ao abrir vídeo: {video_path}")
            return 0, 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved_frames = 0

        for frame_idx in range(0, total_frames, self.frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_path = os.path.join(output_dir, f"frame_{str(saved_frames).zfill(5)}{self.image_format}")
            cv2.imwrite(frame_path, frame)
            saved_frames += 1
        cap.release()
        return total_frames, saved_frames

    def process_all_videos(self):
        try:
            video_files = self.validate_paths()
            print(f"Processando {len(video_files)} vídeos...")
            total_start = time.time()

            for video_file in tqdm(video_files, desc="Processando vídeos"):
                video_path = os.path.join(self.video_folder, video_file)
                person_name = os.path.splitext(video_file)[0]
                output_dir = os.path.join(self.output_folder, person_name)
                os.makedirs(output_dir, exist_ok=True)
                start_time = time.time()
                total_frames, saved_frames = self.extract_frames(video_path, output_dir)
                elapsed = time.time() - start_time
                print(f"\n{person_name}: {saved_frames} frames salvos (de {total_frames}) em {elapsed:.2f}s")

            total_elapsed = time.time() - total_start
            print(f"\nProcessamento concluído em {total_elapsed:.2f} segundos")

        except Exception as e:
            print(f"Erro durante o processamento: {str(e)}")

if __name__ == "__main__":
    config = {
        'video_folder': "./videos",
        'output_folder': "./dataset_raw",
        'frame_interval': 5,
        'image_format': '.png'
    }
    extractor = FrameExtractor(**config)
    extractor.process_all_videos()