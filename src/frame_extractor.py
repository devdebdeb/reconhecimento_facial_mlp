import cv2
import os
import time

video_folder = "C:/Users/ANDREMESSINA/reconhecimento-facial-mlp/videos"

output_folder = "C:/Users/ANDREMESSINA/reconhecimento-facial-mlp/dataset"

frame_interval = 9

image_format = '.png'

def extract_frames_from_videos(video_dir, output_dir, interval, img_format):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Pasta de saída principal criada/verificada em: '{output_dir}'")

    try:
        video_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
        if not video_files:
            print(f"ERRO: Nenhum arquivo encontrado na pasta '{video_dir}'. Verifique o caminho.")
            return
    except FileNotFoundError:
        print(f"ERRO: A pasta de vídeos '{video_dir}' não foi encontrada. Verifique o caminho.")
        return
    except Exception as e:
        print(f"ERRO ao listar arquivos em '{video_dir}': {e}")
        return

    print(f"Vídeos encontrados: {video_files}")
    total_start_time = time.time()

    for video_filename in video_files:
        video_path = os.path.join(video_dir, video_filename)

        person_name = os.path.splitext(video_filename)[0]
        if not person_name:
            print(f"AVISO: Não foi possível extrair nome do arquivo '{video_filename}'. Pulando...")
            continue

        print(f"\n--- Processando vídeo: '{video_filename}' (Pessoa: '{person_name}') ---")

        person_output_folder = os.path.join(output_dir, person_name)
        os.makedirs(person_output_folder, exist_ok=True)
        print(f"Salvando frames em: '{person_output_folder}'")

        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            print(f"ERRO: Não foi possível abrir o vídeo '{video_path}'. Pulando...")
            continue

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        print(f"Info do vídeo: FPS={fps:.2f}, Total de Frames={total_frames}, Duração={duration:.2f}s")

        frame_count = 0
        saved_frame_count = 0
        video_start_time = time.time()

        while True:
            success, frame = video_capture.read()

            if not success:
                break

            if frame_count % interval == 0:

                frame_filename = f"{person_name}_frame_{str(saved_frame_count).zfill(5)}{img_format}"
                output_path = os.path.join(person_output_folder, frame_filename)

                cv2.imwrite(output_path, frame)
                saved_frame_count += 1

                if saved_frame_count % 100 == 0:
                    print(f"  ... {saved_frame_count} frames salvos...")

            frame_count += 1

        video_capture.release()
        video_end_time = time.time()
        print(f"Processamento de '{video_filename}' concluído.")
        print(f"Total de frames lidos: {frame_count}")
        print(f"Total de frames salvos: {saved_frame_count}")
        print(f"Tempo gasto neste vídeo: {video_end_time - video_start_time:.2f} segundos")

    total_end_time = time.time()
    print(f"\n--- Extração de frames concluída para todos os vídeos ---")
    print(f"Tempo total gasto: {total_end_time - total_start_time:.2f} segundos")

if __name__ == "__main__":
    extract_frames_from_videos(video_folder, output_folder, frame_interval, image_format)
