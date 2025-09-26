import os
import subprocess
import sys
import cv2
import csv
import pandas as pd
from ultralytics import YOLO

# Caminho do vídeo
video_path = sys.argv[1]
nome_video = os.path.splitext(os.path.basename(video_path))[0]

# Modelo YOLOv8 COCO pré-treinado
model = YOLO("yolov8s.pt")

# Saída da detecção frame a frame
csv_pred = f"Output\\detec_yolov8_{nome_video}.csv"

# Coordenadas das vagas
vagas_path = f"Video\\vagas_{nome_video}.csv"
df_vagas = pd.read_csv(vagas_path)
vagas = [tuple(row) for row in df_vagas.values]




# Função de interseção entre caixas
def intersects(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    return not (xA + wA < xB or xA > xB + wB or yA + hA < yB or yA > yB + hB)

# Abre o vídeo
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps
print(f"FPS: {fps}, Total de frames: {total_frames}, Duração: {duration_sec:.2f} segundos")

# Prepara CSV
with open(csv_pred, mode="w", newline="") as file:
    writer = csv.writer(file)
    header = ["frame"] + [f"vaga_{i}" for i in range(len(vagas))]
    writer.writerow(header)

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Faz a inferência com YOLOv8
        results = model(frame, stream=True)

        carros = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if model.names[cls] == "car" and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    carros.append((x1, y1, w, h))

        # Verifica ocupação das vagas
        ocupacao_frame = []
        for vaga in vagas:
            ocupada = any(intersects(vaga, carro) for carro in carros)
            ocupacao_frame.append(1 if ocupada else 0)

        # Escreve no CSV
        writer.writerow([frame_number] + ocupacao_frame)
        frame_number += 1

cap.release()
print(f" Predições salvas em: {csv_pred}")

# Chama o script de comparação de métricas
# Argumentos: pred, ground truth, saída
csv_true = f"Video\\ground_truth_{nome_video}.csv"
csv_out  = f"Output\\yolov8_metrics_{nome_video}.csv"

subprocess.run(["python", "comparar_metricas.py", csv_pred, csv_true, csv_out])
