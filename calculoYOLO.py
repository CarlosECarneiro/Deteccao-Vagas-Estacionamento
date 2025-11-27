import os
import subprocess
import sys
import cv2
import csv
import pandas as pd
from ultralytics import YOLO

# Recebe o caminho do vídeo enviado pelo main
video_path = sys.argv[1]
nome_video = os.path.splitext(os.path.basename(video_path))[0]

# Carrega o modelo YOLOv8 pré-treinado
model = YOLO("yolov8s.pt")

# Define caminhos dos arquivos CSV
csv_pred = f"Output\\detec_yolov8_{nome_video}.csv"
vagas_path = f"Vagas\\vagas_{nome_video}.csv"
csv_true = f"Video\\ground_truth_{nome_video}.csv"
csv_out  = f"Output\\yolov8_metrics_{nome_video}.csv"

# Lê o CSV com coordenadas e IDs das vagas
df_vagas = pd.read_csv(vagas_path)
vagas = [(int(row["SlotId"]), int(row["X"]), int(row["Y"]),
            int(row["W"]), int(row["H"]))
            for _, row in df_vagas.iterrows()]

# Função de interseção entre caixas
def intersects(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    return not (xA + wA < xB or xA > xB + wB or yA + hA < yB or yA > yB + hB)

# Abre o vídeo
cap = cv2.VideoCapture(video_path)
# Cabeçalho do CSV com SlotIds
colunas = ["frame"] + [f"vaga{slot_id}" for slot_id, *_ in vagas]

# Abre CSV de saída para gravar predições
with open(csv_pred, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(colunas)

    frame_number = 0
    while True:
        ret, frame = cap.read()

        # Se não conseguir ler mais frames, fim do vídeo
        if not ret:
            break

        # Realiza a detecção com YOLO
        results = model(frame, stream=True)

        # lista de carros encontrados
        carros = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])   # tipo do objeto
                conf = float(box.conf[0])   # confiança da detecção

                # Detecta apenas carros com confiança > 0.5
                if model.names[cls] == "car" and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    carros.append((x1, y1, w, h))

        # Verifica ocupação por vaga
        linha = [frame_number]
        for slot_id, x, y, w, h in vagas:

            # Marca vaga como ocupada se intersectar com algum carro
            ocupada = any(intersects((x, y, w, h), carro) for carro in carros)

            # Adiciona o status da vaga à linha do frame
            linha.append(1 if ocupada else 0)

        # Escreve resultado do frame no CSV
        writer.writerow(linha)
        frame_number += 1

cap.release()

print(f"CSV de ocupação salvo como {csv_pred}")

# Chama o script de comparação de métricas
subprocess.run(["python", "comparar_metricas.py", csv_pred, csv_true, csv_out])
