import os
import subprocess
import sys
import cv2
import csv
import pandas as pd

# Recebe o caminho do vídeo enviado pelo main
video_path = sys.argv[1]
nome_video = os.path.splitext(os.path.basename(video_path))[0]

# Define caminhos dos arquivos CSV
csv_pred = f"Output\\detec_canny_{nome_video}.csv"
vagas_path = f"Vagas\\vagas_{nome_video}.csv"
csv_true = f"Video\\ground_truth_{nome_video}.csv"
csv_out  = f"Output\\canny_metrics_{nome_video}.csv"

# Lê o CSV com coordenadas e IDs das vagas
df_vagas = pd.read_csv(vagas_path)

# Converte cada linha do CSV em coordenadas (x, y, largura, altura)
vagas = [(int(row["SlotId"]), int(row["X"]), int(row["Y"]),
            int(row["W"]), int(row["H"]))
            for _, row in df_vagas.iterrows()]

# Limiar para decidir se a vaga está ocupada (percentual de bordas)
limiar_ocupacao = 0.07  # 7% de pixels de borda 

# Abre o vídeo
cap = cv2.VideoCapture(video_path)

# Cabeçalho do CSV com SlotIds
colunas = ["frame"] + [f"vaga{slot_id}" for slot_id, *_ in vagas]

# Loop de detecção frame a frame
with open(csv_pred, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(colunas)

    frame_number = 0   # contador de frames
    while True:
        ret, frame = cap.read()

        # Se não conseguir ler mais frames, fim do vídeo
        if not ret:
            print("Fim do vídeo.")
            break
        
        # Converte o frame para cinza e aplica filtro gaussiano para reduzir ruído
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        # Aplica detecção de bordas no frame todo
        edges = cv2.Canny(frame_gray, 50, 150)

        linha = [frame_number]  # primeiro item = número do frame

        # Processa cada vaga individualmente
        for slot_id, x, y, w, h in vagas:

            # recorta apenas a vaga
            roi_edges = edges[y:y+h, x:x+w]

            total_pixels = w * h

            # conta pixels detectados
            bordas = cv2.countNonZero(roi_edges)

            # Decide se a vaga está ocupada com base no percentual de bordas
            ocupada = bordas > total_pixels * limiar_ocupacao

            # Adiciona o status da vaga à linha do frame
            linha.append(1 if ocupada else 0)

        # Escreve resultado do frame no CSV
        writer.writerow(linha)
        frame_number += 1

cap.release()

print(f"CSV de ocupação salvo como {csv_pred}")

# Chama o script de comparação de métricas
subprocess.run(["python", "comparar_metricas.py", csv_pred, csv_true, csv_out])
