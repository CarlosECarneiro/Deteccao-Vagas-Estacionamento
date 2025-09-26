import os
import subprocess
import sys
import cv2
import pandas as pd

# Caminho para o vídeo
video_path = sys.argv[1]
nome_video = os.path.splitext(os.path.basename(video_path))[0]

# Saída da detecção frame a frame
csv_pred = f"Output\\detec_canny_{nome_video}.csv"

# Coordenadas das vagas
vagas_path = f"Video\\vagas_{nome_video}.csv"
df_vagas = pd.read_csv(vagas_path)
vagas = [tuple(row) for row in df_vagas.values]


limiar_ocupacao = 0.07  # 7%

cap = cv2.VideoCapture(video_path)
frame_idx = 0

resultado_por_frame = []  # Lista para armazenar os dados

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    edges = cv2.Canny(frame_gray, 50, 150)

    linha = [frame_idx]  # Primeiro item da linha é o número do frame

    for (x, y, w, h) in vagas:
        roi_edges = edges[y:y+h, x:x+w]
        total_pixels = w * h
        bordas = cv2.countNonZero(roi_edges)
        ocupada = bordas > total_pixels * limiar_ocupacao
        linha.append(1 if ocupada else 0)  # 1 = ocupada, 0 = livre

    resultado_por_frame.append(linha)
    frame_idx += 1

cap.release()

# Salva resultados frame a frame
colunas = ["frame"] + [f"vaga_{i}" for i in range(len(vagas))]
df_frames = pd.DataFrame(resultado_por_frame, columns=colunas)
df_frames.to_csv(csv_pred, index=False)
print(f"CSV de ocupação salvo como {csv_pred}")

# Chama o script de comparação de métricas
# Argumentos: pred, ground truth, saída
csv_true = f"Video\\ground_truth_{nome_video}.csv"
csv_out  = f"Output\\canny_metrics_{nome_video}.csv"

subprocess.run(["python", "comparar_metricas.py", csv_pred, csv_true, csv_out])
