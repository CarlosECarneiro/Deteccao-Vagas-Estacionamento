import os
import sys
import time
import cv2
import pandas as pd
import tkinter as tk
from tkinter import messagebox

# Caminho do vídeo
video_path = sys.argv[1]
nome_video = os.path.splitext(os.path.basename(video_path))[0]

# Coordenadas das vagas
vagas_path = f"Video\\vagas_{nome_video}.csv"
df_vagas = pd.read_csv(vagas_path)
vagas = [tuple(row) for row in df_vagas.values]


limiar_ocupacao = 0.07  # 7% de pixels de borda 

# Abre o vídeo
cap = cv2.VideoCapture(video_path)

tempos = []  # lista para armazenar tempo de cada frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo.")
        break

    start = time.time()  # início do processamento do frame

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    # Detecta bordas na imagem toda
    edges = cv2.Canny(frame_gray, 50, 150)

    frame_out = frame.copy()

    for idx, (x, y, w, h) in enumerate(vagas):
        roi_edges = edges[y:y+h, x:x+w]

        total_pixels = w * h
        bordas = cv2.countNonZero(roi_edges)

        ocupada = bordas > total_pixels * limiar_ocupacao

        cor = (0, 0, 255) if ocupada else (0, 255, 0)
        texto = "Ocupada" if ocupada else "Livre"
        cv2.rectangle(frame_out, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(frame_out, f"{texto}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    frame_out = cv2.resize(frame_out, (960, 352))
    cv2.imshow("Vagas - Canny", frame_out)
    tempos.append(time.time() - start)  # tempo decorrido

    if cv2.waitKey(30) & 0xFF == 27:  # ESC para sair
        break
    if cv2.getWindowProperty("Vagas - Canny", cv2.WND_PROP_VISIBLE) < 1:
        print("Janela foi fechada.")
        break

cap.release()

tempo_medio = sum(tempos) / len(tempos) if tempos else 0

mensagem_tempo = f"Tempo médio por frame: {tempo_medio:.4f} segundos"

root = tk.Tk()
root.withdraw()
messagebox.showinfo("Desempenho", mensagem_tempo)

cv2.destroyAllWindows()
