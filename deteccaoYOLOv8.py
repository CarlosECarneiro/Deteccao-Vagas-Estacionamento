import time
import tkinter as tk
from tkinter import messagebox
import os
import sys
import cv2
import pandas as pd
from ultralytics import YOLO

# Caminho do vídeo
video_path = sys.argv[1]
nome_video = os.path.splitext(os.path.basename(video_path))[0]

# Carrega YOLOv8 COCO pré-treinado
model = YOLO("yolov8s.pt") 

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

tempos = []  # lista para armazenar tempo de cada frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo.")
        break
    
    start = time.time()  # início do processamento do frame
    
    # Faz a inferência
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
                # desenha caixas dos carros
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"Car {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Checa vagas
    for i, vaga in enumerate(vagas):
        ocupada = any(intersects(vaga, carro) for carro in carros)
        color = (0, 0, 255) if ocupada else (0, 255, 0)
        status = "Ocupada" if ocupada else "Livre"
        x, y, w, h = vaga
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, status, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    frame = cv2.resize(frame, (960, 352))
    cv2.imshow("YOLOv8 - Estacionamento", frame)
    tempos.append(time.time() - start)  # tempo decorrido

    if cv2.waitKey(30) & 0xFF == 27:  # ESC para sair
        break
    if cv2.getWindowProperty("YOLOv8 - Estacionamento", cv2.WND_PROP_VISIBLE) < 1:
        print("Janela foi fechada.")
        break

cap.release()

tempo_medio = sum(tempos) / len(tempos) if tempos else 0

mensagem_tempo = f"Tempo médio por frame: {tempo_medio:.4f} segundos"

root = tk.Tk()
root.withdraw()
messagebox.showinfo("Desempenho", mensagem_tempo)

cv2.destroyAllWindows()
