import time
import tkinter as tk
from tkinter import messagebox
import os
import sys
import cv2
import pandas as pd
from ultralytics import YOLO

# Recebe o caminho do vídeo enviado pelo main
video_path = sys.argv[1]
nome_video = os.path.splitext(os.path.basename(video_path))[0]

# Carrega o modelo YOLOv8 pré-treinado
model = YOLO("yolov8s.pt") 

# Caminho do arquivo CSV com as coordenadas das vagas
vagas_path = f"Vagas\\vagas_{nome_video}.csv"

# Lê o CSV com as vagas
df_vagas = pd.read_csv(vagas_path)

# Converte cada linha do CSV em coordenadas (x, y, largura, altura)
vagas = [(int(row["X"]), int(row["Y"]),
            int(row["W"]), int(row["H"]))
            for _, row in df_vagas.iterrows()]


# Função de interseção entre caixas
def intersects(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    return not (xA + wA < xB or xA > xB + wB or yA + hA < yB or yA > yB + hB)

# Obter dimensões da tela
root = tk.Tk()
largura_tela = root.winfo_screenwidth()
altura_tela = root.winfo_screenheight()
root.destroy()

# Função para redimensionar tela
def redimensionar_para_tela(frame):
    altura, largura = frame.shape[:2]
    escala = min(largura_tela / largura * 0.9, altura_tela / altura * 0.9, 1.0)
    nova_largura = int(largura * escala)
    nova_altura = int(altura * escala)
    return cv2.resize(frame, (nova_largura, nova_altura))

# Abre o vídeo
cap = cv2.VideoCapture(video_path)

# Lista para guardar o tempo de processamento de cada frame
tempos = [] 

while True:
    ret, frame = cap.read()

    # Se não conseguir ler mais frames, fim do vídeo
    if not ret:
        print("Fim do vídeo.")
        break
    
    # Marca o início do tempo de processamento do frame
    start = time.time()
    
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

                # desenha caixas dos carros
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"Car {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Verifica cada vaga do estacionamento
    for i, vaga in enumerate(vagas):
        # Marca como ocupada se qualquer carro intersecta a vaga
        ocupada = any(intersects(vaga, carro) for carro in carros)
        cor = (0, 0, 255) if ocupada else (0, 255, 0)
        texto = "Ocupada" if ocupada else "Livre"
        x, y, w, h = vaga

        # Desenha a caixa da vaga e escreve o texto
        cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(frame, texto, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    # Redimensiona o frame para caber na tela
    frame = redimensionar_para_tela(frame)

    # Mostra o vídeo com as vagas processadas
    cv2.imshow("YOLOv8 - Estacionamento", frame)

    # Guarda o tempo de processamento do frame
    tempos.append(time.time() - start)  

    if cv2.waitKey(30) & 0xFF == 27:  # ESC para sair
        break
    if cv2.getWindowProperty("YOLOv8 - Estacionamento", cv2.WND_PROP_VISIBLE) < 1:
        print("Janela foi fechada.")
        break

cap.release()

# Calcula o tempo médio por frame
tempo_medio = sum(tempos) / len(tempos) if tempos else 0

mensagem_tempo = f"Tempo médio por frame: {tempo_medio:.4f} segundos"

root = tk.Tk()
root.withdraw()
messagebox.showinfo("Desempenho", mensagem_tempo)

cv2.destroyAllWindows()
