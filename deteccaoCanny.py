import os
import sys
import time
import cv2
import pandas as pd
import tkinter as tk
from tkinter import messagebox

# Recebe o caminho do vídeo enviado pelo main
video_path = sys.argv[1]
nome_video = os.path.splitext(os.path.basename(video_path))[0]

# Caminho do arquivo CSV com as coordenadas das vagas
vagas_path = f"Vagas\\vagas_{nome_video}.csv"

# Lê o CSV com as vagas
df_vagas = pd.read_csv(vagas_path)

# Converte cada linha do CSV em coordenadas (x, y, largura, altura)
vagas = [(int(row["X"]), int(row["Y"]),
            int(row["W"]), int(row["H"]))
            for _, row in df_vagas.iterrows()]

# Limiar para decidir se a vaga está ocupada (percentual de bordas)
limiar_ocupacao = 0.07  # 7% de pixels de borda 

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

    # Converte o frame para cinza e aplica filtro gaussiano para reduzir ruído
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    # Aplica detecção de bordas no frame todo
    edges = cv2.Canny(frame_gray, 50, 150)

    # Cópia para desenhar as caixas de vagas
    frame_out = frame.copy()

    # Processa cada vaga individualmente
    for idx, (x, y, w, h) in enumerate(vagas):

        # recorta apenas a vaga
        roi_edges = edges[y:y+h, x:x+w]

        total_pixels = w * h

        # conta pixels detectados
        bordas = cv2.countNonZero(roi_edges) 

        # Decide se a vaga está ocupada com base no percentual de bordas
        ocupada = bordas > total_pixels * limiar_ocupacao

        # Define cor e texto para a vaga
        cor = (0, 0, 255) if ocupada else (0, 255, 0)
        texto = "Ocupada" if ocupada else "Livre"

        # Desenha a caixa da vaga e escreve o texto
        cv2.rectangle(frame_out, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(frame_out, f"{texto}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    # Redimensiona o frame para caber na tela
    frame_out = redimensionar_para_tela(frame_out)

    # Mostra o vídeo com as vagas processadas
    cv2.imshow("Vagas - Canny", frame_out)

    # Guarda o tempo de processamento do frame
    tempos.append(time.time() - start)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC para sair
        break
    if cv2.getWindowProperty("Vagas - Canny", cv2.WND_PROP_VISIBLE) < 1:
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
