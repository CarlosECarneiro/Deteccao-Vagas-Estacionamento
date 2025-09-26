import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

# Caminho padrão da pasta de vídeos
CAMINHO_VIDEOS = os.path.join("Video")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Processamento de Vídeo com YOLO")

        self.video_path = ""
        self.operacao = tk.StringVar(value="detectar")

        # Botão para selecionar vídeo
        self.btn_selecionar = tk.Button(root, text="Selecionar Vídeo", command=self.selecionar_video)
        self.btn_selecionar.pack(pady=10)

        # Label para exibir vídeo selecionado
        self.label_video = tk.Label(root, text="Nenhum vídeo selecionado", fg="gray")
        self.label_video.pack()

        # Opções de operação
        self.radio1 = tk.Radiobutton(root, text="Detectar carros com Canny", variable=self.operacao, value="detectarCanny")
        self.radio2 = tk.Radiobutton(root, text="Detectar carros com YOLO", variable=self.operacao, value="detectarYOLO")
        self.radio3 = tk.Radiobutton(root, text="Calcular Métricas Canny", variable=self.operacao, value="csvCanny")
        self.radio4 = tk.Radiobutton(root, text="Calcular Métricas YOLO", variable=self.operacao, value="csvYOLO")
        self.radio1.pack()
        self.radio2.pack()
        self.radio3.pack()
        self.radio4.pack()

        # Botão para executar
        self.btn_executar = tk.Button(root, text="Executar", command=self.executar_operacao)
        self.btn_executar.pack(pady=20)

    def selecionar_video(self):
        arquivo = filedialog.askopenfilename(initialdir=CAMINHO_VIDEOS, title="Selecionar Vídeo",
                                             filetypes=(("Arquivos de vídeo", "*.mp4 *.avi *.mov"), ("Todos os arquivos", "*.*")))
        if arquivo:
            self.video_path = arquivo
            self.label_video.config(text=os.path.basename(arquivo), fg="black")

    def executar_operacao(self):
        if not self.video_path:
            messagebox.showwarning("Atenção", "Por favor, selecione um vídeo.")
            return

        operacao = self.operacao.get()
        
        if operacao == "detectarCanny":
            script = "deteccaoCanny.py"
        elif operacao == "detectarYOLO":
            script = "deteccaoYOLOv8.py"
        elif operacao == "csvCanny":
            script = "calculoCanny.py"
        elif operacao == "csvYOLO":
            script = "calculoYOLO.py"
        else:
            messagebox.showerror("Erro", "Operação desconhecida.")
            return

        try:
            subprocess.Popen(["python", script, self.video_path])
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Erro", f"Erro ao executar o script:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
