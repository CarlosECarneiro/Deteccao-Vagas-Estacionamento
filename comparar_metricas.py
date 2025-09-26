import sys
import pandas as pd
from sklearn.metrics import confusion_matrix

# Arquivos
csv_pred = sys.argv[1] # saídas da detecção
csv_true = sys.argv[2] # anotações manuais
csv_out  = sys.argv[3] # saída com métricas

# Lê CSVs
df_pred = pd.read_csv(csv_pred)
df_true = pd.read_csv(csv_true)

# Remove coluna "frame" (mantém só as vagas)
y_pred = df_pred.drop(columns=["frame"]).values.flatten()
y_true = df_true.drop(columns=["frame"]).values.flatten()

# Calcula matriz de confusão
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

# Métricas
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Cria DataFrame com os resultados
df_result = pd.DataFrame([{
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}])
# Salva em CSV
df_result.to_csv(csv_out, index=False)

# Exibe no terminal (opcional)
print("TP,TN,FP,FN,Accuracy,Precision,Recall,F1-Score")
print(f"{tp},{tn},{fp},{fn},{accuracy},{precision},{recall},{f1}")

# Mostra em janela com tkinter
import tkinter as tk
from tkinter import messagebox

mensagem = (
    f"TP: {tp}\n"
    f"TN: {tn}\n"
    f"FP: {fp}\n"
    f"FN: {fn}\n\n"
    f"Accuracy:  {accuracy:.4f}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall:    {recall:.4f}\n"
    f"F1-Score:  {f1:.4f}"
)

root = tk.Tk()
root.withdraw()
messagebox.showinfo("Métricas de Avaliação", mensagem)