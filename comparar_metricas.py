import sys
import pandas as pd
from sklearn.metrics import confusion_matrix
import tkinter as tk
from tkinter import messagebox

# Recebe o caminho dos arquivos CSV
csv_pred = sys.argv[1]  # Saída da detecção (predito)
csv_true = sys.argv[2]  # Ground truth (real)
csv_out  = sys.argv[3]  # Saída com métricas

# Lê os CSV da detecção e o ground truth
df_pred = pd.read_csv(csv_pred)
df_true = pd.read_csv(csv_true)

# Remove a coluna "frame"
df_pred = df_pred.drop(columns=["frame"], errors="ignore")
df_true = df_true.drop(columns=["frame"], errors="ignore")

# Garante que as colunas estão na mesma ordem
colunas_comuns = sorted(set(df_pred.columns) & set(df_true.columns))
if not colunas_comuns:
    raise ValueError("Nenhuma vaga em comum entre os arquivos predito e ground truth!")

# Redefine com a mesma ordem
df_pred = df_pred[colunas_comuns]
df_true = df_true[colunas_comuns]

# Converte todos os valores em um vetor 1D
y_pred = df_pred.values.flatten()
y_true = df_true.values.flatten()

# Calcula a matriz de confusão: TN, FP, FN, TP
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

# Calcula métricas de classificação
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Salva resultados
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

# Salva métricas no CSV de saída
df_result.to_csv(csv_out, index=False)

# Exibe resultados
print("TP,TN,FP,FN,Accuracy,Precision,Recall,F1-Score")
print(f"{tp},{tn},{fp},{fn},{accuracy},{precision},{recall},{f1}")

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

# Exibe mensagem em janela do Tkinter
root = tk.Tk()
root.withdraw()
messagebox.showinfo("Métricas de Avaliação", mensagem)
