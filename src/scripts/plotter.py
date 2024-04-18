import pandas as pd
import plotly as pl
import numpy as np
from datasets.hyperion_dataset import COLS_ALLOWED_PLOTTER

class Plotter():
   def __init__(self, path) -> None:
      #genererici controlli sul path passato
      if self.excel_check(path):
         self.df = self.create_df(path)

    # 0) lettura file excel caricato -> controllo colonne (Testo, Stralcio, Repertorio)
    # 1) creazione dataframe con pandas a partire da path excel
    # 2) pulizia/preparazione dataframe (tenere solo le colonne di indice, domanda(meglio se numero), età, genere, ruolo, repertorio, arcipelago)
    #                                   (per ogni riga bianca ripetere il contenuto della precedente)
    # 3) tutti i metodi per i vari grafici
   def excel_check(path):
      pass

   def create_df(self, path):
      df = pd.read_excel(path)
      df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
      # crea una nuova colonna "num_risposta" in cui scorre la colonna testo e per ogni valore letto diverso da NaN incrementa il contatore e inserisce il valore, altrimenti ri-inserisce il valore precedente
      num_risposta = []
      index = 0
      for i in df['Testo'].isna():
         if i == False:
            index = index +1
         num_risposta.append(index)
      df['num_risposta'] = num_risposta
      # riduce le colonne a: num_risposta, 'num_domanda', 'Età', 'Genere', 'Ruolo', 'Repertorio', 'Arcipelago'
      list_col=[]
      for i in df.columns:
         if i in COLS_ALLOWED_PLOTTER:
            list_col.append(i)
         new_df = df[list_col]
      # scorre colonne 'num_domanda', 'Età', 'Genere', 'Ruolo' e sostituisce tutti i valori NaN con il valore della cella precedente
      for column_name in new_df.columns:
         new_df[column_name] = self.replace_nan_with_previous(new_df[column_name])
      return new_df
   
   def replace_nan_with_previous(column):
      previous_value = None
      for i in range(len(column)):
         if pd.isna(column[i]):
               column[i] = previous_value
         else:
               previous_value = column[i]
      return column

   def plot_peso(self):pass