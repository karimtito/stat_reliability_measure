import argparse
import tkinter as tk
from tkinter import ttk
import pandas as pd
# Create argument parser

class GUI_Config:
    dataset = 'mnist'
    model = 'dnn2'
    def __init__(self, dataset=dataset, model=model):
        self.dataset = dataset
        self.model = model
        
        
gui_config = GUI_Config()
parser = argparse.ArgumentParser(description='Process dataset and model.')
parser.add_argument('-dataset', type=str, help='Path to dataset')
parser.add_argument('-model', type=str, help='Path to model')

for key, value in parser.parse_args()._get_kwargs():
    setattr(GUI_Config, key, value)
    

# Sample data (you will replace this with your database results)
res_df = pd.read_csv(f'logs/exp_model_{gui_config.dataset}/aggr_res.csv')

# Function to populate the treeview widget with data
def populate_treeview(results,nb_rows=25):
    
    for i,result in enumerate(results):
        if i>=nb_rows:
            break
        tree.insert("", "end", values=(
            result["method_name"],
            result.get("T", ""),
            result.get("N", ""),
            result.get("alpha", ""),
            result.get("ess_alpha", ""),
            result.get("batch_size", ""),
            result.get("mean_est",""),
            result.get("std_est","")
        ))

# Create the main window
root = tk.Tk()
root.title("Reliability Estimation Results")

# Create a treeview widget
tree = ttk.Treeview(root, columns=("Method", "T", "N", "Alpha", "ESS Alpha", "Batch Size", "Probability of Failure","Std Est"), show="headings")
tree.heading("Method", text="Method")
tree.heading("T", text="T")
tree.heading("N", text="N")
tree.heading("Alpha", text="Alpha")
tree.heading("ESS Alpha", text="ESS Alpha")
tree.heading("Batch Size", text="Batch Size")
tree.heading("Probability of Failure", text="Probability of Failure")
tree.heading("Std Est", text="Std Est")
tree.pack()

# Populate the treeview with sample data
populate_treeview(res_df.to_dict(orient='records'))

# Start the Tkinter main loop
root.mainloop()
