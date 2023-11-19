import PySimpleGUI as sg

# Sample data (you will replace this with your database results)
sample_results = [
    {"method": "MLS_SMC", "T": 10, "N": 100, "alpha": 0.95, "ess_alpha": 1000, "probability_of_failure": 0.1},
    {"method": "MALA_SMC", "T": 20, "N": 200, "alpha": 0.90, "ess_alpha": 1500, "probability_of_failure": 0.05},
    {"method": "IS", "N": 50, "batch_size": 10, "probability_of_failure": 0.2},
]

# Define the layout of the PySimpleGUI window
layout = [
    [sg.Table(values=[], headings=["Method", "T", "N", "Alpha", "ESS Alpha", "Batch Size", "Probability of Failure"],
              auto_size_columns=False, justification="right",
              num_rows=min(25, len(sample_results)))],
    [sg.Exit()]
]

# Create the PySimpleGUI window
window = sg.Window("Reliability Estimation Results", layout, resizable=True)

# Populate the table with sample data
table = window['Table']
table.update(values=sample_results)

# Event loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break

window.close()
