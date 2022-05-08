"""Functions to handle I/O operations."""

import csv
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

def ask_csv_location():
    """Ask user to select a CSV file from the file system."""

    return filedialog.askopenfilename(
        title = 'Select CSV file',
        filetypes = (("CSV Files","*.csv"),),
    )

def read_csv(csv_path:str=None) -> list[list]:
    """Read CSV file and return data as a list of lists.
    If no path is provided, ask user to select a CSV file."""

    path = csv_path if csv_path else ask_csv_location()
    with open(path, 'r') as csv_file:
        csv_data = list(csv.reader(csv_file))
        return csv_data
