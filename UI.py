
import tkinter as tk
from tkinter import *

# Create the main window
root = tk.Tk()

# Set the title of the window
root.title("Cyberbullying Detector")

# Set the size of the window
root.geometry("850x850")

panel_1 = PanedWindow(bd=4, relief="raised", bg="red")
panel_1.pack(fill=BOTH, expand=1)

label = Label(panel_1, text="Enter Tweet")
panel_1.add(label)

textField1 = Entry()
panel_1.add(textField1)

panel_2 = PanedWindow(panel_1, orient=VERTICAL, bd=4, relief="raised", bg="blue")
panel_1.add(panel_2)

top = Label(panel_2, text="Top Panel")
panel_2.add(top)

bottom = Label(panel_2, text="Bottom Panel")
panel_2.add(bottom)


# Start the GUI event loop
root.mainloop()