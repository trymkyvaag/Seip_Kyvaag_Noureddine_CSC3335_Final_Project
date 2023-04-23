
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

textField1 = Entry()
textField1.pack(fill='x')

enterButton = Button(text="Submit")
enterButton.pack()

panel_2 = PanedWindow(panel_1, orient=VERTICAL, bd=4, relief="raised", bg="blue")
panel_1.add(panel_2)

top = Label(panel_2, text="Status Panel")
panel_2.add(top)

bottom = Label(panel_2, text="Display Panel")
panel_2.add(bottom)

quitButton = Button(text="Quit")
quitButton.pack()

# Start the GUI event loop
root.mainloop()