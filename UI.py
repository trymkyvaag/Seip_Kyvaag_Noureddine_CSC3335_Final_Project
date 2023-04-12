
from tkinter import *
class UI():

    #Inspired from: https://www.geeksforgeeks.org/gui-chat-application-using-tkinter-in-python/
    def __init__(self) -> None:
        # GUI
        window = Tk()

        window.geometry("700x600")

        # Create title label
        title_label = Label(window, text="PC AI")
        title_label.pack(anchor='n')

        # Create title entry
        title_entry = Entry(window, width=100)
        title_entry.pack(anchor='sw')
        window.mainloop()
   


ui = UI()