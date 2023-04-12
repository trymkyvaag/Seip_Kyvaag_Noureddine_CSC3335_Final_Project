
from tkinter import *
class UI():

    def __init__(self) -> None:
        root = Tk()
        root.title("Chatbot")
        root.geometry("720x500")
        
        BG_GRAY = "#ABB2B9"
        BG_COLOR = "#17202A"
        TEXT_COLOR = "#EAECEE"
        
        FONT = "Helvetica 14"
        FONT_BOLD = "Helvetica 13 bold"

        lable1 = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="Welcome to: Should I tweet this?", font=FONT_BOLD, pady=10, width=20, height=1).grid(
    row=0)
 
        txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=360)
        txt.grid(row=1, column=0, columnspan=2)
        
        scrollbar = Scrollbar(txt)
        scrollbar.place(relheight=1, relx=0.974)
        root.mainloop()
   
    def make_entry(self, label, var):
        l = Label(self.top, text=label)
        l.grid(row=self.row, column=0, sticky="nw")
        e = Entry(self.top, textvariable=var, exportselection=0)
        e.grid(row=self.row, column=1, sticky="nwe")
        self.row = self.row + 1
        return e
    
    def make_frame(self,labeltext=None):
        if labeltext:
            l = Label(self.top, text=labeltext)
            l.grid(row=self.row, column=0, sticky="nw")
        f = Frame(self.top)
        f.grid(row=self.row, column=1, columnspan=1, sticky="nwe")
        self.row = self.row + 1
        return f

ui = UI()