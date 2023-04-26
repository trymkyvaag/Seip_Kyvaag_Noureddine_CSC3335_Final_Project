"""
    Authors: Jad Noureddine, Garald Seip
    Spring 2023
    CSC 3335
    
    This file implements the front-end GUI of our project.
    
    Inspired by: https://www.geeksforgeeks.org/gui-chat-application-using-tkinter-in-python/
"""

import tkinter as tk

BG_COLOR = "#0E1111"
TEXT_COLOR = "#FFFFFF"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class CyberDetect:
    
    def __init__(self):
        self.window = tk.Tk()
        self._setup_main_window()
        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("Cyberbullying Detector")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)
        
        # head label
        head_label = tk.Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                              text="Cyberbullying Detector", font=FONT_BOLD, pady=10)
        head_label.pack(side=tk.TOP, fill=tk.X)
        
        # tiny divider
        line = tk.Canvas(self.window, width=450, height=1, bg=TEXT_COLOR)
        line.pack(pady=5)
        
        # chat display area
        chat_display_frame = tk.Frame(self.window)
        chat_display_frame.pack(side=tk.TOP, padx=5, pady=5)
        chat_display_frame.configure(bg=BG_COLOR)
        
        self.text_widget = tk.Text(chat_display_frame, width=50, height=25, bg=BG_COLOR, fg=TEXT_COLOR,
                                    font=FONT, padx=10, pady=10)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text_widget.configure(cursor="arrow", state=tk.DISABLED)
        
        scrollbar = tk.Scrollbar(chat_display_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar.config(command=self.text_widget.yview)
        self.text_widget.config(yscrollcommand=scrollbar.set)
        
        # message entry box
        msg_entry_frame = tk.Frame(self.window)
        msg_entry_frame.pack(side=tk.BOTTOM, padx=5, pady=5)
        msg_entry_frame.configure(bg=BG_COLOR)
        
        self.msg_entry = tk.Entry(msg_entry_frame, bg=TEXT_COLOR, fg=BG_COLOR, font=FONT)
        self.msg_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # send button
        send_button = tk.Button(msg_entry_frame, text="Submit", font=FONT_BOLD, width=20, bg=TEXT_COLOR, fg=BG_COLOR,
                                command=lambda: self._on_enter_pressed(None))
        send_button.pack(side=tk.LEFT, padx=10, pady=10)
     
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        if()
        self._insert_message(msg, "Tweet")
        
    def _insert_message(self, message, sender):
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, sender + ": " + message + "\n\n")
        self.text_widget.configure(state=tk.DISABLED)
        self.msg_entry.delete(0, tk.END)

if __name__ == "__main__":
    app = CyberDetect()
    app.run()