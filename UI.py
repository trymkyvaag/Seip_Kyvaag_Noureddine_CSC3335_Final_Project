"""
    Authors: Jad Noureddine, Garald Seip
    Spring 2023
    CSC 3335
    
    This file implements the front-end GUI of our project.
    
    Inspired by: https://www.geeksforgeeks.org/gui-chat-application-using-tkinter-in-python/
"""

import re
import tkinter as tk
from Data_Storage import Data
from Tweet_Analysis import analyse_tweet

BLACK = "#0E1111"
GRAY = '#2A2A2A'
WHITE = "#FFFFFF"
BLUE = '#57C8FF'
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class CyberDetect:
    
    def __init__(self):
        self.model = analyse_tweet(True)
        self.data = Data()
        
        self.window = tk.Tk()
        self._setup_main_window()
        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("Tweet Analyzer")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BLACK)
        
        # head label
        head_label = tk.Label(self.window, bg=BLACK, fg=WHITE,
                              text="Tweet Analyzer", font=FONT_BOLD, pady=10)
        head_label.pack(side=tk.TOP, fill=tk.X)
        
        # tiny divider
        line = tk.Canvas(self.window, width=450, height=1, bg=WHITE)
        line.pack(pady=5)
        
        # chat display area
        chat_display_frame = tk.Frame(self.window)
        chat_display_frame.pack(side=tk.TOP, padx=5, pady=5)
        chat_display_frame.configure(bg=BLACK)
        
        self.text_widget = tk.Text(chat_display_frame, width=60, height=30, bg=BLACK, fg=WHITE,
                                    font='TkFixedFont', padx=10, pady=10)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text_widget.configure(cursor="arrow", state=tk.DISABLED)
        
        scrollbar = tk.Scrollbar(chat_display_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar.config(command=self.text_widget.yview)
        self.text_widget.config(yscrollcommand=scrollbar.set)
        
        # message entry box
        msg_entry_frame = tk.Frame(self.window)
        msg_entry_frame.pack(side=tk.BOTTOM, padx=5, pady=5)
        msg_entry_frame.configure(bg=BLACK)
        
        self.msg_entry = tk.Entry(msg_entry_frame, bg=GRAY, fg=WHITE, font=FONT)
        self.msg_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # send button
        send_button = tk.Button(msg_entry_frame, text="Submit", font=FONT_BOLD, width=20, bg=BLUE, fg=BLACK,
                                command=lambda: self._on_enter_pressed(None))
        send_button.pack(side=tk.LEFT, padx=10, pady=10)
     
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        if(re.sub(' ', '', msg) != ''):
        #     self._insert_message(data.clean_tweet(msg), "Tweet")
            self._insert_message(msg, "Tweet", False)
            
            analysis = self.model.analyze_tweet(msg, True)
            
            self._insert_message(msg, "Analysis")
        
    def _insert_message(self, to_display, sender, left: bool = True):
        self.text_widget.configure(state=tk.NORMAL)
        
        message = to_display
        # Underline hack.
        sender = sender + ' '
        
        # Left aligns the text for display.
        if(left):
            sender = "\u0332".join(sender)
            message = self.align(message, True)
        # Right aligns the text for display.
        else:
            orig_len = len(sender)
            sender = "\u0332".join(sender)
            
            # Makes it 60 characters long.
            while(orig_len < 60):
                orig_len += 1
                sender = ' ' + sender
                
            sender = '%60s' % sender
            
            message = self.align(message, False)
            
        self.text_widget.insert(tk.END, sender + "\n")
        
        # Displays message.
        self.text_widget.insert(tk.END, message + "\n\n")
        self.text_widget.configure(state=tk.DISABLED)
        self.msg_entry.delete(0, tk.END)
        
    def align(self, to_align: str, left: bool):
        """
        This function left/right aligns text for the GUI using a very disgusting feeling implementation.
        I don't like that I did this.

        Args:
            to_align (str): The text to aling.
            left (bool): To left align the text or not.

        Returns:
            str: Aligned text is returned.
        """
        temp = to_align
        final = []
        
        # Removes leading whitespace
        while(temp.startswith(' ') and len(temp) > 1):
            temp = temp[1:]
        
        # Disgusting hard coded numbers!
        while(len(temp) > 40):
            curr_slice = temp[:40]
            temp = temp[40:]
            
            index = curr_slice.rfind(' ')
            # Ew.
            if(index > -1 and index < 39):
                temp = curr_slice[index:] + temp
                curr_slice = curr_slice[:index]
            
            # Gross!
            if(not left):
                curr_slice = '                    ' + curr_slice
            final.append(curr_slice)
        
            # Removes leading whitespace
            while(temp.startswith(' ') and len(temp) > 1):
                temp = temp[1:]
        
        # Bad!
        if(not left):
            temp = '                    ' + temp
        final.append(temp)
        
        return '\n'.join(final)

if __name__ == "__main__":
    app = CyberDetect()
    app.run()