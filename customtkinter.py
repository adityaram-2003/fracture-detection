# customtkinter.py

from tkinter import Label, Button, Frame, PhotoImage, font


class CTk(Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)


class CTkFrame(Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)


class CTkLabel(Label):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)


class CTkButton(Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)


class CTkFont(font.Font):
    def __init__(self, family, size):
        super().__init__(family=family, size=size)


class CTkImage:
    def __init__(self, image):
        self.image = image
