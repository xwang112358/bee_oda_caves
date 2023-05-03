import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageEditor:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Editor")

        # Create the canvas and scrollbar widgets
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_h = tk.Scrollbar(self.master, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        self.scrollbar_v = tk.Scrollbar(self.master, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(xscrollcommand=self.scrollbar_h.set, yscrollcommand=self.scrollbar_v.set)
        self.canvas.bind('<Configure>', self.on_canvas_configure)

        # Create the "Load Image" button and the zoom buttons
        tk.Button(self.master, text="Load Image", command=self.load_image).pack()
        tk.Button(self.master, text="Zoom In", command=lambda: self.zoom(1.2)).pack()
        tk.Button(self.master, text="Zoom Out", command=lambda: self.zoom(0.8)).pack()

        # Initialize the image and zoom level variables
        self.img = None
        self.photo_img = None
        self.zoom_level = 1.0

    def load_image(self):
        # Prompt the user to select an image file
        file_path = filedialog.askopenfilename()
        if file_path:
            # Load the image and convert it to a Tkinter PhotoImage object
            img = Image.open(file_path)
            self.img = img.copy()
            self.photo_img = ImageTk.PhotoImage(img)
            self.display_image()

    def display_image(self):
        # Display the image on the canvas, scaled to the current zoom level
        if self.photo_img:
            self.canvas.delete('all')
            self.canvas.create_image(0, 0, image=self.photo_img, anchor=tk.NW)
            self.canvas.scale('all', 0, 0, self.zoom_level, self.zoom_level)

            # Update the scrollbars
            self.canvas.config(scrollregion=self.canvas.bbox('all'))

    def on_canvas_configure(self, event):
        # Update the scrollbars when the canvas is resized
        self.canvas.config(scrollregion=self.canvas.bbox('all'))

    def zoom(self, factor):
        # Zoom in or out by the specified factor
        self.zoom_level *= factor
        self.photo_img = ImageTk.PhotoImage(self.img.resize((int(self.img.width*self.zoom_level), int(self.img.height*self.zoom_level))))
        self.display_image()

# Create the Tkinter window and start the event loop
root = tk.Tk()
app = ImageEditor(root)
root.mainloop()



