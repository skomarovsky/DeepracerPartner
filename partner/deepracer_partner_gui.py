import tkinter as tk
from tkinter import Menu, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import deepracer_utility as du
from PIL import Image, ImageTk


class DeepRacerPartnerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepRacer Partner")
        self.root.geometry("1024x768")  # Initial window size
        self.root.minsize(800, 600)  # Minimum window size

        self.current_file = None

        # Create a menu
        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)

        # File menu
        self.file_menu = Menu(self.menu)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open Track", command=self.open_file)
        self.export_track_menu = self.file_menu.add_command(label="Export Track to CSV", command=self.export_track,
                                                            state='disabled')
        self.export_segments_menu = self.file_menu.add_command(label="Export Segments to CSV",
                                                               command=self.export_segments, state='disabled')
        self.file_menu.add_command(label="Exit", command=self.exit_app)

        # Action menu
        self.action_menu = Menu(self.menu)
        self.menu.add_cascade(label="Action", menu=self.action_menu)
        self.show_segments_menu = self.action_menu.add_command(label="Show Segments", command=self.show_segments,
                                                               state='disabled')
        self.hide_segments_menu = self.action_menu.add_command(label="Hide Segments", command=self.hide_segments,
                                                               state='disabled')

        # Help menu
        help_menu = Menu(self.menu)
        self.menu.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_help)

        # Create a canvas for the background image
        self.bg_canvas = tk.Canvas(root, width=1024, height=768)
        self.bg_canvas.pack(fill="both", expand=True)

        # Create a canvas for plotting
        self.plot_canvas = tk.Canvas(root, width=1024, height=768, bg='white')
        self.plot_canvas.pack(fill="both", expand=True)

        # Place plot canvas above the background canvas
        self.plot_canvas.lift(self.bg_canvas)

        # Load and display the background image
        self.show_background_image()

        # Placeholder for plot
        self.figure, self.ax = plt.subplots(figsize=(10, 10))
        self.plot_widget = FigureCanvasTkAgg(self.figure, master=self.plot_canvas)
        self.plot_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Track resizing
        self.root.bind('<Configure>', self.on_resize)

    def show_background_image(self):
        self.bg_image_original = Image.open("DeepRacer_Partner_Background_1080p.png")
        self.bg_image = ImageTk.PhotoImage(self.bg_image_original)
        self.bg_image_id = self.bg_canvas.create_image(0, 0, image=self.bg_image, anchor="nw")

    def on_resize(self, event):
        self.update_background_image(event.width, event.height)
        if hasattr(self, 'plot_widget'):
            self.plot_widget.get_tk_widget().config(width=event.width, height=event.height)

    def update_background_image(self, width, height):
        resized_image = self.bg_image_original.resize((width, height), Image.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(resized_image)
        self.bg_canvas.itemconfig(self.bg_image_id, image=self.bg_image)

    def create_plot(self):
        if hasattr(self, 'plot_widget'):
            self.plot_widget.get_tk_widget().destroy()
        self.figure, self.ax = plt.subplots(figsize=(10, 10))
        self.plot_widget = FigureCanvasTkAgg(self.figure, master=self.plot_canvas)
        self.plot_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if file_path:
            self.current_file = file_path
            self.create_plot()
            self.plot_track()
            self.action_menu.entryconfig("Show Segments", state='normal')
            self.file_menu.entryconfig(self.export_track_menu, state='normal')
            self.file_menu.entryconfig(self.export_segments_menu, state='normal')

    def plot_track(self):
        if self.current_file:
            self.ax.clear()
            self.ax.grid(True)  # Add the grid
            du.plot_track(self.current_file, self.plot_widget)
            self.plot_widget.draw()

    def show_segments(self):
        if self.current_file:
            self.ax.clear()
            self.create_plot()  # Recreate the plot canvas for segments
            self.ax.grid(True)  # Add the grid
            du.plot_track_segments(self.current_file, self.plot_widget)
            self.plot_widget.draw()
            self.action_menu.entryconfig("Show Segments", state='disabled')
            self.action_menu.entryconfig("Hide Segments", state='normal')

    def hide_segments(self):
        if self.current_file:
            self.ax.clear()
            self.create_plot()  # Recreate the plot canvas for track
            self.ax.grid(True)  # Add the grid
            du.plot_track(self.current_file, self.plot_widget)
            self.plot_widget.draw()
            self.action_menu.entryconfig("Show Segments", state='normal')
            self.action_menu.entryconfig("Hide Segments", state='disabled')

    def export_track(self):
        if self.current_file:
            output_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if output_path:
                du.save_track_to_csv(self.current_file, output_path)

    def export_segments(self):
        if self.current_file:
            output_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if output_path:
                du.save_segments_to_csv(self.current_file, output_path)

    def show_help(self):
        messagebox.showinfo("Deepracer Partner", "Deepracer Partner by S. Komarovsky")

    def exit_app(self):
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = DeepRacerPartnerApp(root)
    root.mainloop()