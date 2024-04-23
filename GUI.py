import tkinter as tk
from tkinter import ttk
from Core import AnalyzeTrackCore
import time
  # GLobal variable for multiprocessor tracking calculation progress

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.patches as patches

from tkinter import filedialog, messagebox
import os
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
#import av
#import itertools
import threading
import cv2
import tifffile


class VideoPlayer(tk.Toplevel):
    def __init__(self, gui, filePath, master=None):
        super().__init__(master)
        self.title("Film Player")
        self.gui = gui
        self.core = self.gui.core
        self.threshold = tk.StringVar(value="0")
        self.frame_number = tk.IntVar()

        with tifffile.TiffFile(filePath) as tif:
            # Extract frames from pages
            self.tiff_stack = [page.asarray() for page in tif.pages]
            self.tiff_stack = np.array(self.tiff_stack)

        self.total_frames, self.frame_height, self.frame_width = self.tiff_stack.shape[:3]

        #self.cap = av.open(filePath, "r")
        #self.vidstream = self.cap.decode(video=0)
        #self.video_stream = self.cap.streams.video[0]
        #self.total_frames = self.cap.streams.video[0].frames
        #self.frame_width = self.cap.streams.video[0].width
        #self.frame_height = self.cap.streams.video[0].height



        self.frame_number = tk.IntVar(value=1)
        self.zoom_factor = 1.0
        self.zoom_factor_film = 1.0
        self.percentage_y = 0
        self.percentage_x = 0
        self.new_x = 0
        self.new_y = 0



        # Create GUI components

        self.create_widgets()


    def create_widgets(self):
        # Create a canvas for displaying the frames



        self.canvas = tk.Canvas(self, width=self.frame_width, height=self.frame_height)
        self.canvas.pack()


        self.label = tk.Label(self, text="Number of the track:")
        self.label.pack(side = "top")

        # Entry
        self.track_number_entry = tk.Entry(self)
        self.track_number_entry.pack(side = "top")

        # Create a label to display the frame number
        self.reset_zoom = tk.Button(self, text="Zoom reset", command=self.zoom_reset)
        self.reset_zoom.pack()



        # Button
        self.visualize_button = tk.Button(self, text="Visualize Track", command=self.get_draw_track)
        self.visualize_button.pack(pady=10)

        button1 = ttk.Button(self, text="Informations", command=lambda: self.open_sub_window())
        button1.pack(pady=10)


        self.frame_label = tk.Label(self, text="Frame: 0")
        self.frame_label.pack()
        # Create a toolbar with a slider for frame navigation
        toolbar = tk.Frame(self)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        frame_slider = ttk.Scale(
            toolbar,
            from_=1,
            to=self.total_frames,
            variable=self.frame_number,
            orient=tk.HORIZONTAL,
            command=self.update_frame_with_mouse_position
        )
        frame_slider.pack(side = tk.BOTTOM,fill=tk.X)
        #run_button = tk.Button(toolbar, text="Run Video", command=self.run_video)
        #run_button.pack(side=tk.BOTTOM)

        self.canvas.bind("<MouseWheel>", self.film_zoom)
        self.canvas.bind("<Control-MouseWheel>", self.zoom)
        self.canvas.bind("<Motion>", self.update_mouse_position)

        # Update the frame initially
        self.update_frame_with_mouse_position(self.frame_number.get())
    """
    def run_video(self):
        current_frame = self.frame_number.get()
        self.update_frame_with_mouse_position(current_frame)
        self.after(100,run_video)# Adjust the interval (in milliseconds) as needed
    """

    def update_mouse_position(self, event):
        # Get the current mouse position
        x, y = event.x, event.y

        # Update the text on the canvas with the mouse position
        self.canvas.itemconfig(self.mouse_position_text, text=f"Mouse Position: ({x}, {y})")


    def open_sub_window(self):
        sub_window = tk.Toplevel(self)

        label_style = {'background': 'lightgray', 'padx': 10, 'pady': 5, 'bd': 1, 'relief': 'solid'}

        self.calculated_green = tk.Label(sub_window, text=" Number of momomers tracked: 0", **label_style)
        self.calculated_green.pack()

        self.calculated_red = tk.Label(sub_window, text=" Number of dimers tracked: 0", **label_style)
        self.calculated_red.pack()

        self.calculated_ratio = tk.Label(sub_window, text=" Ratio of monomers : 0 %", **label_style)
        self.calculated_ratio.pack()

        self.label_entry = ttk.Label(sub_window, text="Enter a value for the treshold of the overlap:")
        self.label_entry.pack()


        # Create an entry widget
        self.entry = ttk.Entry(sub_window,textvariable=tk.StringVar(value="0"))
        self.entry.pack()




        print("treshhold =", self.threshold)

        self.overlap_count = tk.Label(sub_window, text=" Number of overlapping positions :  ", **label_style)
        self.overlap_count.pack()

        self.button1 = ttk.Button(sub_window, text="Get overlap information", command=self.draw_overlap)
        self.button1.pack()

        self.calculated_green.config(text=f" Number of momomers tracked: {self.core.compteur_green}")
        self.calculated_red.config(text=f" Number of dimers tracked: {self.core.compteur_red}")
        self.calculated_ratio.config(text=f" Ratio of monomers: {np.round(self.core.ratio, 4)} %")






    def draw_overlap(self):
        self.core.overlap = 0
        self.threshold = int(self.entry.get())
        for i in range(self.total_frames) :
            print("i =",i)
            self.core.calculate_distance(i,self.threshold)

        self.overlap_count.config(text=f" Number of overlaping positions : {self.core.overlap} ")

    def get_draw_track(self):

        track_number = self.track_number_entry.get()

        position_track_x,position_track_y = self.core.calculate_draw_tracks(track_number)

        # Example: Draw or update a box on the canvas
        # You should replace this with your actual drawing logic
        self.canvas.delete("track")  # Remove previous box
        self.canvas.delete("track_label")
        for i in range(np.size(position_track_x,0)) :
            x = position_track_x[i]
            y = position_track_y[i]


            zoomed_x = x * self.zoom_factor * self.zoom_factor_film - self.new_x
            zoomed_y = y * self.zoom_factor * self.zoom_factor_film - self.new_y

            radius = 3 * self.zoom_factor * self.zoom_factor_film

            x1 = zoomed_x - radius
            y1 = zoomed_y - radius
            x2 = zoomed_x + radius
            y2 = zoomed_y + radius

            label_mouvement = 8 * self.zoom_factor * self.zoom_factor_film

            if i == 1 :
                label_text = str(i)
                self.canvas.create_text(zoomed_x-label_mouvement, zoomed_y, text=label_text, fill="white", tags="track_label")

            self.canvas.create_oval(x1, y1, x2, y2,outline="blue", width=2, tags="track")

    def zoom_reset(self):
        self.zoom_factor = 1.0
        self.zoom_factor_film = 1.0
        self.percentage_y = 0
        self.percentage_x = 0
        self.new_x = 0
        self.new_y = 0
        self.update_frame_with_mouse_position(self.frame_number.get())

    def film_zoom(self, event):
        # Get the mouse position relative to the canvas
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Calculate the percentage of the film width and height the mouse is at
        self.percentage_x = x / self.canvas.winfo_width()*self.zoom_factor
        self.percentage_y = y / self.canvas.winfo_height()*self.zoom_factor

        # Adjust zoom factor for the film without changing player size
        if event.delta > 0:
            # Zoom in on the film
            self.zoom_factor_film *= 1.1
        else:
            # Zoom out on the film
            self.zoom_factor_film /= 1.1

        # Limit the zoom factor within a reasonable range
        self.zoom_factor_film = max(0.1, min(5.0, self.zoom_factor_film))

        # Update the frame with the new film zoom factor

        self.update_frame_with_mouse_position(frame_number=self.frame_number.get())

    def update_frame_with_mouse_position(self, frame_number):

        frame_number = int(float(frame_number))

        specific_frame = self.tiff_stack[frame_number]
        #self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(frame_number)))


        #ret, frame = self.cap.read()
        #for current_frame_number, frame in enumerate(self.vidstream):
        #    if current_frame_number == frame_number:
                # Process or display the frame as needed
        #        print(f"Retrieved frame index: {current_frame_number}")
        #        image_array = np.array(frame.to_image())
                # Perform further processing or display the image as needed
        #        break




        pil_image = Image.fromarray(specific_frame)

        # Apply general zoom factor for player size
        zoomed_width = int(self.frame_width * self.zoom_factor)
        zoomed_height = int(self.frame_height * self.zoom_factor)

        # Apply film-specific zoom factor
        film_zoomed_width = int(zoomed_width * self.zoom_factor_film)
        film_zoomed_height = int(zoomed_height * self.zoom_factor_film)

        # Calculate the position to keep the mouse at the same relative position
        self.new_x = self.percentage_x * (film_zoomed_width - zoomed_width)
        self.new_y = self.percentage_y * (film_zoomed_height - zoomed_height)

        # Resize the frame based on both factors
        pil_image = pil_image.resize((film_zoomed_width, film_zoomed_height))

        # Convert PIL Image to Tkinter PhotoImage
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # Update the canvas with the new image


        self.canvas.config(width=zoomed_width, height=zoomed_height)
        self.canvas.create_image(-self.new_x, -self.new_y, anchor=tk.NW, image=self.tk_image)

        self.mouse_position_text = self.canvas.create_text(10, 10, text="test", anchor=tk.NW, fill="white")
        self.canvas.itemconfig(self.mouse_position_text, text=f"Mouse Position: (O, 0)")


        # Update the frame label
        self.frame_label.config(text=f"Frame: {int(float(frame_number))}")



        # Get box position
        box_position,tracker_red = self.get_box_position(int(float(frame_number)))

        # Draw or update the box on the canvas
        self.draw_box(box_position,tracker_red)
        self.get_draw_track()

    '''
    def update_frame(self, frame_number):
        print(frame_number)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_number))
        ret, frame = self.cap.read()

        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            # Convert BGR frame to RGB for display
            zoomed_width = int(self.frame_width * self.zoom_factor)
            zoomed_height = int(self.frame_height * self.zoom_factor)

            # Apply film-specific zoom factor
            film_zoomed_width = int(zoomed_width * self.zoom_factor_film)
            film_zoomed_height = int(zoomed_height * self.zoom_factor_film)

            pil_image = pil_image.resize((film_zoomed_width, film_zoomed_height))

            # Convert PIL Image to Tkinter PhotoImage
            self.tk_image = ImageTk.PhotoImage(pil_image)

            # Update the canvas with the new image
            self.canvas.config(width=zoomed_width, height=zoomed_height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

            # Update the frame label
            self.frame_label.config(text=f"Frame: {int(float(frame_number))}")

            self.calculated_green.config(text=f" Number of momomers tracked: {self.core.compteur_green}")
            self.calculated_red.config(text=f" Number of dimers tracked: {self.core.compteur_red}")
            self.calculated_ratio.config(text=f" Ratio of monomers: {np.round(self.core.ratio,4)} %")

            # Get box position
            box_position = self.get_box_position(np.round(float(frame_number)))



            # Draw or update the box on the canvas
            self.draw_box(box_position)
    '''
    def get_box_position(self, frame_number):
        # Example: Calculate box position based on the frame number
        # You should replace this with your actual logic
        position,tracker_red = self.core.calculate_box_position(frame_number)
        return(position,tracker_red)

    def draw_box(self, positions,tracker_red):
        # Example: Draw or update a box on the canvas
        # You should replace this with your actual drawing logic
        self.canvas.delete("box")  # Remove previous box
        tracker = 0
        for position in positions :
            x,y = position
            zoomed_x = x * self.zoom_factor * self.zoom_factor_film - self.new_x
            zoomed_y = y * self.zoom_factor * self.zoom_factor_film - self.new_y

            radius = self.core.Rayon_boite * self.zoom_factor * self.zoom_factor_film

            x1 = zoomed_x - radius
            y1 = zoomed_y - radius
            x2 = zoomed_x + radius
            y2 = zoomed_y + radius


            if tracker_red[tracker] == 1 :
                self.canvas.create_oval(x1, y1, x2, y2,outline="red", width=2, tags="box")
            else :
                self.canvas.create_oval(x1, y1, x2, y2, outline="green", width=2, tags="box")

            tracker+=1

    def zoom(self, event):
        # Adjust zoom factor based on the mouse wheel direction
        if event.delta > 0:
            self.zoom_factor *= 1.1  # Zoom in
        else:
            self.zoom_factor /= 1.1  # Zoom out

        # Limit the zoom factor within a reasonable range
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))

        # Update the frame with the new zoom factor
        self.update_frame_with_mouse_position(self.frame_number.get())

    def run(self):
        self.mainloop()



class OpenFilesWindow(tk.Toplevel):
    def __init__(self, gui, master=None):
        super().__init__(master)
        self.title("Open Files ")
        self.gui = gui
        self.core = self.gui.core

        # Increase font size or other attributes
        self.my_font = ('Arial', 14)

        self.file_name_var = tk.StringVar()
        self.video_name_var = tk.StringVar()
        self.video_name_var.set("None")
        self.current_task_var = tk.StringVar()
        self.current_task_var.set("Aucune tâche en cours.")
        #self.grab_set()  # The grab_set method renders the window modal, meaning that the user will not be able to interact with other application windows while this window is open.top left corner

        self.setup_gui()

    def setup_gui(self):
        tk.Label(self, text="Track Data (from trackmate):", font=self.my_font).grid(row=0, column=0)
        tk.Entry(self, textvariable=self.file_name_var, state='readonly', font=self.my_font).grid(row=0, column=1)
        tk.Button(self, text="Browse", command=self.browse_file, font=self.my_font).grid(row=0, column=2)

        tk.Label(self, text="Video File (optional):", font=self.my_font).grid(row=1, column=0)
        tk.Entry(self, textvariable=self.video_name_var, state='readonly', font=self.my_font).grid(row=1, column=1)
        tk.Button(self, text="Browse", command=self.browse_video, font=self.my_font).grid(row=1, column=2)

        tk.Button(self, text="OK", command=self.start_calculation, font=self.my_font).grid(row=2, column=0)
        tk.Button(self, text="Cancel", command=self.destroy, font=self.my_font).grid(row=2, column=1)


        tk.Label(self, textvariable=self.current_task_var, font=self.my_font).grid(row=3, column=0)
        self.progress = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self.progress.grid(row=3, column=1, columnspan=2)

    def browse_file(self):
        file_name = filedialog.askopenfilename()
        if not file_name:
            self.file_name_var.set("None")
        else:
            self.file_name_var.set(file_name)

    def browse_video(self):
        video_name = filedialog.askopenfilename()
        if not video_name:
            self.video_name_var.set("None")
        else:
            self.video_name_var.set(video_name)

    def start_calculation(self):
        if self.file_name_var.get():
            # self.core.load_txt is a long-running task
            params = {}
            params["filepath_track"] = self.file_name_var.get()
            if self.video_name_var.get():
                params["filepath_video"] = self.video_name_var.get()

                #self.openVideoPlayer.run()

            else:
                params["filepath_video"] = None

            self.core.is_computing = True
            threading.Thread(target=self.core.load_data, args=((params,))).start()
            #self.core.load_data(params)
            # Update progress every 0.3 seconds
            self.after(300, self.update_progress, params)





    def update_progress(self,params):

        #print(" task_counter GUI =",self.core.task_counter)
        self.current_task_var.set(self.core.current_task)
        if self.core.nTracks is not None:
            #progress_value = 50
            progress_value = float(self.core.task_counter) / self.core.nTracks * 100
            self.progress["value"] = progress_value
            self.update_idletasks()


        # if progress_value >= 100:
        #     self.destroy()
        #     return

        if not self.core.is_computing:
            # Calculation Ended -> Display Result
            self.gui.fill_treeview_with_tracks()
            if params["filepath_video"] != 'None' :
                self.openvideoplayer = VideoPlayer(gui=self, filePath=params["filepath_video"], master=self.master)
                #self.destroy()

        self.after(300, self.update_progress , params)


class pyAnalyzeTrack():
    def __init__(self):
        self.root = tk.Tk()
        self.core = AnalyzeTrackCore()  # Model in model/View Pattern
        # self.camera = DummyCamera()
        self.iid_track_dict = {}

        # The string in the list correspond to the variable name of the Track class
        self.list_plotable_and_filter_data = ["nSpots", "r_gauss", "r_msd", "r_cov", "red_mean", "green_mean"]

        self.create_gui()
        self.create_menu()

        self.root.protocol("WM_DELETE_WINDOW", self.onQuit)

    def run(self):
        self.root.title("py Analyze Track")
        self.root.deiconify()
        self.root.mainloop()

    def export(self):
        file_path = filedialog.asksaveasfile(title="Export file name ?")
        if file_path == None or file_path == '':
            return None
        self.core.exportData(file_path.name)

    def create_Menu(self):
        pass

    def create_Treeview(self):
        # self.frame_treeview_track = tk.LabelFrame(self.root, text="Tracks", borderwidth=2)
        # self.frame_filter = tk.LabelFrame(self.root, text="Analyze", borderwidth=2)
        # self.frame_analyze = tk.LabelFrame(self.root, text="Analyze", borderwidth=2)
        #
        # self.frame_treeview_track.pack(side="top", fill="both", expand=True)
        # self.frame_filter.pack(side="top", fill="both", expand=True)
        # self.frame_analyze.pack(side="left", fill="both", expand=True)

        # https://riptutorial.com/tkinter/example/31885/customize-a-treeview
        self.tree_view = ttk.Treeview(self.frame_treeview_track)
        self.tree_view["columns"] = ("num", "nb Spot", "r gauss", "r cov", "r_msd", "red", "green", "t", "x", "y", "z")
        # remove first empty column with the identifier
        # self.tree_view['show'] = 'headings'
        # tree.column("#0", width=270, minwidth=270, stretch=tk.NO) tree.column("one", width=150, minwidth=150, stretch=tk.NO) tree.column("two", width=400, minwidth=200) tree.column("three", width=80, minwidth=50, stretch=tk.NO)

        columns_text = ["num", "nb Spot", "r gauss", "r_msd", "r cov", "red", "green", "t", "x", "y", "z"]

        self.tree_view.column("#0", width=25, stretch=tk.NO)
        self.tree_view.column(columns_text[0], width=75, stretch=tk.YES, anchor=tk.CENTER)  # num
        self.tree_view.column(columns_text[1], width=50, stretch=tk.YES, anchor=tk.CENTER)  # nb Spot
        self.tree_view.column(columns_text[2], width=125, stretch=tk.YES, anchor=tk.CENTER)  # r gauss
        self.tree_view.column(columns_text[3], width=125, stretch=tk.YES, anchor=tk.CENTER)  # r msd
        self.tree_view.column(columns_text[4], width=125, stretch=tk.YES, anchor=tk.CENTER)  # r cov
        self.tree_view.column(columns_text[5], width=75, stretch=tk.YES, anchor=tk.CENTER)  # red
        self.tree_view.column(columns_text[6], width=75, stretch=tk.YES, anchor=tk.CENTER)  # green
        self.tree_view.column(columns_text[7], width=50, stretch=tk.YES, anchor=tk.CENTER)  # t
        self.tree_view.column(columns_text[8], width=75, stretch=tk.YES, anchor=tk.CENTER)  # x
        self.tree_view.column(columns_text[9], width=75, stretch=tk.YES, anchor=tk.CENTER)  # y
        self.tree_view.column(columns_text[10], width=75, stretch=tk.YES, anchor=tk.CENTER)  # z

        # self.tree_view.heading("name", text="name")
        for col in columns_text:
            self.tree_view.heading(col, text=col,
                                   command=lambda _col=col: self.treeview_sort_column(self.tree_view, _col, False))

        # # FIXME only change text color to light gray
        self.tree_view.tag_configure('filtered', foreground='gray50')
        self.tree_view.tag_configure('filtered', background='gray20')

        self.tree_view.tag_configure('highlighted', background='gray90')
        self.tree_view.tag_configure('highlighted', foreground='gold3')

        ysb = ttk.Scrollbar(self.frame_treeview_track, orient='vertical', command=self.tree_view.yview)
        self.tree_view.grid(row=0, column=0, sticky='nsew')
        ysb.grid(row=0, column=1, sticky='ns')
        self.tree_view.configure(yscroll=ysb.set)

        self.tree_view.bind('<<TreeviewSelect>>', self.treeview_track_select)
        self.tree_view.bind("<Double-1>", self.on_double_click_treeview)

        # self.notebook = ttk.Notebook(self.frame_analyze)
        # self.notebook.pack(expand=True, fill="both")

        s = ttk.Style()
        s.theme_create("MyStyle", parent="alt", settings={
            "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0]}},
            "TNotebook.Tab": {"configure": {"padding": [100, 10],
                                            "font": ('URW Gothic L', '11', 'bold')}, }})
        s.theme_use("MyStyle")

        self.check_show_filtered_iv = tk.IntVar(value=0)

    # FIXME le mettre quelque part de pertinent.
    # tk.Checkbutton(self.frame_treeview_track, text='show Filtered as gray', variable=self.check_show_filtered_iv, command=self.update_ui(), onvalue=1, offvalue=0).grid(row=0, column=0, columnspan=2)

    def create_params(self):
        # Frame Param Exp
        #################
        ttk.Label(self.frame_param_exp, text='T (°C) :').grid(row=0, column=0, padx=8)
        self.T_sv = tk.StringVar(value='20')
        ttk.Entry(self.frame_param_exp, textvariable=self.T_sv, justify=tk.CENTER, width=7).grid(row=0, column=1,padx=8)

        # TODO combo avec solvant
        ttk.Label(self.frame_param_exp, text='eta (Pa.s) :').grid(row=1, column=0, padx=8)
        self.eta_sv = tk.StringVar(value='0.001')
        ttk.Entry(self.frame_param_exp, textvariable=self.eta_sv, justify=tk.CENTER, width=7).grid(row=1, column=1,padx=8)
        self.solvant_combo_sv = tk.StringVar()
        cb = ttk.Combobox(self.frame_param_exp, width=13, justify=tk.CENTER, textvariable=self.solvant_combo_sv,values='')
        # cb.bind('<<ComboboxSelected>>', self.change_algo)
        cb['values'] = ('water', 'DMSO', 'ethanol')
        self.solvant_combo_sv.set('water')
        cb.grid(row=1, column=2, padx=8)

        ttk.Label(self.frame_param_exp, text='Δt (ms) :').grid(row=2, column=0, padx=8)
        self.delta_t_sv = tk.StringVar(value='33')
        ttk.Entry(self.frame_param_exp, textvariable=self.delta_t_sv, justify=tk.CENTER, width=7).grid(row=2, column=1,padx=8)

        ttk.Label(self.frame_param_exp, text='Δpix (nm) :').grid(row=3, column=0, padx=8)
        self.Delta_pix_sv = tk.StringVar(value='263')
        ttk.Entry(self.frame_param_exp, textvariable=self.Delta_pix_sv, justify=tk.CENTER, width=7).grid(row=3,column=1,padx=8)

        ttk.Label(self.frame_param_exp, text='Nombre de points minimum acceptable dans une trajectoire').grid(row=4,column=0,padx=8)
        self.nb_spot_min_sv = tk.StringVar(value='10')
        e = ttk.Entry(self.frame_param_exp, textvariable=self.nb_spot_min_sv, justify=tk.CENTER, width=7).grid(row=4, column=1,padx=8)

        tk.Button(self.frame_param_exp, text="Export radius covariance", command=self.export).grid(row=5, column=0)

        # Frame Param Algo
        ##################
        ttk.Label(self.frame_param_exp, text='Drift compensation :').grid(row=0, column=0, padx=8)
        self.drift_algo_cv = tk.StringVar()
        cb = ttk.Combobox(self.frame_param_algo, width=13, justify=tk.CENTER, textvariable=self.drift_algo_cv,values='')
        # cb.bind('<<ComboboxSelected>>', self.change_algo)
        cb['values'] = ('None', "neighbors", 'self')
        self.drift_algo_cv.set('neighbors')
        cb.grid(row=0, column=1, padx=8)

    def create_filters(self):
        pad_filter = 5

        self.is_not_f1 = True
        self.button_not_f1 = tk.Button(self.frame_filter, text="not", width=3, command=self.toggle_not_f1)
        self.button_not_f1.grid(row=0, column=0, padx=pad_filter)

        # So that not start as false
        self.toggle_not_f1()

        self.filter_1_low_sv = tk.StringVar()
        ttk.Entry(self.frame_filter, textvariable=self.filter_1_low_sv, justify=tk.CENTER, width=12).grid(row=0,column=1,padx=pad_filter)
        ttk.Label(self.frame_filter, text=' < ').grid(row=0, column=2, padx=pad_filter)

        self.cb_value_filter_1_sv = tk.StringVar()
        self.cb_value_filter_1 = ttk.Combobox(self.frame_filter, width=25, justify=tk.CENTER,textvariable=self.cb_value_filter_1_sv, values='')
        self.cb_value_filter_1['values'] = self.list_plotable_and_filter_data
        self.cb_value_filter_1.set('None')
        self.cb_value_filter_1.bind('<<ComboboxSelected>>', self.change_filter1_type)
        self.cb_value_filter_1.grid(row=0, column=3, padx=pad_filter)

        ttk.Label(self.frame_filter, text=' < ').grid(row=0, column=4, padx=pad_filter)
        self.filter_1_high_sv = tk.StringVar()
        ttk.Entry(self.frame_filter, textvariable=self.filter_1_high_sv, justify=tk.CENTER, width=12).grid(row=0,column=5,padx=pad_filter)

        self.cb_value_filter_bool_op_sv = tk.StringVar()
        self.cb_value_filter_bool_op = ttk.Combobox(self.frame_filter, width=25, justify=tk.CENTER,textvariable=self.cb_value_filter_bool_op_sv, values='')
        self.cb_value_filter_bool_op['values'] = ["and", "or", "xor"]
        self.cb_value_filter_bool_op.set('or')

        self.cb_value_filter_bool_op.grid(row=1, column=0, columnspan=6, padx=pad_filter, pady=10)

        self.is_not_f2 = True
        self.button_not_f2 = tk.Button(self.frame_filter, text="not", width=3, command=self.toggle_not_f2)
        self.button_not_f2.grid(row=2, column=0, padx=pad_filter)
        # So that not start as false
        self.toggle_not_f2()

        self.filter_2_low_sv = tk.StringVar()
        ttk.Entry(self.frame_filter, textvariable=self.filter_2_low_sv, justify=tk.CENTER, width=12).grid(row=2,column=1,padx=pad_filter)
        ttk.Label(self.frame_filter, text=' < ').grid(row=2, column=2, padx=pad_filter)

        self.cb_value_filter_2_sv = tk.StringVar()
        self.cb_value_filter_2 = ttk.Combobox(self.frame_filter, width=25, justify=tk.CENTER,textvariable=self.cb_value_filter_2_sv, values='')
        self.cb_value_filter_2['values'] = self.cb_value_filter_1['values'] = self.list_plotable_and_filter_data
        self.cb_value_filter_2.set('None')
        self.cb_value_filter_2.bind('<<ComboboxSelected>>', self.change_filter2_type)
        self.cb_value_filter_2.grid(row=2, column=3, padx=pad_filter)

        ttk.Label(self.frame_filter, text=' < ').grid(row=2, column=4, padx=pad_filter)

        self.filter_2_high_sv = tk.StringVar()
        ttk.Entry(self.frame_filter, textvariable=self.filter_2_high_sv, justify=tk.CENTER, width=12).grid(row=2,column=5,padx=pad_filter)

        ttk.Button(self.frame_filter, text="Filter", width=3, command=self.launch_filter).grid(row=3, column=0,columnspan=6,padx=pad_filter)
        ttk.Button(self.frame_filter, text="CLEAR Filter", command=self.clear_filter).grid(row=4, column=0,columnspan=6,padx=pad_filter)

    def launch_filter(self):
        def convert_sv_float(sv):
            str_ = sv.get()
            try:
                f = float(str_)
                return f
            except ValueError:
                return None

        low1 = convert_sv_float(self.filter_1_low_sv)
        high1 = convert_sv_float(self.filter_1_high_sv)
        type1 = self.cb_value_filter_1_sv.get()

        bool_op = self.cb_value_filter_bool_op_sv.get()

        type2 = self.cb_value_filter_2_sv.get()
        low2 = convert_sv_float(self.filter_2_low_sv)
        high2 = convert_sv_float(self.filter_2_high_sv)

        self.core.filter_tracks(low1, high1, type1, self.is_not_f1, bool_op, low2, high2, type2, self.is_not_f2)

        self.update_ui()

    def clear_filter(self):
        pass

    def update_ui(self):
        self.fill_treeview_with_tracks

    # TODO Graph updates

    def change_filter2_type(self, event=None):
        type_ = self.cb_value_filter_2_sv.get()

    def change_filter1_type(self, event=None):
        type_ = self.cb_value_filter_1_sv.get()

    def toggle_not_f1(self):
        if self.is_not_f1:
            self.button_not_f1.config(font=('courier', 12, 'normal'), foreground='black', fg='gray50')
            self.is_not_f1 = False
        else:
            self.button_not_f1.config(font=('courier', 12, 'bold'), foreground='black', fg='gray0')
            self.is_not_f1 = True

    def toggle_not_f2(self):
        if self.is_not_f2:
            self.button_not_f2.config(font=('courier', 12, 'normal'), foreground='black', fg='gray50')
            self.is_not_f2 = False
        else:
            self.button_not_f2.config(font=('courier', 12, 'bold'), foreground='black', fg='gray0')
            self.is_not_f2 = True

    def create_all_tracks_analyze(self):
        self.fig_result_all_tracks = plt.Figure(figsize=(4, 4), dpi=100)
        self.frame_graph_all_tracks_plot = tk.Frame(self.frame_graph_all_tracks)
        self.frame_graph_all_tracks_plot.pack(side='left', fill='both', expand=1)
        self.frame_graph_all_tracks_params = tk.Frame(self.frame_graph_all_tracks)
        self.frame_graph_all_tracks_params.pack(side='left', fill='both', expand=1)

        self.ax_result_all_tracks = self.fig_result_all_tracks.add_subplot(111)

        plt.subplots_adjust(hspace=0)
        self.fig_result_all_tracks.set_tight_layout(True)
        self.canvas_result_all_tracks = FigureCanvasTkAgg(self.fig_result_all_tracks,
                                                          master=self.frame_graph_all_tracks)
        self.canvas_result_all_tracks.draw()
        # self.canvas_result_all_track.get_tk_widget().pack(side='top', fill='both', expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas_result_all_tracks, self.frame_graph_all_tracks)
        self.canvas_result_all_tracks._tkcanvas.pack(side='top', fill='both', expand=1)

        self.plot_all_mode_sv = tk.StringVar()
        cb = ttk.Combobox(self.frame_graph_all_tracks_params, width=13, justify=tk.CENTER,textvariable=self.plot_all_mode_sv,values='')

        # cb.bind('<<ComboboxSelected>>', self.change_algo)
        cb['values'] = ('Sum gaussian', 'Lags', 'Radius Histogram', 'Radius Red Histogram', 'Radius Green Histogram')
        self.plot_all_mode_sv.set('Sum gaussian')
        cb.grid(row=0, column=0, padx=8)

        tk.Label(self.frame_graph_all_tracks_params, text='Rayon (Covariance) =').grid(row=3, column=0)
        self.max_index_sv = tk.StringVar()  # Valeur en x max de la gaussienne
        self.maximum = tk.Label(self.frame_graph_all_tracks_params, textvariable=self.max_index_sv).grid(row=3, column=1)

        tk.Label(self.frame_graph_all_tracks_params, text="Nombre de spots moyen =").grid(row=4, column=0)
        self.moyenne_spots = tk.StringVar()  # Valeur en x max de la gaussienne
        self.M_spots = tk.Label(self.frame_graph_all_tracks_params, textvariable=self.moyenne_spots).grid(row=4,column=1)



    def create_current_track_analyze(self):
        self.fig_result_current_track = plt.Figure(figsize=(4, 4), dpi=100)
        self.frame_graph_current_track_plot = tk.Frame(self.frame_graph_current_track)
        self.frame_graph_current_track_plot.pack(side='left', fill='both', expand=1)
        self.frame_graph_current_track_params = tk.Frame(self.frame_graph_current_track)
        self.frame_graph_current_track_params.pack(side='left', fill='both', expand=1)
        self.ax_result_current_track = self.fig_result_current_track.add_subplot(111)

        plt.subplots_adjust(hspace=0)
        self.fig_result_current_track.set_tight_layout(True)
        self.canvas_result_current_track = FigureCanvasTkAgg(self.fig_result_current_track, master=self.frame_graph_current_track_plot)
        # self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        self.toolbar_current_track = NavigationToolbar2Tk(self.canvas_result_current_track, self.frame_graph_current_track_plot)
        self.canvas_result_current_track._tkcanvas.pack(side='left', fill='both', expand=1)

        self.plot_current_mode_sv = tk.StringVar()
        cb = ttk.Combobox(self.frame_graph_current_track_params, width=13, justify=tk.CENTER, textvariable = self.plot_current_mode_sv,values = '')
        # cb.bind('<<ComboboxSelected>>', self.change_algo)
        cb['values'] = ('track', 'Histogram displacement', 'test free motion via cov "', 'Lags', 'DCT')
        self.plot_current_mode_sv.set('track')
        cb.grid(row=0, column=0, padx=8)

    def create_gui(self):
        self.frame_treeview_track = tk.LabelFrame(self.root, text="Tracks", borderwidth=2)
        # self.frame_filter = tk.LabelFrame(self.root, text="Analyze", borderwidth=2)
        # self.frame_analyze = tk.LabelFrame(self.root, text="Analyze", borderwidth=2)

        self.frame_treeview_track.pack(side="top", fill="both", expand=True, anchor='center')
        # self.frame_filter.pack(side="top", fill="both", expand=True)
        # self.frame_analyze.pack(side="left", fill="both", expand=True)

        self.frame_param_filter = tk.Frame(self.root)
        self.frame_param_filter.pack(side="left", fill="both", expand=True)
        self.frame_param_exp = tk.LabelFrame(self.frame_param_filter, text="Params Exp", borderwidth=2)
        self.frame_param_exp.pack(side="top", fill="both", expand=True)
        self.frame_param_algo = tk.LabelFrame(self.frame_param_filter, text="Params Algo", borderwidth=2)
        self.frame_param_algo.pack(side="top", fill="both", expand=True)
        self.frame_filter = tk.LabelFrame(self.frame_param_filter, text="Filters", borderwidth=2)
        self.frame_filter.pack(side="top", fill="both", expand=True)

        self.frame_graph = tk.Frame(self.root)
        self.frame_graph.pack(side="left", fill="both", expand=True)

        self.frame_graph_full_sample = tk.LabelFrame(self.frame_graph, text="All tracks", borderwidth=2)
        self.frame_graph_full_sample.pack(side="top", fill="both", expand=True)
        self.frame_graph_all_tracks = tk.Frame(self.frame_graph_full_sample)
        self.frame_graph_all_tracks.pack(side="left", fill="both", expand=True)

        self.frame_graph_current_track = tk.LabelFrame(self.frame_graph, text="Current track", borderwidth=2)
        self.frame_graph_current_track.pack(side="left", fill="both", expand=True)

        self.create_params()
        self.create_filters()
        self.create_Treeview()
        self.create_all_tracks_analyze()
        self.create_current_track_analyze()

    def create_menu(self):
        self.menu_system = tk.Menu(self.root)

        # FILE#############
        self.menu_file = tk.Menu(self.menu_system, tearoff=0)
        self.menu_file.add_command(label='open', underline=1, accelerator="Ctrl+o", command=self.open_xml_trackmate)
        self.menu_file.add_command(label='generate', underline=1, accelerator="Ctrl+g", command=self.create_menu_generate)#self.generate_data)
        self.menu_file.add_command(label='Save State', underline=1, accelerator="Ctrl+s", command=self.save_state)
        # self.master.bind_all("<Control-s>", self.saveState)
        self.menu_file.add_command(label='Load State', underline=1, accelerator="Ctrl+l", command=self.load_state)
        self.menu_file.add_command(label='Save Gaussian Data', underline=1, accelerator="Ctrl+t", command=self.save_data_gauss)
        self.menu_file.add_command(label='Save Histogram Data', underline=1, accelerator="Ctrl+t",command=self.save_data_hist)
        self.menu_file.add_command(label='Save Red Histogram Data', underline=1, accelerator="Ctrl+t",command=self.save_red_data_hist)
        self.menu_file.add_command(label='Save Green Histogram Data', underline=1, accelerator="Ctrl+t",command=self.save_green_data_hist)
        self.menu_system.add_cascade(label="File", menu=self.menu_file)

        self.root.config(menu=self.menu_system)

        # self.master.bind_all("<Control-o>", self.askOpenSPC_file)
    def save_data_gauss(self):
        filePath = filedialog.asksaveasfile(title="Save State", defaultextension=".txt",filetypes=[("Text files", "*.txt")])
        if filePath == None or filePath.name == '':
            return None
        self.core.save_data_gauss(filePath)

    def save_data_hist(self):
        filePath = filedialog.asksaveasfile(title="Save State", defaultextension=".txt",filetypes=[("Text files", "*.txt")])
        if filePath == None or filePath.name == '':
            return None
        self.core.save_data_hist(filePath)

    def save_red_data_hist(self):
        filePath = filedialog.asksaveasfile(title="Save State", defaultextension=".txt",filetypes=[("Text files", "*.txt")])
        if filePath == None or filePath.name == '':
            return None
        self.core.save_red_data_hist(filePath)

    def save_green_data_hist(self):
        filePath = filedialog.asksaveasfile(title="Save State", defaultextension=".txt",filetypes=[("Text files", "*.txt")])
        if filePath == None or filePath.name == '':
            return None
        self.core.save_green_data_hist(filePath)


    def create_menu_generate(self):

        entry_window = tk.Toplevel(self.root)
        entry_window.title("Parameters of the particles")

        # Create entry widgets in the new window


        entry_label_1 = tk.Label(entry_window, text="Diameter of first particles (nm) :")
        entry_label_1.pack(pady=5)

        particle_mean_diam_nm = tk.StringVar(value = "60")
        entry_widget_1 = tk.Entry(entry_window, textvariable=particle_mean_diam_nm)
        entry_widget_1.pack(pady=5)

        entry_label_2 = tk.Label(entry_window, text="Diameter of second particles (nm) :")
        entry_label_2.pack(pady=5)

        particle_mean_diam_nm_2 = tk.StringVar(value= "120")
        entry_widget_2 = tk.Entry(entry_window, textvariable=particle_mean_diam_nm_2)
        entry_widget_2.pack(pady=5)

        entry_label_3 = tk.Label(entry_window, text="Relative sigma of the distribution of the particles size : ")
        entry_label_3.pack(pady=5)

        particle_diam_sigma_relative = tk.StringVar(value = "0.0")
        entry_widget_3 = tk.Entry(entry_window, textvariable=particle_diam_sigma_relative)
        entry_widget_3.pack(pady=5)

        entry_label_4 = tk.Label(entry_window, text="Number of particles :")
        entry_label_4.pack(pady=5)

        nb_particle = tk.StringVar(value = "20000")
        entry_widget_4 = tk.Entry(entry_window, textvariable=nb_particle)
        entry_widget_4.pack(pady=5)

        entry_label_5 = tk.Label(entry_window, text="Number of frame per track :")
        entry_label_5.pack(pady=5)

        nb_of_frame = tk.StringVar(value = "200")
        entry_widget_5 = tk.Entry(entry_window, textvariable=nb_of_frame)
        entry_widget_5.pack(pady=5)

        entry_label_6 = tk.Label(entry_window, text="Ratio of monomères (%) :")
        entry_label_6.pack(pady=5)

        ratio_monomere = tk.StringVar(value="50")
        entry_widget_6 = tk.Entry(entry_window, textvariable=ratio_monomere)
        entry_widget_6.pack(pady=5)

        action_button = tk.Button(entry_window, text="Generate",command=lambda : self.get_generate_data(particle_mean_diam_nm, particle_mean_diam_nm_2,particle_diam_sigma_relative, nb_particle,nb_of_frame,ratio_monomere))
        action_button.pack(pady=10)



    def get_generate_data(self,var1,var2,var3,var4,var5,var6):

        params_dict = {
            "Diam1": float(var1.get()),
            "Diam2": float(var2.get()),
            "sigma": float(var3.get()),
            "nb_particle": int(var4.get()),
            "nb_frame": int(var5.get()),
            "ratio_monomere": int(var6.get())
        }
        self.generate_data(params_dict)




    def generate_data(self,params_dict):
        # TODO creer une interface graphique pour collecter les parameters

        self.core.generate_brownian_track(params_dict)
        self.insert_tracks_tree_view()
        self.plot_result_all_track()
        pass

    def open_xml_trackmate(self):
        """Obsolete open trackmate data in xlm format. We use here txt data (csv style).:return:"""
        self.openFilesWIndow = OpenFilesWindow(gui=self, master=self.root)
        return

    def fill_treeview_with_tracks(self):
        self.clear_treeview()
        self.insert_tracks_tree_view()
        #self.treeview_track_select()

    def clear_treeview(self):
        self.tree_view.delete(*self.tree_view.get_children())

    def insert_tracks_tree_view(self):

        def round_float(value):
            return f"{value:.2f}"



        for num_track in range(self.core.nTracks):
            track = self.core.tracks[num_track]
            tags_ = []
            if track.is_filtered:
                tags_.append("filtered")
                if not self.check_show_filtered_iv.get():
                    continue
            # if track.is_highlighted:
            # 	tags_.append("highlighted")

            iid_track = self.tree_view.insert(parent="",index='end',values=(str(num_track),str(track.nSpots),round_float(track.r_gauss * 1E9) + "+/-" + round_float(track.error_r_gauss * 1E9),round_float(track.r_cov * 1E9) + "+/-" + round_float(track.error_r_cov * 1E9),
            round_float(track.r_msd * 10 ** 9) + "+/-" + round_float(track.error_r_msd * 10 ** 9),
            round_float(track.red_mean),
            round_float(track.green_mean),
            "",  # x
            "",  # y
            "",  # z
            ""  # t
            ),
            tags = tags_
            )




            # insert the position of the spots -> Time consuming and not very useful ?

            # for i in range(track.nSpots):
            # 	# TODO format string with less digit
            # 	x, y, z, t = track.x[i], track.y[i], track.z[i], track.t[i]
            #
            # 	iid = self.tree_view.insert(parent=iid_track, index='end',
            # 								values=(
            # 									"", "", "", "", str(t), str(x),
            # 									str(y),
            # 									str(z)))
            # 	self.tree_view.item(iid_track, open=False)
        #self.treeview_track_select()

    def treeview_track_select(self, event):
        track, num_track = self.get_selected_track_from_treeview()
        self.plot_current_track(num_track)
        self.plot_result_all_track()

    def plot_current_track(self, num_track):
        self.plot_current_track_mode = self.plot_current_mode_sv.get()
        if self.plot_current_track_mode == "track":
            self.plot_track_trajectory(num_track)
        elif self.plot_current_track_mode == "Histogram displacement":
            self.plot_gaussian_fit(num_track)
        elif self.plot_current_track_mode == "test free motion via cov":
            self.plot_test_free_MSD(num_track)
        elif self.plot_current_track_mode == 'Lags':
            self.plot_Lags(num_track)
        elif self.plot_current_track_mode == 'DCT':
            self.plot_DCT(num_track)

    def plot_result_all_track(self):
        self.max_index_sv.set(str(self.core.Max_index)+"(m)")
        self.moyenne_spots.set(str(int(self.core.moyenne_spots)))


        self.plot_all_track_mode = self.plot_all_mode_sv.get()
        if self.plot_all_track_mode == "Sum gaussian":
            self.ax_result_all_tracks.clear()
            self.fig_result_all_tracks.set_tight_layout(True)
            x, y = self.core.x_full_gauss, self.core.Gauss_full_track
            self.ax_result_all_tracks.plot(x, y)
            self.ax_result_all_tracks.set_xlabel("Radius (nm)")
            self.ax_result_all_tracks.set_ylabel("Somme des gaussiennes normalisés")
            self.fig_result_all_tracks.canvas.draw()
        elif self.plot_all_track_mode == "Lags":
            self.ax_result_all_tracks.clear()
            self.fig_result_all_tracks.set_tight_layout(True)
            x = self.core.Controle
            y = self.core.Controle * 0
            z = self.core.Controle_variance
            self.ax_result_all_tracks.plot(x)
            self.ax_result_all_tracks.errorbar(range(len(x)), x, yerr=z)
            self.ax_result_all_tracks.set_xlabel("Time Lags j")
            self.ax_result_all_tracks.set_ylabel("<$\Delta x_n * \Delta x_{n+m}$>")
            self.fig_result_all_tracks.canvas.draw()
        elif self.plot_all_track_mode == "Radius Histogram":
            self.ax_result_all_tracks.clear()
            self.fig_result_all_tracks.set_tight_layout(True)
            # Histoy,Histox = np.histogram(self.core.Moyenner,bins=100)
            filtered_data = np.array(self.core.Moyenner)[ np.array(self.core.Moyenner) <= self.core.lim_max * 10 ** 9 ]
            Histoy, Histox = np.histogram(filtered_data, bins = int(self.core.lim_max * 10 ** 9 / 5))
            print("Bin =", Histox)
            print("Values =", Histoy)
            #plt.bar(Histox[:-1], Histoy, width=np.diff(Histox), edgecolor='black')
            #plt.hist(self.core.Moyenner, bins='auto', edgecolor='black')
            self.ax_result_all_tracks.bar(Histox[:-1], Histoy,width=np.diff(Histox),edgecolor = 'black')
            #Bin = Histox
            #plt.bar(Histox[:-1], Histoy, color='blue')
            self.ax_result_all_tracks.set_xlim([self.core.lim_min *10 ** 9, self.core.lim_max * 10 ** 9])
            self.ax_result_all_tracks.set_xlabel("Radius (nm)")
            self.ax_result_all_tracks.set_ylabel("Occurence")
            self.fig_result_all_tracks.canvas.draw()
        elif self.plot_all_track_mode == "Radius Green Histogram":
            self.ax_result_all_tracks.clear()
            self.fig_result_all_tracks.set_tight_layout(True)
            filtered_data = np.array(self.core.Moyenner)[self.core.number_tracks_green]
            filtered_data = np.array(filtered_data)[np.array(filtered_data) <= self.core.lim_max * 10 ** 9]
            Histoy, Histox = np.histogram(filtered_data, bins=int(self.core.lim_max * 10 ** 9 / 5))
            print("Bin =", Histox)
            print("Values =", Histoy)
            self.ax_result_all_tracks.bar(Histox[:-1], Histoy, width=np.diff(Histox), edgecolor='black')
            self.ax_result_all_tracks.set_xlim([self.core.lim_min * 10 ** 9, self.core.lim_max * 10 ** 9])
            self.ax_result_all_tracks.set_xlabel("Radius (nm) of green track")
            self.ax_result_all_tracks.set_ylabel("Occurence")
            self.fig_result_all_tracks.canvas.draw()
        elif self.plot_all_track_mode == "Radius Red Histogram":
            self.ax_result_all_tracks.clear()
            self.fig_result_all_tracks.set_tight_layout(True)
            filtered_data = np.array(self.core.Moyenner)[self.core.number_tracks_red]
            filtered_data = np.array(filtered_data)[np.array(filtered_data) <= self.core.lim_max * 10 ** 9]
            Histoy, Histox = np.histogram(filtered_data, bins=int(self.core.lim_max * 10 ** 9 / 5))
            print("Bin =", Histox)
            print("Values =", Histoy)
            self.ax_result_all_tracks.bar(Histox[:-1], Histoy, width=np.diff(Histox), edgecolor='black')
            self.ax_result_all_tracks.set_xlim([self.core.lim_min * 10 ** 9, self.core.lim_max * 10 ** 9])
            self.ax_result_all_tracks.set_xlabel("Radius (nm) of red track")
            self.ax_result_all_tracks.set_ylabel("Occurence")
            self.fig_result_all_tracks.canvas.draw()
    def get_selected_track_from_treeview(self):
        id_selected_item = self.tree_view.focus()
        parent_iid = self.tree_view.parent(id_selected_item)
        selected_item = self.tree_view.item(id_selected_item)
        parent_item = self.tree_view.item(parent_iid)

        if parent_iid == "":
            # The user has clicked on the header of the track
            num_track = selected_item["values"][0]
        else:
            num_track = parent_item["values"][0]

        return self.core.tracks[num_track], num_track

    def on_double_click_treeview(self, event):
        pass

        # region = self.tree_view.identify("region", event.x, event.y)
        # if region == "heading":
        # 	# Returns the data column identifier of the cell at position x. The tree column has ID #0.
        # 	column_id = self.tree_view.identify_column(event.x)
        # 	print(column_id)

    def treeview_sort_column(self, tv, col, reverse):
        pass

         # l = [(tv.set(k, col), k) for k in tv.get_children('')]
        # l.sort(reverse=reverse)
        #
        # # rearrange items in sorted positions
        # for index, (val, k) in enumerate(l):
        # 	tv.move(k, '', index)
        #
        # # reverse sort next time
        # tv.heading(col, command=lambda _col=col: self.treeview_sort_column(tv, _col, not reverse))
        #
        # if col == "":
        # 	pass
        # pass

        # TODO sort column https://stackoverflow.com/questions/1966929/tk-treeview-column-sort

    def plot_track_trajectory(self, num_track):
        x = self.core.get_x_data(num_track)
        y = self.core.get_y_data(num_track)

        self.ax_result_current_track.clear()
        self.fig_result_current_track.set_tight_layout(True)
        self.ax_result_current_track.plot(x, y)
        self.ax_result_current_track.plot(x[0], y[0], "ro")
        self.ax_result_current_track.plot(x[-1], y[-1], "bo")
        self.ax_result_current_track.set_xlabel("x")
        self.ax_result_current_track.set_ylabel("y")
        self.ax_result_current_track.set_title("Trajectory")
        self.fig_result_current_track.canvas.draw()

    def plot_test_free_MSD(self, num_track):
        self.core.tracks[num_track].test_free_diffusion_via_MSD()
        msd_x = self.core.tracks[num_track].test_msd_x
        msd_y = self.core.tracks[num_track].test_msd_y

        # FIXME 16 -> nb magique
        x_axis = np.arange(1, 16)

        self.ax_free_MSD.clear()

        self.ax_free_MSD.plot(x_axis, msd_x, "o", label='x')
        self.ax_free_MSD.plot(x_axis, msd_y, "o", label='y')
        self.ax_free_MSD.set_xlabel("lag j")
        self.ax_free_MSD.set_ylabel(r"$\Delta x_n \Delta x_{n + j}$")
        self.ax_free_MSD.legend()
        self.ax_free_MSD.set_title("Test free diffusion via MSD")
        self.figure_test_free_MSD.set_tight_layout(True)
        self.figure_test_free_MSD.canvas.draw()

    def plot_Lags(self, num_track):
        y = self.core.Controle_track[:, num_track]
        x = self.core.Controle_track[:, num_track]*0
        z = self.core.Controle_track_variance[:, num_track]
        self.ax_result_current_track.clear()
        self.fig_result_current_track.set_tight_layout(True)
        self.ax_result_current_track.plot(y)
        self.ax_result_current_track.errorbar(range(len(y)), y, yrr=z)
        self.ax_result_current_track.set_xlabel("Lags j")
        self.ax_result_current_track.set_ylabel(r"$\Delta x_n \Delta x_{n + j}$")
        self.fig_result_current_track.canvas.draw()

    def plot_DCT(self, num_track):
        y = self.core.Spectre_mean[num_track]
        z = np.arange(0.5 * self.core.taille[num_track], len(y) * self.core.taille[num_track], self.core.taille[num_track])
        self.ax_result_current_track.clear()
        self.fig_result_current_track.set_tight_layout(True)
        self.ax_result_current_track.setter(z, y)
        self.ax_result_current_track.set_xlabel("Mode k")
        self.ax_result_current_track.set_ylabel("$P_k / [Delta t^2$]")
        self.ax_result_current_track.set_xlim([0, self.core.taille[num_track] * self.core.division])
        self.ax_result_current_track.set_ylim([0, 4])
        self.fig_result_current.canvas.draw()

    def plot_gaussian_fit( self, num_track):
        def gaussian(x, result_fit):

            """
            Intermediate helpinf function for calculating more point for the x_axis of gaussian fit
            """
            amp, cen, wid = result_fit.params["amp"], result_fit.params["cen"], result_fit.params["wid"]
            return amp * np.exp(-(x - cen) ** 2 / wid)


        diff_x, hist_x, boundaries_x, result_x, center_x, width_x = self.core.gaussian_delta_fit(self.core.tracks[num_track].x)
        diff_y, hist_y, boundaries_y, result_y, center_y, width_y = self.core.gaussian_delta_fit(self.core.tracks[num_track].y)

        self.ax_result_current_track.clear()
        self.fig_result_current_track.set_tight_layout(True)

        self.ax_result_current_track.bar(boundaries_x, hist_x, alpha=0.5)
        x= np.linspace(np.min(boundaries_x), np.max(boundaries_x),150)
        fit_x = gaussian(x, result_x)
        self.ax_result_current_track.plot(x, fit_x)

        self.ax_result_current_track.bar(boundaries_y, hist_y, alpha=0.5)
        y = np.linspace(np.min(boundaries_y), np.max(boundaries_y), 150)
        fit_y = gaussian(y, result_y)

        self.ax_result_current_track.plot(y, fit_y)

        self.fig_result_current_track.canvas.draw()


    def save_state(self):
        filePath = filedialog.asksaveasfile(title="Save State")
        if filePath == None or filePath.name == '':
            return None
        self.core.save_state(filePath.name)


    def load_state(self):
        filePath = filedialog.askopenfilename(title="State")
        if filePath == None or filePath == '':
            return None
        self.core.load_state(filePath)


    def onQuit(self):
        # paramFile =en('param.ini', 'w')
        # paramFile.write(self.saveDir)
        self.root.destroy()
        self.root.quit()
