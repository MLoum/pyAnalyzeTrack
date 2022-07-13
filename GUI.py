import tkinter as tk
from tkinter import ttk
from Core import AnalyzeTrackCore

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.patches as patches

from tkinter import filedialog, messagebox
import os
import numpy as np



class pyAnalyzeTrack():
	def __init__(self):
		self.root = tk.Tk()
		self.core = AnalyzeTrackCore()	# Model in model/View Pattern
		# self.camera = DummyCamera()
		self.iid_track_dict = {}


		self.create_gui()
		self.create_menu()

		self.root.protocol("WM_DELETE_WINDOW", self.onQuit)


	def run(self):
		self.root.title("py Analyze Track")
		self.root.deiconify()
		self.root.mainloop()


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
		self.tree_view["columns"] = ("num", "nb Spot", "r gauss", "r cov", "t", "x", "y", "z")
		# remove first empty column with the identifier
		# self.tree_view['show'] = 'headings'
		# tree.column("#0", width=270, minwidth=270, stretch=tk.NO) tree.column("one", width=150, minwidth=150, stretch=tk.NO) tree.column("two", width=400, minwidth=200) tree.column("three", width=80, minwidth=50, stretch=tk.NO)

		columns_text = ["num", "nb Spot", "r gauss", "r cov", "t", "x", "y", "z"]

		self.tree_view.column("#0", width=25, stretch=tk.NO)
		self.tree_view.column(columns_text[0], width=75, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[1], width=50, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[2], width=75, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[3], width=75, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[4], width=75, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[5], width=75, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[6], width=50, stretch=tk.YES, anchor=tk.CENTER)
		self.tree_view.column(columns_text[7], width=75, stretch=tk.YES, anchor=tk.CENTER)
		# self.tree_view.column(columns_text[8], width=50, stretch=tk.YES, anchor=tk.CENTER)
		# self.tree_view.column(columns_text[9], width=50, stretch=tk.YES, anchor=tk.CENTER)
		# self.tree_view.column(columns_text[10], width=50, stretch=tk.YES, anchor=tk.CENTER)
		# self.tree_view.column(columns_text[11], width=50, stretch=tk.YES, anchor=tk.CENTER)
		# self.tree_view.column(columns_text[12], width=50, stretch=tk.YES, anchor=tk.CENTER)
		# self.tree_view.column(columns_text[13], width=50, stretch=tk.YES, anchor=tk.CENTER)
		# self.tree_view.column(columns_text[14], width=50, stretch=tk.YES, anchor=tk.CENTER)
		# self.tree_view.column(columns_text[15], width=50, stretch=tk.YES, anchor=tk.CENTER)

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


	def create_params(self):
		# Frame Param Exp
		#################
		ttk.Label(self.frame_param_exp, text='T (°C) :').grid(row=0, column=0, padx=8)
		self.T_sv = tk.StringVar(value='20')
		ttk.Entry(self.frame_param_exp, textvariable=self.T_sv, justify=tk.CENTER, width=7).grid(row=0, column=1,
																								 padx=8)

		# TODO combo avec solvant
		ttk.Label(self.frame_param_exp, text='eta (Pa.s) :').grid(row=1, column=0, padx=8)
		self.eta_sv = tk.StringVar(value='0.001')
		ttk.Entry(self.frame_param_exp, textvariable=self.eta_sv, justify=tk.CENTER, width=7).grid(row=1, column=1,
																								   padx=8)
		self.solvant_combo_sv = tk.StringVar()
		cb = ttk.Combobox(self.frame_param_exp, width=13, justify=tk.CENTER, textvariable=self.solvant_combo_sv,
						  values='')
		# cb.bind('<<ComboboxSelected>>', self.change_algo)
		cb['values'] = ('water', 'DMSO', 'ethanol')
		self.solvant_combo_sv.set('water')
		cb.grid(row=1, column=2, padx=8)

		ttk.Label(self.frame_param_exp, text='Δt (ms) :').grid(row=2, column=0, padx=8)
		self.delta_t_sv = tk.StringVar(value='33')
		ttk.Entry(self.frame_param_exp, textvariable=self.delta_t_sv, justify=tk.CENTER, width=7).grid(row=2, column=1,
																									   padx=8)

		ttk.Label(self.frame_param_exp, text='Δpix (nm) :').grid(row=3, column=0, padx=8)
		self.Delta_pix_sv = tk.StringVar(value='263')
		ttk.Entry(self.frame_param_exp, textvariable=self.Delta_pix_sv, justify=tk.CENTER, width=7).grid(row=3,column = 1,padx=8)


		ttk.Label(self.frame_param_exp, text='Nombre de points minimum acceptable dans une trajectoire').grid(row=4,column=0,padx=8)
		self.nb_spot_min_sv= tk.StringVar(value='10')
		e = ttk.Entry(self.frame_param_exp, textvariable=self.nb_spot_min_sv, justify=tk.CENTER, width=7).grid(row=4, column = 1, padx=8)





		# Frame Param Algo
		##################
		ttk.Label(self.frame_param_exp, text='Drift compensation :').grid(row=0, column=0, padx=8)
		self.drift_algo_cv = tk.StringVar()
		cb = ttk.Combobox(self.frame_param_algo, width=13, justify=tk.CENTER, textvariable=self.drift_algo_cv,
						  values='')
		# cb.bind('<<ComboboxSelected>>', self.change_algo)
		cb['values'] = ('None', "neighbors", 'self')
		self.drift_algo_cv.set('neighbors')
		cb.grid(row=0, column=1, padx=8)

	def create_filters(self):
		pad_filter = 5

		# ttk.Label(self.frame_filter, text='           Filter 1           ', background="white").grid(row=0,
		# 																							 column=0,
		# 																							 columnspan=6,
		# 																							 padx=pad_filter,
		# 																							 pady=10)
		# ttk.Label(self.frame_filter, text='           Filter 2           ', background="white").grid(row=2,
		# 																							 column=7,
		# 																							 columnspan=6,
		# 																							 padx=pad_filter)

		self.is_not_f1 = True
		self.button_not_f1 = tk.Button(self.frame_filter, text="not", width=3, command=self.toggle_not_f1)
		self.button_not_f1.grid(row=0, column=0, padx=pad_filter)

		# So that not start as false
		self.toggle_not_f1()

		self.filter_1_low_sv = tk.StringVar()
		ttk.Entry(self.frame_filter, textvariable=self.filter_1_low_sv, justify=tk.CENTER, width=12).grid(row=0,
																										  column=1,
																										  padx=pad_filter)
		ttk.Label(self.frame_filter, text=' < ').grid(row=0, column=2, padx=pad_filter)

		self.cb_value_filter_1_sv = tk.StringVar()
		self.cb_value_filter_1 = ttk.Combobox(self.frame_filter, width=25, justify=tk.CENTER,
											  textvariable=self.cb_value_filter_1_sv, values='')
		self.cb_value_filter_1['values'] = ["None", "duration", "nb_photon", "CPS", "p1", "p2", "p3", "p4", "p5", "p6",
											"p7", "p8"]
		self.cb_value_filter_1.set('None')
		self.cb_value_filter_1.bind('<<ComboboxSelected>>', self.change_filter1_type)
		self.cb_value_filter_1.grid(row=0, column=3, padx=pad_filter)

		ttk.Label(self.frame_filter, text=' < ').grid(row=0, column=4, padx=pad_filter)
		self.filter_1_high_sv = tk.StringVar()
		ttk.Entry(self.frame_filter, textvariable=self.filter_1_high_sv, justify=tk.CENTER, width=12).grid(row=0,
																										   column=5,
																										   padx=pad_filter)

		self.cb_value_filter_bool_op_sv = tk.StringVar()
		self.cb_value_filter_bool_op = ttk.Combobox(self.frame_filter, width=25, justify=tk.CENTER,
													textvariable=self.cb_value_filter_bool_op_sv, values='')
		self.cb_value_filter_bool_op['values'] = ["and", "or", "xor"]
		self.cb_value_filter_bool_op.set('or')

		self.cb_value_filter_bool_op.grid(row=1, column=0, columnspan=6, padx=pad_filter, pady=10)

		self.is_not_f2 = True
		self.button_not_f2 = tk.Button(self.frame_filter, text="not", width=3, command=self.toggle_not_f2)
		self.button_not_f2.grid(row=2, column=0, padx=pad_filter)
		# So that not start as false
		self.toggle_not_f2()

		self.filter_2_low_sv = tk.StringVar()
		ttk.Entry(self.frame_filter, textvariable=self.filter_2_low_sv, justify=tk.CENTER, width=12).grid(row=2,
																										  column=1,
																										  padx=pad_filter)
		ttk.Label(self.frame_filter, text=' < ').grid(row=2, column=2, padx=pad_filter)

		self.cb_value_filter_2_sv = tk.StringVar()
		self.cb_value_filter_2 = ttk.Combobox(self.frame_filter, width=25, justify=tk.CENTER,
											  textvariable=self.cb_value_filter_2_sv, values='')
		self.cb_value_filter_2['values'] = ["None", "duration", "nb_photon", "CPS", "p1", "p2", "p3", "p4", "p5", "p6",
											"p7", "p8"]
		self.cb_value_filter_2.set('None')
		self.cb_value_filter_2.bind('<<ComboboxSelected>>', self.change_filter2_type)
		self.cb_value_filter_2.grid(row=2, column=3, padx=pad_filter)

		ttk.Label(self.frame_filter, text=' < ').grid(row=2, column=4, padx=pad_filter)

		self.filter_2_high_sv = tk.StringVar()
		ttk.Entry(self.frame_filter, textvariable=self.filter_2_high_sv, justify=tk.CENTER, width=12).grid(row=2, column=5, padx=pad_filter)

		ttk.Button(self.frame_filter, text="Filter", width=3, command=self.launch_filter).grid(row=3, column=0,
																							   columnspan=6,
																							   padx=pad_filter)
		ttk.Button(self.frame_filter, text="CLEAR Filter", command=self.clear_filter).grid(row=4, column=0,
																								 columnspan=6,
																								 padx=pad_filter)

	def launch_filter(self):
		def convert_sv_float(sv):
			str_ = sv.get()
			try:
				f = float(str_)
				return f
			except ValueError:
				return 0

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
		pass

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
		self.canvas_result_all_tracks= FigureCanvasTkAgg(self.fig_result_all_tracks, master=self.frame_graph_all_tracks)
		self.canvas_result_all_tracks.draw()
		#self.canvas_result_all_track.get_tk_widget().pack(side='top', fill='both', expand=1)

		self.toolbar = NavigationToolbar2Tk(self.canvas_result_all_tracks, self.frame_graph_all_tracks)
		self.canvas_result_all_tracks._tkcanvas.pack(side='top', fill='both', expand=1)

		self.plot_all_mode_sv = tk.StringVar()
		cb = ttk.Combobox(self.frame_graph_all_tracks_params, width=13, justify=tk.CENTER,
						  textvariable=self.plot_all_mode_sv,
						  values='')


		# cb.bind('<<ComboboxSelected>>', self.change_algo)
		cb['values'] = ('Sum gaussian', 'Lags', 'Radius Histogram')
		self.plot_all_mode_sv.set('Sum gaussian')
		cb.grid(row=0, column=0, padx=8)

		tk.Label(self.frame_graph_all_tracks_params, text='Rayon (Covariance) =').grid(row=3, column=0)
		self.max_index_sv = tk.StringVar() # Valeur en x max de la gaussienne

		self.maximum = tk.Label(self.frame_graph_all_tracks_params, textvariable=self.max_index_sv).grid(row=3			, column=1)





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
		cb = ttk.Combobox(self.frame_graph_current_track_params, width=13, justify=tk.CENTER, textvariable=self.plot_current_mode_sv,
						  values='')
		# cb.bind('<<ComboboxSelected>>', self.change_algo)
		cb['values'] = ('track', 'Histogram displacement', 'test free motion via cov"','Lags','DCT')
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
		self.menu_file.add_command(label='generate', underline=1, accelerator="Ctrl+g", command=self.generate_data)
		self.menu_system.add_cascade(label="File", menu=self.menu_file)

		self.root.config(menu=self.menu_system)

	# self.master.bind_all("<Control-o>", self.askOpenSPC_file)

	def generate_data(self):
		#TODO creer une interface graphique pour collecter les parameters

		self.core.generate_brownian_track(params_dict=None)
		self.insert_tracks_tree_view()
		self.plot_result_all_track()
		pass

	def open_xml_trackmate(self):
		filePath = filedialog.askopenfilename(title="Open Trackmate xml file")
		# TODO logic in controller ?
		if filePath == None or filePath == '':
			return None
		else:
			extension = os.path.splitext(filePath)[1]
			if extension not in (".xml", ".txt"):
				messagebox.showwarning("Open file",
									   "The file has not the correct .spc, .pt3, .ttt, extension. Aborting")
				return None
			else:
				self.saveDir = os.path.split(filePath)[0]
				if extension == ".xml":
					self.core.loadxmlTrajs(filePath)
				elif extension == ".txt":
					self.core.change_filter(self.nb_spot_min_sv)
					self.core.load_txt(filePath)
				self.insert_tracks_tree_view()
				self.plot_result_all_track()

	def fill_treeview_with_tracks(self):
		self.clear_treeview()

	def clear_treeview(self):
		self.tree_view.delete(*self.tree_view.get_children())

	def insert_tracks_tree_view(self):
		for num_track in range(self.core.nTracks):
			# print(num_track)
			track = self.core.tracks[num_track]
			iid_track = self.tree_view.insert(parent="", index='end', values=(
				str(num_track), str(track.nSpots), str(track.r_gauss) + "+/-" + str(track.error_r_gauss), str(track.r_cov) + "+/-" + str(track.error_r_cov), "", "", "", ""))
			self.iid_track_dict[str(num_track)] = iid_track
			# print (track_data.size)


			# insert the position of the spots -> Time consuming

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
			y = self.core.Controle*0
			z = self.core.Controle_variance
			self.ax_result_all_tracks.plot(x)
			self.ax_result_all_tracks.errorbar(range(len(x)),x,yerr = z)
			self.ax_result_all_tracks.set_xlabel("Time Lags j")
			self.ax_result_all_tracks.set_ylabel("<$\Delta x_n * \Delta x_{n+m}$>")
			self.fig_result_all_tracks.canvas.draw()
		elif self.plot_all_track_mode == "Radius Histogram":
			self.ax_result_all_tracks.clear()
			self.fig_result_all_tracks.set_tight_layout(True)
			# Histoy,Histox = np.histogram(self.core.Moyenner,bins=100)
			Histoy, Histox = np.histogram(self.core.Moyenner,bins='auto')
			self.ax_result_all_tracks.bar(Histox[:-1], Histoy)
			self.ax_result_all_tracks.set_xlim([self.core.lim_min*10**9,self.core.lim_max*10**9])
			self.ax_result_all_tracks.set_xlabel("Radius (nm)")
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

		#FIXME 16 -> nb magique
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
		y = self.core.Controle_track[:,num_track]
		x = self.core.Controle_track[:,num_track]*0
		z = self.core.Controle_track_variance[:,num_track]
		self.ax_result_current_track.clear()
		self.fig_result_current_track.set_tight_layout(True)
		self.ax_result_current_track.plot(y)
		self.ax_result_current_track.errorbar(range(len(y)),y,yerr = z)
		self.ax_result_current_track.set_xlabel("Lags j")
		self.ax_result_current_track.set_ylabel(r"$\Delta x_n \Delta x_{n + j}$")
		self.fig_result_current_track.canvas.draw()
	def plot_DCT(self,num_track):
		y = self.core.Spectre_mean[num_track]
		z = np.arange(0.5*self.core.taille[num_track],len(y)*self.core.taille[num_track],self.core.taille[num_track])
		self.ax_result_current_track.clear()
		self.fig_result_current_track.set_tight_layout(True)
		self.ax_result_current_track.scatter(z,y)
		self.ax_result_current_track.set_xlabel("Mode k")
		self.ax_result_current_track.set_ylabel("$P_k / [D(\Delta t^2$]")
		self.ax_result_current_track.set_xlim([0, self.core.taille[num_track]*self.core.division])
		self.ax_result_current_track.set_ylim([0,4])
		self.fig_result_current_track.canvas.draw()


	def plot_gaussian_fit(self, num_track):
		def gaussian(x, result_fit):
			"""
			Intermediate helpinf function for calculating more point for the x_axis of the gaussian fit
			"""
			amp, cen, wid = result_fit.params["amp"], result_fit.params["cen"], result_fit.params["wid"]
			return amp * np.exp(-(x - cen) ** 2 / wid)

		# FIXME ? Recalculating the data ?
		diff_x, hist_x, boundaries_x, result_x, center_x, width_x = self.core.gaussian_delta_fit(self.core.tracks[num_track].x)
		diff_y, hist_y, boundaries_y, result_y, center_y, width_y = self.core.gaussian_delta_fit(self.core.tracks[num_track].y)


		self.ax_result_current_track.clear()
		self.fig_result_current_track.set_tight_layout(True)

		self.ax_result_current_track.bar(boundaries_x, hist_x, alpha=0.5)
		x = np.linspace(np.min(boundaries_x), np.max(boundaries_x), 150)
		fit_x = gaussian(x, result_x)
		self.ax_result_current_track.plot(x, fit_x)

		self.ax_result_current_track.bar(boundaries_y, hist_y, alpha=0.5)
		y = np.linspace(np.min(boundaries_y), np.max(boundaries_y), 150)
		fit_y = gaussian(y, result_y)
		self.ax_result_current_track.plot(y, fit_y)

		self.fig_result_current_track.canvas.draw()


	def onQuit(self):
		# paramFile = open('param.ini', 'w')
		# paramFile.write(self.saveDir)
		self.root.destroy()
		self.root.quit()

