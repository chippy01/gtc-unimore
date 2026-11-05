#!/usr/bin/python3
"""
Small GUI for tools/compare_compression.sh: choose a texture, set the parameters, run the comparison, and browse
the resulting plots. Start it with: tools/gui.py
The system Python is used on purpose: the conda environment's Tk has no scalable fonts, and the GUI only needs Tk
and Pillow (Fedora: python3-tkinter, python3-pillow-tk). The comparison itself still runs in the image-gs environment.
"""
import glob
import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "compare_compression.sh")
COMPARISONS = os.path.join(ROOT, "comparisons")
EMPTY_MODES = ["uv", "none", "grey", "black", "white", "blurred"]
PLOTS = [("Quality vs size", "rd_curves.png"), ("Quality vs size on disk", "rd_zstd.png"),
         ("Quality vs GPU memory", "rd_gpu.png"), ("Sizes on disk and in GPU memory", "sizes.png")]
IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.exr"), ("All files", "*")]
UV_TYPES = [("Mesh or mask image", "*.obj *.png *.jpg *.jpeg *.tif *.tiff"), ("All files", "*")]
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
SURFACE = "#fcfcfb"  # Same background as the plots


class App:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.stopping = False
        self.output = queue.Queue()
        self.photos = {}  # Tk does not keep references to displayed images
        self.image_cache = {}
        self.render_jobs = {}
        self.comparison_dirs = {}
        root.title("GTC compression benchmark")
        root.geometry("1500x900")
        try:
            root.attributes("-zoomed", True)  # Start maximized, the plots need the room
        except tk.TclError:
            pass
        ttk.Style().theme_use("clam")
        # Tk's default sizes are tiny on high-resolution screens
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(size=11)
        self.hint_font = tkfont.nametofont("TkDefaultFont").copy()
        self.hint_font.configure(size=9)
        paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left, right = ttk.Frame(paned, padding=10), ttk.Frame(paned, padding=10)
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        self._build_inputs(left)
        self._build_parameters(left)
        self._build_run(left)
        self._build_plots(right)
        self.refresh_comparisons()
        root.after(100, self._drain_output)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # Inputs and parameters

    def _build_inputs(self, parent):
        frame = ttk.LabelFrame(parent, text="Texture", padding=8)
        frame.pack(fill=tk.X)
        frame.columnconfigure(1, weight=1)
        self.input_kind = tk.StringVar(value="folder")
        self.folder, self.texture, self.uv_file = tk.StringVar(), tk.StringVar(), tk.StringVar()
        ttk.Radiobutton(frame, text="Texture folder, with its .obj mesh (e.g. RWT85)", variable=self.input_kind,
                        value="folder", command=self._update_inputs).grid(row=0, column=0, columnspan=3, sticky="w")
        self.folder_widgets = self._path_row(frame, 1, "Folder", self.folder, lambda: self._browse_dir(self.folder))
        ttk.Radiobutton(frame, text="Texture file + UV map", variable=self.input_kind, value="files",
                        command=self._update_inputs).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.file_widgets = self._path_row(frame, 3, "Texture", self.texture,
                                           lambda: self._browse_file(self.texture, IMAGE_TYPES))
        self.file_widgets += self._path_row(frame, 4, "UV map", self.uv_file,
                                            lambda: self._browse_file(self.uv_file, UV_TYPES))
        self._hint(frame, 5, ".obj mesh, or a mask image where white = used. Empty: the .obj next to the texture")
        self._update_inputs()

    def _build_parameters(self, parent):
        frame = ttk.LabelFrame(parent, text="Parameters", padding=8)
        frame.pack(fill=tk.X, pady=(10, 0))
        frame.columnconfigure(1, weight=1)
        self.gaussians = tk.StringVar(value="2500 5000 10000 20000 40000")
        self.scales = tk.StringVar(value="1 2 4 8 16")
        self.empty_mode = tk.StringVar(value="uv")
        self.tol, self.extra, self.label = tk.StringVar(), tk.StringVar(), tk.StringVar()
        # On by default: training an 8196x8196 texture does not fit into 8 GB of GPU memory
        self.cap_enabled, self.cap_size = tk.BooleanVar(value=True), tk.StringVar(value="4096")
        cap = ttk.Frame(frame)
        ttk.Checkbutton(cap, text="Cap textures at", variable=self.cap_enabled, command=self._update_cap).pack(side=tk.LEFT)
        self.cap_entry = ttk.Entry(cap, textvariable=self.cap_size, width=6)
        self.cap_entry.pack(side=tk.LEFT, padx=(6, 4))
        ttk.Label(cap, text="px (longest side)").pack(side=tk.LEFT)
        # (label, widget, hint, stretch the widget to the frame width)
        rows = [("Gaussian counts", ttk.Entry(frame, textvariable=self.gaussians),
                 "One Image-GS training run each, empty = none", True),
                ("BC/ASTC scales", ttk.Entry(frame, textvariable=self.scales),
                 "Downscaling factors, 4 = 1/4 width and height", True),
                ("Empty-area mode", ttk.Combobox(frame, textvariable=self.empty_mode, values=EMPTY_MODES, state="readonly",
                                                 width=10), "Used for training and for scoring every method", False),
                ("Tolerance", ttk.Entry(frame, textvariable=self.tol, width=10),
                 "Empty = default (2 levels for flat fills, 0.25 for blurred)", False),
                ("Extra main.py options", ttk.Entry(frame, textvariable=self.extra),
                 "e.g. --pos_bits=12 --feat_bits=10", True),
                ("Experiment label", ttk.Entry(frame, textvariable=self.label),
                 "Empty = derived from the extra options", True),
                ("GPU memory", cap, "Larger textures are scaled down first; results get a _max<px> suffix", False)]
        for index, (text, widget, hint, stretch) in enumerate(rows):
            ttk.Label(frame, text=text).grid(row=2*index, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
            widget.grid(row=2*index, column=1, sticky="we" if stretch else "w", pady=(6, 0))
            self._hint(frame, 2*index + 1, hint)

    def _update_cap(self):
        self.cap_entry.configure(state=tk.NORMAL if self.cap_enabled.get() else tk.DISABLED)

    def _build_run(self, parent):
        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, pady=(10, 0))
        self.run_button = ttk.Button(buttons, text="Run comparison", command=self.run)
        self.run_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(6, 0))
        self.status = tk.StringVar(value="Idle")
        ttk.Label(buttons, textvariable=self.status).pack(side=tk.LEFT, padx=(12, 0))
        self.log = ScrolledText(parent, width=56, height=20, state=tk.DISABLED, wrap=tk.WORD, font=("monospace", 10))
        self.log.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _path_row(self, frame, row, text, variable, browse):
        label = ttk.Label(frame, text=text)
        label.grid(row=row, column=0, sticky="w", padx=(18, 8), pady=(4, 0))
        entry = ttk.Entry(frame, textvariable=variable, width=32)
        entry.grid(row=row, column=1, sticky="we", pady=(4, 0))
        button = ttk.Button(frame, text="Browse…", command=browse)
        button.grid(row=row, column=2, padx=(6, 0), pady=(4, 0))
        return [label, entry, button]

    def _hint(self, frame, row, text):
        ttk.Label(frame, text=text, foreground="#52514e", font=self.hint_font).grid(row=row, column=1, columnspan=2,
                                                                                    sticky="w")

    def _update_inputs(self):
        folder_mode = self.input_kind.get() == "folder"
        for widget in self.folder_widgets:
            widget.configure(state=tk.NORMAL if folder_mode else tk.DISABLED)
        for widget in self.file_widgets:
            widget.configure(state=tk.DISABLED if folder_mode else tk.NORMAL)

    def _browse_dir(self, variable):
        path = filedialog.askdirectory(initialdir=ROOT, title="Choose a texture folder")
        if path:
            variable.set(path)

    def _browse_file(self, variable, filetypes):
        path = filedialog.askopenfilename(initialdir=ROOT, filetypes=filetypes)
        if path:
            variable.set(path)

    # Running compare_compression.sh

    def build_command(self):
        """Returns the compare_compression.sh command line, or raises ValueError with a message for the user."""
        counts, scales, tol = self.gaussians.get().split(), self.scales.get().split(), self.tol.get().strip()
        if not all(count.isdigit() and int(count) > 0 for count in counts):
            raise ValueError("Gaussian counts must be positive whole numbers separated by spaces")
        if len(scales) == 0 or not all(scale.isdigit() and int(scale) > 0 for scale in scales):
            raise ValueError("Give at least one BC/ASTC scale, as positive whole numbers separated by spaces")
        if tol:
            try:
                float(tol)
            except ValueError:
                raise ValueError("The tolerance must be a number") from None
        command = ["bash", SCRIPT, "-g", " ".join(counts), "-s", " ".join(scales), "-m", self.empty_mode.get()]
        for flag, value in (("-t", tol), ("-x", self.extra.get().strip()), ("-n", self.label.get().strip())):
            if value:
                command += [flag, value]
        if self.cap_enabled.get():
            cap = self.cap_size.get().strip()
            if not (cap.isdigit() and int(cap) > 0):
                raise ValueError("The resolution cap must be a positive whole number of pixels")
            command += ["-r", cap]
        if self.input_kind.get() == "folder":
            target = self.folder.get().strip()
            if not os.path.isdir(target):
                raise ValueError("Choose an existing texture folder")
        else:
            target, uv_file = self.texture.get().strip(), self.uv_file.get().strip()
            if not os.path.isfile(target):
                raise ValueError("Choose an existing texture file")
            if uv_file:
                if not os.path.isfile(uv_file):
                    raise ValueError("The UV map file does not exist")
                command += ["-u", uv_file]
        return command + [target]

    def run(self):
        try:
            command = self.build_command()
        except ValueError as error:
            messagebox.showerror("Cannot start the comparison", str(error))
            return
        self._append(f"$ {shlex.join(command)}\n")
        env = dict(os.environ, PYTHONUNBUFFERED="1")  # Stream Python output into the log as it happens
        # A separate session puts the script and everything it starts in one process group, so Stop ends all of them
        self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, start_new_session=True)
        self.stopping = False
        threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status.set("Running…")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.stopping = True
            self.status.set("Stopping…")
            os.killpg(self.process.pid, signal.SIGTERM)

    def _read_output(self, process):
        """Runs in a background thread, Tk widgets are only touched from the main thread in _drain_output()."""
        for line in process.stdout:
            self.output.put(line)
        self.output.put(("finished", process.wait()))

    def _drain_output(self):
        try:
            while True:
                item = self.output.get_nowait()
                if isinstance(item, tuple):
                    self._finished(item[1])
                elif "%|" not in item:  # Skip progress bars, they arrive as one line per update
                    self._append(ANSI_ESCAPE.sub("", item))
        except queue.Empty:
            pass
        self.root.after(100, self._drain_output)

    def _finished(self, returncode):
        self.process = None
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        if self.stopping:
            self.status.set("Stopped")
        elif returncode == 0:
            self.status.set("Finished")
        else:
            self.status.set(f"Failed (exit code {returncode}), see the log")
        self._append(f"--- {self.status.get()}\n")
        self.refresh_comparisons(select_newest=True)

    def _append(self, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def on_close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Comparison running", "A comparison is still running. Stop it and quit?"):
                return
            self.stop()
        self.root.destroy()

    # Plots

    def _build_plots(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Comparison").pack(side=tk.LEFT)
        self.comparison = tk.StringVar()
        self.comparison_box = ttk.Combobox(top, textvariable=self.comparison, state="readonly", width=34)
        self.comparison_box.pack(side=tk.LEFT, padx=(8, 0))
        self.comparison_box.bind("<<ComboboxSelected>>", lambda event: self.show_plots())
        ttk.Button(top, text="Refresh", command=self.refresh_comparisons).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Open folder", command=self.open_folder).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(parent, text="Double-click a plot to open it full size", foreground="#52514e",
                  font=self.hint_font).pack(side=tk.BOTTOM, anchor="w", pady=(4, 0))
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.canvases = {}
        for title, fname in PLOTS:
            # A canvas (not a label) so that the image never changes the size the layout gives it
            canvas = tk.Canvas(notebook, width=1, height=1, background=SURFACE, highlightthickness=0)
            canvas.bind("<Configure>", lambda event, name=fname: self._schedule_render(name))
            canvas.bind("<Double-Button-1>", lambda event, name=fname: self.open_plot(name))
            notebook.add(canvas, text=title)
            self.canvases[fname] = canvas

    def refresh_comparisons(self, select_newest=False):
        directories = [directory for directory in glob.glob(os.path.join(COMPARISONS, "*", "*"))
                       if any(os.path.isfile(os.path.join(directory, fname)) for _, fname in PLOTS)]
        directories.sort(key=os.path.getmtime, reverse=True)
        self.comparison_dirs = {os.path.relpath(directory, COMPARISONS): directory for directory in directories}
        names = list(self.comparison_dirs)
        self.comparison_box["values"] = names
        if names and (select_newest or self.comparison.get() not in self.comparison_dirs):
            self.comparison.set(names[0])
        self.show_plots()

    def show_plots(self):
        for _, fname in PLOTS:
            self._render(fname)

    def _plot_path(self, fname):
        directory = self.comparison_dirs.get(self.comparison.get())
        path = None if directory is None else os.path.join(directory, fname)
        return path if path is not None and os.path.isfile(path) else None

    def _schedule_render(self, fname):
        """Redraw once resizing pauses, instead of on every intermediate size."""
        if fname in self.render_jobs:
            self.root.after_cancel(self.render_jobs[fname])
        self.render_jobs[fname] = self.root.after(150, lambda: self._render(fname))

    def _render(self, fname):
        self.render_jobs.pop(fname, None)
        canvas = self.canvases[fname]
        width, height = canvas.winfo_width(), canvas.winfo_height()
        canvas.delete("all")
        path = self._plot_path(fname)
        if path is None:
            canvas.create_text(width // 2, height // 2, text="No plot yet: run a comparison", fill="#52514e")
            return
        if width < 50 or height < 50:
            return
        key = (path, os.path.getmtime(path))
        if key not in self.image_cache:
            self.image_cache[key] = Image.open(path).convert("RGB")
        image = self.image_cache[key].copy()
        image.thumbnail((width - 10, height - 10), Image.LANCZOS)
        self.photos[fname] = ImageTk.PhotoImage(image)
        canvas.create_image(width // 2, height // 2, image=self.photos[fname])

    def open_plot(self, fname):
        path = self._plot_path(fname)
        if path is not None:
            subprocess.Popen(["xdg-open", path])

    def open_folder(self):
        directory = self.comparison_dirs.get(self.comparison.get())
        if directory is not None:
            subprocess.Popen(["xdg-open", directory])


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
