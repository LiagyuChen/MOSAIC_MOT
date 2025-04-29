import os
import cv2
import time
import requests
import threading
import tkinter as tk
import multiprocessing
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox
from mosaic_mot import mosaic_mot
from mosaic_det_seg import MosaicDetSeg


# Model URLs
MODEL_URLS = {
    "yoloe-11l-seg-pf.pt": "https://github.com/fudan-zvg/SOFT/releases/download/v0.1.0/yoloe-11l-seg-pf.pt",
    "FastSAM-s.pt": "https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/FastSAM-s/FastSAM-s.pt"
}

mosaic_det_seg_default_config = {
    # Few-Shot Selector configuration
    "few_shot_configs": {
        # DINOv2 layers
        "early_layer": 4,
        "mid_layer": 8,
        "late_layer": 12,
        "early_layer_weight": 0.1,
        "mid_layer_weight": 0.3,
        "late_layer_weight": 0.6,
        # Performance settings
        "max_workers": min(16, multiprocessing.cpu_count()),
        "batch_size": 32,
        # Image processing
        "min_image_size": 20,
        "aug_angles": [30, 60, 90, 150, 180, 270],
        "aug_scales": [0.5, 0.75],
        # SLIC settings
        "slic_n_segments": 16,
        "slic_compactness": 15,
        "slic_sigma": 3.0,
    },

    # Mosaic Detection and Segmentation configuration
    "mosaic_mode": "accurate",
    "large_image_threshold": 1088,
    "mask_area_thresh": 0.8,
    "conflict_thresh": 0.1,
    "mask_contrain_ratio": 0.95,
    "bridge_thresh": 3,
    "mask_nms_iou": 0.9,
    "box_nms_iou": 0.9,
    "similarity_threshold": 0.6,
    "over_match_thresh": 0.45,
    "patch_match_thresh": 0.45,
    "contour_weight": 0.5,
    "patch_weight": 0.5,
    "inner_mask_over_thresh": 0.7,
    "inner_mask_partial_thresh": 0.95,
    "shape_diff_thresh": 0.2,
    "patch_coverage_thresh": 0.3,
    "patch_missing_thresh": 0.05,
    "min_mask_area": 100,
    "visualization_alpha": 0.6,
}


mosaic_mot_default_config = {
    # Few-Shot Selector configurations
    "few_shot_configs": {
        # DINOv2 layers
        "early_layer": 4,
        "mid_layer": 8,
        "late_layer": 12,
        "early_layer_weight": 0.1,
        "mid_layer_weight": 0.3,
        "late_layer_weight": 0.6,
        # Performance settings
        "max_workers": min(16, multiprocessing.cpu_count()),
        "batch_size": 32,
        # Image processing
        "min_image_size": 20,
        "aug_angles": [30, 60, 90, 150, 180, 270],
        "aug_scales": [0.5, 0.75],
        # SLIC settings
        "slic_n_segments": 16,
        "slic_compactness": 15,
        "slic_sigma": 3.0,
    },

    # MOSAIC-DET-SEG configurations
    "mosaic_mode": "accurate",
    "large_image_threshold": 1088,
    "mask_area_thresh": 0.8,
    "conflict_thresh": 0.1,
    "mask_contrain_ratio": 0.95,
    "bridge_thresh": 3,
    "mask_nms_iou": 0.9,
    "box_nms_iou": 0.9,
    "over_match_thresh": 0.45,
    "patch_match_thresh": 0.45,
    "contour_weight": 0.5,
    "patch_weight": 0.5,
    "inner_mask_over_thresh": 0.7,
    "inner_mask_partial_thresh": 0.95,
    "shape_diff_thresh": 0.2,
    "patch_coverage_thresh": 0.3,
    "patch_missing_thresh": 0.05,
    "min_mask_area": 100,
    "similarity_threshold": 0.6,

    # MOSAIC-MOT specific parameters
    "max_frames_missing": 30,
    "frames_to_cache": 3,
    "bbox_iou_threshold": 0.4,
    "mask_iou_threshold": 0.3,
    "embedding_similarity_threshold": 0.6,
    "size_prior_threshold": 0.3,
    "hu_weight": 0.3,
    "embedding_weight": 0.7,
    "min_area_ratio": 0.4,
    "class_similarity_threshold": 0.6,

    # Visualization configuration
    "draw_color": (255, 0, 0),
    "line_thickness": 2,
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.5,
    "text_color": (255, 0, 0),
    "visualization_alpha": 0.6,
}

# Default configuration
DEFAULT_CONFIG = {
    "mode": "accurate",
    "similarity_threshold": 0.7,
    "embedding_similarity_threshold": 0.7,
    "detection_threshold": 30,
    "detection_score_threshold": 0.3,
    "segmentation_threshold": 0.25
}

# Theme colors
THEME = {
    "bg": "#f0f0f0",
    "primary": "#1a73e8",
    "primary_light": "#4285f4",
    "secondary": "#34a853",
    "error": "#ea4335",
    "warning": "#fbbc05",
    "success": "#0f9d58",
    "text": "#202124",
    "text_light": "#5f6368",
    "disabled": "#dadce0",
    "border": "#dadce0",
    "panel_bg": "#ffffff"
}


class CustomButton(ttk.Button):
    """Custom styled button"""
    def __init__(self, parent, **kwargs):
        self.style_name = kwargs.pop('style_name', 'Custom.TButton')
        super().__init__(parent, style=self.style_name, **kwargs)


class DownloadProgressbar(ttk.Progressbar):
    """Custom progressbar for downloads with status label"""
    def __init__(self, parent, **kwargs):
        style = ttk.Style()
        style.configure("Download.Horizontal.TProgressbar", 
                        troughcolor=THEME["bg"], 
                        background=THEME["primary"])
        
        super().__init__(parent, style="Download.Horizontal.TProgressbar", **kwargs)
        self.status_label = ttk.Label(parent, text="Waiting...", foreground=THEME["text_light"])
        self.pack(fill=tk.X, padx=20, pady=(5, 0))
        self.status_label.pack(fill=tk.X, padx=20, pady=(0, 10))
        
    def update_status(self, text):
        self.status_label.config(text=text)


class DownloadThread(threading.Thread):
    """Thread for downloading model files"""
    def __init__(self, url, save_path, progress_callback=None, completion_callback=None):
        super().__init__()
        self.url = url
        self.save_path = save_path
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.daemon = True
        
    def run(self):
        try:
            filename = os.path.basename(self.save_path)
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            downloaded = 0
            
            with open(self.save_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    downloaded += len(data)
                    progress = int(downloaded / total_size * 100) if total_size > 0 else 0
                    if self.progress_callback:
                        self.progress_callback(filename, progress)
            
            if self.completion_callback:
                self.completion_callback(filename, self.save_path)
                
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(filename, -1, str(e))


class MosaicInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MOSAIC-MOT Interface")
        self.geometry("1200x800")  # Increased window size
        self.minsize(1000, 700)    # Increased minimum size

        # Configure the grid layout
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)  # Header
        self.rowconfigure(1, weight=1)  # Main content
        self.rowconfigure(2, weight=0)  # Status bar
        
        # Initialize variables
        self.current_source_image = None
        self.current_result_image = None
        self.source_tk_img = None
        self.result_tk_img = None
        self.video_source = None
        self.video_result = None
        self.video_playing = False
        
        # Initialize configs
        self.det_seg_config = mosaic_det_seg_default_config.copy()
        self.mot_config = mosaic_mot_default_config.copy()
        
        # Set up custom styles
        self.setup_styles()
        
        # Create UI elements
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
        # Start checking for required models
        self.after(100, self.check_required_models)  # Slight delay to allow UI to render first
    
    def setup_styles(self):
        """Set up custom styles for widgets"""
        self.style = ttk.Style()
        
        # Configure theme color scheme
        self.style.configure("TFrame", background=THEME["bg"])
        self.style.configure("TLabel", background=THEME["bg"], foreground=THEME["text"])
        self.style.configure("TButton", 
                           background=THEME["primary"], 
                           foreground="white",
                           font=("Segoe UI", 10))
        
        # Specific styles
        self.style.configure("Header.TFrame", background=THEME["primary"])
        self.style.configure("Header.TLabel", 
                           background=THEME["primary"], 
                           foreground="white", 
                           font=("Segoe UI", 18, "bold"),
                           padding=10)
        
        self.style.configure("Subheader.TLabel", 
                           background=THEME["bg"], 
                           foreground=THEME["text"], 
                           font=("Segoe UI", 12, "bold"))
        
        self.style.configure("FileFrame.TLabelframe", 
                           background=THEME["panel_bg"],
                           borderwidth=1, 
                           relief="groove")
        
        self.style.configure("FileFrame.TLabelframe.Label", 
                           background=THEME["panel_bg"],
                           foreground=THEME["text"],
                           font=("Segoe UI", 11, "bold"))
        
        self.style.configure("Content.TFrame", 
                           background=THEME["panel_bg"],
                           borderwidth=1,
                           relief="groove")
        
        self.style.configure("Status.TFrame", background=THEME["bg"])
        self.style.configure("Status.TLabel", 
                           background=THEME["bg"],
                           foreground=THEME["text_light"],
                           font=("Segoe UI", 9))
        
        # Button styles
        self.style.configure("Primary.TButton", 
                           background=THEME["primary"],
                           foreground="#000000")
        self.style.map("Primary.TButton",
                     background=[('active', THEME["primary_light"]), 
                                 ('disabled', THEME["disabled"])])
        
        self.style.configure("Success.TButton", 
                           background=THEME["success"],
                           foreground="#000000")
        self.style.map("Success.TButton",
                     background=[('active', THEME["success"]), 
                                 ('disabled', THEME["disabled"])])
    
    def create_header(self):
        """Create the header section"""
        header_frame = ttk.Frame(self, style="Header.TFrame")
        header_frame.grid(row=0, column=0, sticky="ew")
        
        # Logo/Title area
        header_label = ttk.Label(header_frame, 
                               text="MOSAIC-MOT Tracking Tool", 
                               style="Header.TLabel")
        header_label.pack(pady=10)
        
        # Subtitle
        subtitle = ttk.Label(header_frame, 
                           text="Multiple Object Tracking with Visual Features", 
                           background=THEME["primary"],
                           foreground="white",
                           font=("Segoe UI", 10))
        subtitle.pack(pady=(0, 10))
    
    def create_main_content(self):
        """Create the main content area"""
        main_frame = ttk.Frame(self, style="TFrame", padding=15)
        main_frame.grid(row=1, column=0, sticky="nsew")
        
        # Configure grid
        main_frame.columnconfigure(0, weight=1)  # Left column (path selector)
        main_frame.columnconfigure(1, weight=1)  # Right column (config panel)
        main_frame.rowconfigure(0, weight=0)     # Top row (controls) - smaller height
        main_frame.rowconfigure(1, weight=1)     # Bottom row (preview area) - larger height
        
        # Create side-by-side panels for controls 
        # Left side: File selection section
        self.create_file_selection(main_frame, grid_row=0, grid_col=0)
        
        # Right side: Configuration panel
        self.create_config_panel(main_frame, grid_row=0, grid_col=1)
        
        # Preview section (bottom area, spans both columns and takes most of the space)
        preview_frame = ttk.Frame(main_frame, style="Content.TFrame", padding=15)
        preview_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=10)
        
        # Configure preview frame grid
        preview_frame.columnconfigure(0, weight=1)  # Source preview
        preview_frame.columnconfigure(1, weight=1)  # Result preview
        preview_frame.rowconfigure(0, weight=0)     # Title - small height
        preview_frame.rowconfigure(1, weight=1)     # Content - large height
        
        # Preview title
        source_title = ttk.Label(preview_frame, 
                                text="Source", 
                                style="Subheader.TLabel")
        source_title.grid(row=0, column=0, sticky="w", pady=(0, 5))  # Reduced padding
        
        result_title = ttk.Label(preview_frame, 
                                text="Result", 
                                style="Subheader.TLabel")
        result_title.grid(row=0, column=1, sticky="w", pady=(0, 5))  # Reduced padding
        
        # Source preview content (with minimum size requirements)
        self.source_frame = ttk.Frame(preview_frame, style="Content.TFrame")
        self.source_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.source_frame.columnconfigure(0, weight=1)
        self.source_frame.rowconfigure(0, weight=1)
        
        self.source_canvas = tk.Canvas(self.source_frame, bg=THEME["panel_bg"], highlightthickness=0)
        self.source_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Result preview content (with minimum size requirements)
        self.result_frame = ttk.Frame(preview_frame, style="Content.TFrame")
        self.result_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(0, weight=1)
        
        self.result_canvas = tk.Canvas(self.result_frame, bg=THEME["panel_bg"], highlightthickness=0)
        self.result_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Placeholders
        self.source_placeholder = ttk.Label(
            self.source_frame, 
            text="Source content will appear here", 
            background=THEME["panel_bg"],
            foreground=THEME["text_light"],
            font=("Segoe UI", 11),
            justify="center")
        self.source_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        
        self.result_placeholder = ttk.Label(
            self.result_frame, 
            text="Processed results will appear here", 
            background=THEME["panel_bg"],
            foreground=THEME["text_light"],
            font=("Segoe UI", 11),
            justify="center")
        self.result_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        
        # Store references to important widgets
        self.preview_frame = preview_frame
        self.preview_frame.bind("<Configure>", self.on_preview_resize)
    
    def create_file_selection(self, parent, grid_row=0, grid_col=0):
        """Create the file selection section"""
        file_frame = ttk.LabelFrame(
            parent, 
            text="File Selection", 
            padding=(10, 5),  # Reduced padding
            style="FileFrame.TLabelframe")
        file_frame.grid(row=grid_row, column=grid_col, sticky="nsew", padx=5, pady=5)
        
        # Configure grid
        file_frame.columnconfigure(1, weight=1)
        
        # Video file selection
        ttk.Label(file_frame, 
                text="Source:", 
                background=THEME["panel_bg"],
                foreground=THEME["text"]).grid(row=0, column=0, sticky="w", padx=5, pady=4)  # Reduced padding
        
        self.video_path_var = tk.StringVar()
        video_entry = ttk.Entry(file_frame, textvariable=self.video_path_var, state="readonly")
        video_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=4)  # Reduced padding
        
        self.video_button = ttk.Button(
            file_frame, 
            text="Browse", 
            style="Primary.TButton",
            command=self.select_video_file)
        self.video_button.grid(row=0, column=2, padx=5, pady=4)  # Reduced padding
        
        # Features folder selection
        ttk.Label(file_frame, 
                text="Features:", 
                background=THEME["panel_bg"],
                foreground=THEME["text"]).grid(row=1, column=0, sticky="w", padx=5, pady=4)  # Reduced padding
        
        self.features_folder_var = tk.StringVar()
        features_entry = ttk.Entry(file_frame, textvariable=self.features_folder_var, state="readonly")
        features_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=4)  # Reduced padding
        
        self.features_button = ttk.Button(
            file_frame, 
            text="Browse", 
            style="Primary.TButton",
            command=self.select_features_folder)
        self.features_button.grid(row=1, column=2, padx=5, pady=4)  # Reduced padding
        
        # Output path selection
        ttk.Label(file_frame, 
                text="Output:", 
                background=THEME["panel_bg"],
                foreground=THEME["text"]).grid(row=2, column=0, sticky="w", padx=5, pady=4)  # Reduced padding
        
        self.output_path_var = tk.StringVar()
        output_entry = ttk.Entry(file_frame, textvariable=self.output_path_var, state="readonly")
        output_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=4)  # Reduced padding
        
        self.output_button = ttk.Button(
            file_frame, 
            text="Browse", 
            style="Primary.TButton",
            command=self.select_output_folder)
        self.output_button.grid(row=2, column=2, padx=5, pady=4)  # Reduced padding
        
        # Initially disable buttons with visual feedback
        self.video_button.state(['disabled'])
        self.features_button.state(['disabled'])
        self.output_button.state(['disabled'])
        
        # Add a notice about required models
        model_notice = ttk.Label(
            file_frame,
            text="The interface will be enabled once required models are downloaded.", 
            background=THEME["panel_bg"],
            foreground=THEME["text_light"],
            font=("Segoe UI", 9, "italic"))
        model_notice.grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=(10, 5))
        
        # Add start button at the bottom of the file selection panel
        start_button = ttk.Button(
            file_frame, 
            text="Start Processing", 
            style="Success.TButton",
            command=self.start_processing)
        start_button.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=(5, 2))  # Reduced padding
        
        self.start_button = start_button
        self.start_button.state(['disabled'])  # Initially disabled
        
        self.model_notice = model_notice
    
    def create_config_panel(self, parent, grid_row=0, grid_col=1):
        """Create the configuration panel"""
        config_frame = ttk.LabelFrame(
            parent, 
            text="Configuration", 
            padding=(10, 5),  # Reduced padding
            style="FileFrame.TLabelframe")
        config_frame.grid(row=grid_row, column=grid_col, sticky="nsew", padx=5, pady=5)
        
        # Configure grid
        config_frame.columnconfigure(1, weight=1)
        
        # Process mode (track or detect)
        ttk.Label(config_frame, 
                text="Process Mode:", 
                background=THEME["panel_bg"],
                foreground=THEME["text"]).grid(row=0, column=0, sticky="w", padx=5, pady=4)  # Reduced padding
        
        self.process_mode_var = tk.StringVar(value="Track")
        process_mode_combo = ttk.Combobox(config_frame, 
                                        textvariable=self.process_mode_var,
                                        values=["Track", "Detect"], 
                                        state="readonly",
                                        width=10)
        process_mode_combo.grid(row=0, column=1, sticky="w", padx=5, pady=4)  # Reduced padding
        process_mode_combo.bind("<<ComboboxSelected>>", self.on_process_mode_change)
        
        # Mosaic mode (accurate or fast)
        ttk.Label(config_frame, 
                text="Mosaic Mode:", 
                background=THEME["panel_bg"],
                foreground=THEME["text"]).grid(row=1, column=0, sticky="w", padx=5, pady=4)  # Reduced padding
        
        self.mosaic_mode_var = tk.StringVar(value="Accurate")
        mosaic_mode_combo = ttk.Combobox(config_frame, 
                                       textvariable=self.mosaic_mode_var,
                                       values=["Accurate", "Fast"], 
                                       state="readonly",
                                       width=10)
        mosaic_mode_combo.grid(row=1, column=1, sticky="w", padx=5, pady=4)  # Reduced padding
        
        # Similarity threshold slider
        ttk.Label(config_frame, 
                text="Similarity Threshold:", 
                background=THEME["panel_bg"],
                foreground=THEME["text"]).grid(row=2, column=0, sticky="w", padx=5, pady=4)  # Reduced padding
        
        slider_frame = ttk.Frame(config_frame, style="Content.TFrame")
        slider_frame.grid(row=2, column=1, sticky="ew", padx=5, pady=4)  # Reduced padding
        slider_frame.columnconfigure(0, weight=1)
        
        self.similarity_threshold_var = tk.DoubleVar(value=0.7)
        similarity_slider = ttk.Scale(slider_frame, 
                                    from_=0.0, to=1.0, 
                                    orient="horizontal",
                                    variable=self.similarity_threshold_var,
                                    command=self.update_threshold_label)
        similarity_slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        self.threshold_label = ttk.Label(slider_frame, 
                                      text="0.70", 
                                      width=4,
                                      background=THEME["panel_bg"],
                                      foreground=THEME["text"])
        self.threshold_label.grid(row=0, column=1)
    
    def create_status_bar(self):
        """Create the status bar"""
        status_frame = ttk.Frame(self, style="Status.TFrame", padding=5)
        status_frame.grid(row=2, column=0, sticky="ew")
        
        # Status message
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel")
        status_label.pack(side=tk.LEFT, padx=10)
        
        # Progress bar for model downloads
        self.progress_bar = DownloadProgressbar(status_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack_forget()  # Initially hidden
    
    def check_required_models(self):
        """Check if required models exist in the weights folder"""
        self.status_var.set("Checking for required models...")
        
        # Create weights folder if it doesn't exist
        weights_folder = os.path.join(os.getcwd(), "weights")
        os.makedirs(weights_folder, exist_ok=True)
        
        # Check each required model
        missing_models = []
        for model_name, url in MODEL_URLS.items():
            model_path = os.path.join(weights_folder, model_name)
            if not os.path.exists(model_path):
                missing_models.append((model_name, url, model_path))
        
        if not missing_models:
            self.status_var.set("✓ All required models found. Ready to process.")
            self.model_notice.config(
                text="✓ All required models are available. You can select files to process.",
                foreground=THEME["success"]
            )
            self.enable_interface()
        else:
            self.status_var.set(f"⏳ Downloading {len(missing_models)} missing models...")
            self.progress_bar.pack(side=tk.RIGHT, padx=10)
            
            # Download missing models sequentially
            def download_next_model(index=0):
                if index >= len(missing_models):
                    self.status_var.set("✓ All models downloaded. Ready to process.")
                    self.model_notice.config(
                        text="✓ All required models are available. You can select files to process.",
                        foreground=THEME["success"]
                    )
                    self.progress_bar.pack_forget()
                    self.enable_interface()
                    return
                
                model_name, url, model_path = missing_models[index]
                self.progress_bar.update_status(f"⏳ Downloading {model_name}...")
                
                download_thread = DownloadThread(
                    url, 
                    model_path,
                    progress_callback=self.update_download_progress,
                    completion_callback=lambda _, __: download_next_model(index + 1)
                )
                download_thread.start()
            
            download_next_model()
    
    def update_download_progress(self, filename, progress, error=None):
        """Update download progress"""
        if error:
            self.progress_bar.update_status(f"❌ Error downloading {filename}: {error}")
            return
            
        if progress >= 0:
            self.progress_bar["value"] = progress
            self.progress_bar.update_status(f"⏳ Downloading {filename}: {progress}%")
    
    def enable_interface(self):
        """Enable interface elements after models are available"""
        self.video_button.state(['!disabled'])
        self.features_button.state(['!disabled'])
        self.output_button.state(['!disabled'])
    
    def select_video_file(self):
        """Select a video or image file based on the current mode"""
        mode = self.process_mode_var.get()
        
        if mode == "Track":
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
            )
            if file_path:
                self.video_path_var.set(file_path)
                # Load video preview (first frame)
                self.load_video_preview(file_path)
                self.check_processing_ready()
        else:  # Detect mode
            file_path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
            )
            if file_path:
                self.video_path_var.set(file_path)
                # Load image
                self.load_image(file_path)
                self.check_processing_ready()
    
    def load_video_preview(self, video_path):
        """Load first frame of video for preview"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            # Read first frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Failed to read first frame")
            
            # Get video info
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            self.video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Close video
            cap.release()
            
            # Convert frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Store the frame
            self.current_source_image = rgb_frame
            
            # Display in source canvas
            self.display_image(self.source_canvas, rgb_frame)
            
            # Hide placeholder
            self.source_placeholder.place_forget()
            
            # Clear result canvas
            self.result_canvas.delete("all")
            self.result_placeholder.place(relx=0.5, rely=0.5, anchor="center")
            self.current_result_image = None
            
            # Update status
            duration = self.video_frame_count / self.video_fps if self.video_fps > 0 else 0
            self.status_var.set(f"✓ Video loaded: {self.video_frame_count} frames, {duration:.2f} seconds")
            
        except Exception as e:
            messagebox.showerror("Video Loading Error", f"Failed to load video: {str(e)}")
            self.status_var.set(f"❌ Error loading video: {str(e)}")

    def select_features_folder(self):
        """Select a features folder"""
        folder_path = filedialog.askdirectory(
            title="Select Features Folder"
        )
        if folder_path:
            self.features_folder_var.set(folder_path)
            self.check_processing_ready()
    
    def select_output_folder(self):
        """Select an output folder"""
        folder_path = filedialog.askdirectory(
            title="Select Output Folder"
        )
        if folder_path:
            self.output_path_var.set(folder_path)
            self.check_processing_ready()
    
    def check_processing_ready(self):
        """Check if all required inputs are set to enable processing"""
        if (self.video_path_var.get() and 
            self.features_folder_var.get() and
            self.output_path_var.get()):
            self.start_button.state(['!disabled'])
            self.status_var.set("✓ All files selected. Ready to start processing.")
        else:
            self.start_button.state(['disabled'])
            
            # Provide informative status message about what's missing
            missing = []
            if not self.video_path_var.get():
                missing.append("video file")
            if not self.features_folder_var.get():
                missing.append("features folder")
            if not self.output_path_var.get():
                missing.append("output folder")
                
            if missing:
                missing_str = ", ".join(missing)
                self.status_var.set(f"⚠️ Please select the following: {missing_str}")
    
    def start_processing(self):
        """Start the processing with the selected files"""
        # Get paths
        source_path = self.video_path_var.get()
        features_folder = self.features_folder_var.get()
        output_dir = self.output_path_var.get()
        
        # Construct features file path using the folder path
        features_path = os.path.join(features_folder, "mosaic_example_features.pkl")
        
        # Get the configuration values
        process_mode = self.process_mode_var.get()
        mosaic_mode = self.mosaic_mode_var.get().lower()
        similarity_threshold = self.similarity_threshold_var.get()
        
        # Prepare output path based on process mode
        video_name = os.path.splitext(os.path.basename(source_path))[0]
        if process_mode == "Track":
            output_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
        else:  # Detect mode
            output_path = os.path.join(output_dir, f"{video_name}_det_seg_out.jpg")
        
        # Get weights paths
        weights_folder = os.path.join(os.getcwd(), "weights")
        yoloe_model = os.path.join(weights_folder, "yoloe-11l-seg-pf.pt")
        sam_model = os.path.join(weights_folder, "FastSAM-s.pt")
        
        # Update UI
        self.start_button.state(['disabled'])
        self.video_button.state(['disabled'])
        self.features_button.state(['disabled'])
        self.output_button.state(['disabled'])
        self.status_var.set(f"⏳ {process_mode}ing in progress...")
        
        # Update configs
        if process_mode == "Track":
            # Update MOT config
            self.mot_config["mosaic_mode"] = mosaic_mode
            self.mot_config["similarity_threshold"] = similarity_threshold
            self.mot_config["embedding_similarity_threshold"] = similarity_threshold
            config = self.mot_config
        else:  # Detect
            # Update DetSeg config
            self.det_seg_config["mosaic_mode"] = mosaic_mode
            self.det_seg_config["similarity_threshold"] = similarity_threshold
            config = self.det_seg_config
        
        # Start processing in a separate thread to keep UI responsive
        def processing_thread():
            try:
                if process_mode == "Track":
                    # Call MOT function
                    self.status_var.set(f"⏳ Tracking started... (this may take several minutes)")
                    
                    # Record start time
                    start_time = time.time()
                    
                    # Create debug frames directory if needed
                    debug_frames_dir = os.path.join(output_dir, "debug_frames")
                    os.makedirs(debug_frames_dir, exist_ok=True)
                    
                    # Run mosaic_mot
                    mosaic_mot(
                        bytetrack_cfg_path="bytetrack.yaml",
                        video_path=source_path,
                        output_path=output_path,
                        yoloe_model=yoloe_model,
                        sam_model=sam_model,
                        examples_root=features_folder,
                        cache_file=features_path,
                        dinov2_model="facebook/dinov2-base",
                        config=config,
                        save_annotated_frame_path=debug_frames_dir
                    )
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # When done, update UI and show result from the saved file
                    self.after(0, lambda: self.load_video_result(output_path, processing_time))
                    
                else:  # Detect
                    # Record start time
                    start_time = time.time()
                    
                    # Run MosaicDetSeg
                    self.status_var.set(f"⏳ Detection started...")
                    
                    # Load the image
                    image = cv2.imread(source_path)
                    if image is None:
                        raise ValueError(f"Failed to load image: {source_path}")
                    
                    # Convert to RGB for processing
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Initialize detector
                    detector = MosaicDetSeg(
                        yoloe_model=yoloe_model,
                        sam_model=sam_model,
                        examples_root=features_folder,
                        cache_file=features_path,
                        dinov2_model="facebook/dinov2-base",
                        config=config
                    )
                    
                    # Detect and segment
                    detection_results = detector.detect_and_segment(image_rgb)
                    
                    # Visualize results and save to disk
                    result_image = image_rgb.copy()
                    detector.visualize_results(result_image, detection_results, output_path)
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Display the result loaded from the saved file
                    self.after(0, lambda: self.display_detection_result(result_image, detection_results, processing_time))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                error = e or "An unknown error occurred"
                self.after(0, lambda: self.processing_error(error, process_mode))
        
        # Start processing thread
        threading.Thread(target=processing_thread, daemon=True).start()
    
    def display_detection_result(self, result_image, detection_results, processing_time):
        """Display detection results"""
        try:
            # Load the saved result image from disk instead of using the in-memory result
            output_dir = self.output_path_var.get()
            video_name = os.path.splitext(os.path.basename(self.video_path_var.get()))[0]
            output_path = os.path.join(output_dir, f"{video_name}_det_seg_out.jpg")
            
            # Check if the file exists
            if not os.path.exists(output_path):
                raise ValueError(f"Result file not found: {output_path}")
                
            # Load the image
            result_img = cv2.imread(output_path)
            if result_img is None:
                raise ValueError(f"Failed to load result image: {output_path}")
                
            # Convert to RGB
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            # Display the result image
            self.display_result_image(result_img_rgb)
            
            # Update status with detection information
            num_detections = len(detection_results) if detection_results else 0
            self.status_var.set(f"✅ Detection complete. Found {num_detections} object(s). (processing time: {processing_time:.2f}s)")
            
            # Re-enable UI
            self.start_button.state(['!disabled'])
            self.video_button.state(['!disabled'])
            self.features_button.state(['!disabled'])
            self.output_button.state(['!disabled'])
            
            # Show completion message
            messagebox.showinfo("Detection Complete", 
                              f"Detection complete.\nFound {num_detections} object(s).\nProcessing time: {processing_time:.2f} seconds\nOutput saved to: {output_path}")
                              
        except Exception as e:
            self.processing_error(e, "Detect")
    
    def load_video_result(self, output_path, processing_time):
        """Load and display the result video"""
        try:
            # Check if the file exists before trying to load it
            if not os.path.exists(output_path):
                raise ValueError(f"Result video file not found: {output_path}")
                
            # Open video
            cap = cv2.VideoCapture(output_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open result video: {output_path}")
            
            # Read first frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Failed to read first frame of result video")
            
            # Close video
            cap.release()
            
            # Convert frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display in result canvas
            self.display_result_image(rgb_frame)
            
            # Update status
            self.status_var.set(f"✅ Tracking complete. Output saved to {output_path} (processing time: {processing_time:.2f}s)")
            
            # Re-enable UI
            self.start_button.state(['!disabled'])
            self.video_button.state(['!disabled'])
            self.features_button.state(['!disabled'])
            self.output_button.state(['!disabled'])
            
            # Show completion message
            messagebox.showinfo("Tracking Complete", 
                              f"Tracking complete.\nOutput saved to: {output_path}\nProcessing time: {processing_time:.2f} seconds")
        
        except Exception as e:
            self.processing_error(e, "Track")
    
    def processing_error(self, error, mode):
        """Handle processing error"""
        self.status_var.set(f"❌ Error during {mode.lower()}ing: {str(error)}")
        
        # Re-enable UI
        self.start_button.state(['!disabled'])
        self.video_button.state(['!disabled'])
        self.features_button.state(['!disabled'])
        self.output_button.state(['!disabled'])
        
        # Show error message
        messagebox.showerror(f"{mode} Error", 
                          f"An error occurred during {mode.lower()}ing:\n{str(error)}")

    def update_threshold_label(self, *args):
        """Update the threshold label when slider changes"""
        value = self.similarity_threshold_var.get()
        self.threshold_label.config(text=f"{value:.2f}")

    def on_process_mode_change(self, event):
        """Handle changes to the process mode (track/detect)"""
        # Clear displays when switching modes
        self.clear_displays()
        
        mode = self.process_mode_var.get()
        if mode == "Track":
            self.video_button.config(text="Select Video")
        else:
            self.video_button.config(text="Select Image")
    
    def clear_displays(self):
        """Clear both source and result displays"""
        # Clear canvases
        self.source_canvas.delete("all")
        self.result_canvas.delete("all")
        
        # Show placeholders
        self.source_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self.result_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        
        # Reset stored images/videos
        self.current_source_image = None
        self.current_result_image = None
        
        # Stop video playback if any
        if hasattr(self, 'video_playing') and self.video_playing:
            self.video_playing = False
    
    def load_image(self, path):
        """Load an image from path and display in source canvas"""
        try:
            # Load with OpenCV for processing compatibility
            cv_image = cv2.imread(path)
            if cv_image is None:
                raise ValueError(f"Failed to load image: {path}")
            
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Store the image for processing
            self.current_source_image = rgb_image
            
            # Display in source canvas
            self.display_image(self.source_canvas, rgb_image)
            
            # Hide placeholder
            self.source_placeholder.place_forget()
            
            return rgb_image
        except Exception as e:
            messagebox.showerror("Image Loading Error", f"Failed to load image: {str(e)}")
            return None
    
    def display_image(self, canvas, image, preserve_aspect=True):
        """Display an image on the specified canvas"""
        if image is None:
            return
            
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not fully initialized yet
            canvas.after(100, lambda: self.display_image(canvas, image, preserve_aspect))
            return
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Calculate scaling
        if preserve_aspect:
            # Calculate scale to fit while preserving aspect ratio
            scale_width = canvas_width / img_width
            scale_height = canvas_height / img_height
            scale = min(scale_width, scale_height)
            
            # Calculate new dimensions
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
        else:
            # Stretch to fill
            new_width = canvas_width
            new_height = canvas_height
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to PhotoImage
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        # Store reference to prevent garbage collection
        if canvas == self.source_canvas:
            self.source_tk_img = tk_img
        else:
            self.result_tk_img = tk_img
        
        # Clear canvas and display image
        canvas.delete("all")
        
        # Center the image
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        
        canvas.create_image(x_offset, y_offset, image=tk_img, anchor="nw")
    
    def display_result_image(self, image):
        """Display result image on the result canvas"""
        if image is None:
            return
            
        # Store the result image
        self.current_result_image = image
        
        # Display in result canvas
        self.display_image(self.result_canvas, image)
        
        # Hide placeholder
        self.result_placeholder.place_forget()

    def on_preview_resize(self, event):
        """Handle resize events for the preview area"""
        # Update source and result displays if they exist
        if hasattr(self, 'current_source_image') and self.current_source_image is not None:
            self.display_image(self.source_canvas, self.current_source_image, preserve_aspect=True)
        
        if hasattr(self, 'current_result_image') and self.current_result_image is not None:
            self.display_image(self.result_canvas, self.current_result_image, preserve_aspect=True)
        

if __name__ == "__main__":
    app = MosaicInterface()
    app.mainloop()
