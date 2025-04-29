import os
import cv2
import torch
import numpy as np
from ultralytics import SAM
from PIL import Image, ImageTk
import urllib.request
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import Scale as TkScale


class EnhancedLabelingTool:
    def __init__(
        self,
        screen_width: int,
        screen_height: int, 
        sam_model_path: str = None,
        image_dir: str = None,
        output_dir: str = None
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Set default model path and ensure weights directory exists
        self.weights_dir = os.path.join(os.getcwd(), "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        self.sam_model_path = sam_model_path if sam_model_path else os.path.join(self.weights_dir, "sam_l.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Image and annotation data
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.image_files = []
        self.current_image_index = 0
        self.current_image = None
        self.current_image_path = None
        self.display_image = None
        self.display_scale = 1.0

        # Initialize annotations
        self.masks = []
        self.bboxes = []
        self.labels = []

        # Initialize UI state
        self.is_drawing = False
        self.start_x, self.start_y = 0, 0
        self.current_bbox = None
        self.dragging_mask_index = None
        self.tool_mode = "select"
        self.brush_size = 10
        self.eraser_size = 10
        self.selected_mask_index = None
        self.mask_overlay_alpha = 0.5
        self.scrollbar_width = 30
        self.scrollbar_height = 30
        self.default_canvas_width = 10
        self.default_canvas_height = 10
        self.control_panel_width = 300
        self.control_panel_height = 50

        # Create main window
        self.root = tk.Tk()
        self.root.title("Enhanced Image Labeling Tool")
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")

        # Create Accent.TButton style (without keyword arguments)
        style = ttk.Style()
        style.configure("Accent.TButton", background="#4CAF50", foreground="black", font=("Arial", 12))

        # If paths were not provided, show the folder selection dialog
        if not self.image_dir or not self.output_dir:
            self.show_path_selection_dialog()
        else:
            self.setup_main_interface()

    def check_sam_model(self):
        """
        Check if the SAM model exists and download it if it doesn't
        """
        if not os.path.exists(self.sam_model_path):
            model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_l.pt"
            self.download_model(model_url, self.sam_model_path)
    
    def download_model(self, url, destination):
        """
        Download the model from the URL to the specified destination
        """
        # Clear any existing content from the selection window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create a frame for the download display
        download_frame = ttk.Frame(self.root, padding=20)
        download_frame.pack(expand=True)

        # Add a title
        ttk.Label(
            download_frame, 
            text="Downloading SAM Model", 
            font=("Arial", 16, "bold")
        ).pack(pady=20)

        # Create progress information
        status_label = ttk.Label(
            download_frame, 
            text="Downloading SAM model for auto-annotation. This may take a few minutes...",
            font=("Arial", 12)
        )
        status_label.pack(pady=20)

        progress_var = tk.DoubleVar()
        progress = ttk.Progressbar(
            download_frame, 
            variable=progress_var, 
            maximum=100,
            length=350
        )
        progress.pack(pady=10, padx=20)
        
        progress_label = ttk.Label(download_frame, text="0%")
        progress_label.pack(pady=5)

        # Force update of the window
        self.root.update()

        try:
            # Define a callback function to update the progress
            def update_progress(count, block_size, total_size):
                if total_size > 0:
                    percent = min(100, count * block_size * 100 / total_size)
                    progress_var.set(percent)
                    progress_label.config(text=f"{percent:.1f}%")
                    self.root.update()

            print(f"Downloading SAM model from {url} to {destination}...")
            
            # Download the file with progress callback
            urllib.request.urlretrieve(url, destination, reporthook=update_progress)
            
            print("Download complete!")
            
            # Update labels to show success
            status_label.config(text="Download complete! SAM model has been downloaded successfully.")
            progress_label.config(text="100%")
            
            # Add continue button
            ttk.Button(
                download_frame,
                text="Continue",
                command=self.show_path_selection_dialog,
                style="Accent.TButton"
            ).pack(pady=20)
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            
            # Update labels to show error
            status_label.config(text=f"Failed to download the SAM model: {e}")
            progress_label.config(text="Error")
            
            # Add retry button
            ttk.Button(
                download_frame,
                text="Retry",
                command=lambda: self.download_model(url, destination),
                style="Accent.TButton"
            ).pack(pady=10)
            
            # Add continue anyway button
            ttk.Button(
                download_frame,
                text="Continue Anyway",
                command=self.show_path_selection_dialog
            ).pack(pady=10)

    def show_path_selection_dialog(self):
        """
        Show a dialog to select image and output directories
        """
        # Clear the current window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create a frame for the path selection
        path_frame = ttk.Frame(self.root, padding=20)
        path_frame.pack(expand=True)

        # Add a title
        ttk.Label(
            path_frame, 
            text="Select Directories", 
            font=("Arial", 16, "bold")
        ).pack(pady=20)

        # Image directory selection
        img_frame = ttk.Frame(path_frame)
        img_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(img_frame, text="Image Directory:").pack(side=tk.LEFT, padx=5)
        
        self.img_dir_var = tk.StringVar(value=self.image_dir if self.image_dir else "")
        img_entry = ttk.Entry(img_frame, textvariable=self.img_dir_var, width=50)
        img_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            img_frame, 
            text="Browse...", 
            command=self.select_image_dir
        ).pack(side=tk.LEFT, padx=5)

        # Output directory selection
        out_frame = ttk.Frame(path_frame)
        out_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(out_frame, text="Output Directory:").pack(side=tk.LEFT, padx=5)
        
        self.out_dir_var = tk.StringVar(value=self.output_dir if self.output_dir else "")
        out_entry = ttk.Entry(out_frame, textvariable=self.out_dir_var, width=50)
        out_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            out_frame, 
            text="Browse...", 
            command=self.select_output_dir
        ).pack(side=tk.LEFT, padx=5)

        # Model status
        if os.path.exists(self.sam_model_path):
            model_status = "SAM model found"
            status_color = "#4CAF50"  # Green
        else:
            model_status = "SAM model missing - will be downloaded automatically"
            status_color = "#FFA500"  # Orange
            
        model_frame = ttk.Frame(path_frame)
        model_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(model_frame, text="Model Status:").pack(side=tk.LEFT, padx=5)
        status_label = ttk.Label(model_frame, text=model_status, foreground=status_color)
        status_label.pack(side=tk.LEFT, padx=5)

        # Start button
        ttk.Button(
            path_frame,
            text="Start Labeling",
            command=self.confirm_paths,
            style="Accent.TButton"
        ).pack(pady=20)

    def select_image_dir(self):
        """
        Open a dialog to select the image directory
        """
        directory = filedialog.askdirectory(
            title="Select Image Directory",
            initialdir=self.image_dir if self.image_dir else os.getcwd()
        )
        if directory:
            self.img_dir_var.set(directory)

    def select_output_dir(self):
        """
        Open a dialog to select the output directory
        """
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir if self.output_dir else os.getcwd()
        )
        if directory:
            self.out_dir_var.set(directory)

    def confirm_paths(self):
        """
        Confirm the selected paths and proceed to the main interface
        """
        self.image_dir = self.img_dir_var.get()
        self.output_dir = self.out_dir_var.get()
        
        # Create directories if they don't exist
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check if SAM model exists and download if needed
        if not os.path.exists(self.sam_model_path):
            self.check_sam_model()
        else:
            # Setup the main interface
            self.setup_main_interface()

    def setup_main_interface(self):
        """
        Setup the main labeling interface
        """
        # Clear the current window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Load SAM model
        try:
            print(f"Loading SAM model from {self.sam_model_path} on {self.device}...")
            self.sam = SAM(self.sam_model_path).to(self.device)
            print("SAM model loaded.")
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            self.sam = None
            
            # Show error in the main window instead of messagebox
            error_frame = ttk.Frame(self.root, padding=20)
            error_frame.pack(expand=True)
            
            ttk.Label(
                error_frame, 
                text="Error Loading SAM Model", 
                font=("Arial", 16, "bold"),
                foreground="red"
            ).pack(pady=10)
            
            ttk.Label(
                error_frame,
                text=f"Failed to load the SAM model: {e}\n\nThe application will continue, but segmentation features will be limited.",
                wraplength=600
            ).pack(pady=10)
            
            ttk.Button(
                error_frame,
                text="Continue Anyway",
                command=lambda: self.continue_setup_ui(),
                style="Accent.TButton"
            ).pack(pady=20)
            
            return
            
        # Continue with UI setup
        self.continue_setup_ui()
    
    def continue_setup_ui(self):
        """Continue setting up the UI after model loading (success or failure)"""
        # Setup UI
        self.setup_ui()

        # Load image files
        self.load_image_files()

        # Start with first image if available
        if self.image_files:
            self.load_image(0)
        else:
            print("No images found in the specified directory. Please add images to continue.")
            # Display a message on the canvas
            self.canvas.create_text(
                self.screen_width // 2 - 125, self.screen_height // 2 - 50,
                text="No images found in the directory.\nPlease add images or change another directory.",
                fill="white",
                font=("Arial", 16)
            )

    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left panel for controls
        self.control_panel = ttk.Frame(main_frame, width=250)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Create right panel for image
        self.image_panel = ttk.Frame(main_frame)
        self.image_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Set up control panel widgets
        self.setup_control_panel()

        # Set up image canvas
        self.canvas = tk.Canvas(self.image_panel, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind canvas events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Add scrollbars
        h_scrollbar = ttk.Scrollbar(self.image_panel, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        v_scrollbar = ttk.Scrollbar(self.image_panel, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

    def setup_control_panel(self):
        # Current paths display
        paths_frame = ttk.LabelFrame(self.control_panel, text="Directories")
        paths_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(paths_frame, text=f"Images: {os.path.basename(self.image_dir)}").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(paths_frame, text=f"Output: {os.path.basename(self.output_dir)}").pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Button(
            paths_frame,
            text="Change Directories",
            command=self.show_path_selection_dialog
        ).pack(fill=tk.X, padx=5, pady=5)

        # Tool selection
        tool_frame = ttk.LabelFrame(self.control_panel, text="Tools")
        tool_frame.pack(fill=tk.X, padx=5, pady=5)

        self.tool_var = tk.StringVar(value="select")
        tools = [
            ("Select", "select"),
            ("Bounding Box", "bbox"),
            ("Brush", "brush"),
            ("Eraser", "eraser")
        ]

        for text, value in tools:
            ttk.Radiobutton(
                tool_frame, 
                text=text, 
                value=value, 
                variable=self.tool_var,
                command=self.on_tool_change
            ).pack(anchor=tk.W, padx=5, pady=2)

        # Brush size control
        brush_frame = ttk.LabelFrame(self.control_panel, text="Brush Size")
        brush_frame.pack(fill=tk.X, padx=5, pady=5)

        self.brush_slider = TkScale(
            brush_frame,
            from_=1,
            to=50,
            orient=tk.HORIZONTAL,
            command=self.on_brush_size_change
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(fill=tk.X, padx=5, pady=5)

        # Eraser size control
        eraser_frame = ttk.LabelFrame(self.control_panel, text="Eraser Size")
        eraser_frame.pack(fill=tk.X, padx=5, pady=5)

        self.eraser_slider = TkScale(
            eraser_frame,
            from_=1,
            to=50,
            orient=tk.HORIZONTAL,
            command=self.on_eraser_size_change
        )
        self.eraser_slider.set(self.eraser_size)
        self.eraser_slider.pack(fill=tk.X, padx=5, pady=5)

        # Class label entry
        label_frame = ttk.LabelFrame(self.control_panel, text="Class Label")
        label_frame.pack(fill=tk.X, padx=5, pady=5)

        self.label_entry = ttk.Entry(label_frame)
        self.label_entry.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            label_frame,
            text="Apply Label",
            command=self.apply_label
        ).pack(fill=tk.X, padx=5, pady=5)

        # Navigation controls
        nav_frame = ttk.LabelFrame(self.control_panel, text="Navigation")
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        nav_buttons = ttk.Frame(nav_frame)
        nav_buttons.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            nav_buttons,
            text="Previous",
            command=self.previous_image
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            nav_buttons,
            text="Next",
            command=self.next_image
        ).pack(side=tk.RIGHT, padx=5)

        # Image info
        self.image_info = ttk.Label(nav_frame, text="Image: 0/0")
        self.image_info.pack(fill=tk.X, padx=5, pady=5)

        # Status message frame - for save operations
        self.status_frame = ttk.Frame(self.control_panel)
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_message = ttk.Label(self.status_frame, text="", wraplength=230)
        self.status_message.pack(fill=tk.X, padx=5, pady=5)

        # Save and quit buttons
        action_frame = ttk.Frame(self.control_panel)
        action_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            action_frame,
            text="Save Objects",
            command=self.save_objects
        ).pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            action_frame,
            text="Quit",
            command=self.quit
        ).pack(fill=tk.X, padx=5, pady=5)

    def on_tool_change(self):
        self.tool_mode = self.tool_var.get()

    def on_brush_size_change(self, value):
        self.brush_size = int(float(value))

    def on_eraser_size_change(self, value):
        self.eraser_size = int(float(value))

    def apply_label(self):
        # Apply the current label to the selected mask
        if hasattr(self, 'selected_mask_index') and self.selected_mask_index is not None:
            if 0 <= self.selected_mask_index < len(self.labels):
                self.labels[self.selected_mask_index] = self.label_entry.get()
                self.update_display()

    def load_image_files(self):
        """
        Load all image files from the specified directory
        """

        try:
            self.image_files = [
                f for f in os.listdir(self.image_dir) 
                if f.lower().endswith(('png', 'jpg', 'jpeg'))
            ]
            self.image_files.sort()

            if self.image_files:
                self.image_info.config(
                    text=f"Image: 1/{len(self.image_files)} - {self.image_files[0]}"
                )
            else:
                print(f"No image files found in {self.image_dir}")
                self.image_info.config(text="No images found")
        except Exception as e:
            print(f"Error loading images from {self.image_dir}: {e}")
            self.image_info.config(text=f"Error: {str(e)}")

    def load_image(self, index):
        """
        Load image at the specified index
        """

        if 0 <= index < len(self.image_files):
            # Save current annotations if we have an image loaded
            if self.current_image is not None and self.masks:
                self.save_objects()

            # Clear current annotations
            self.masks = []
            self.bboxes = []
            self.labels = []
            self.selected_mask_index = None

            # Load new image
            self.current_image_index = index
            self.current_image_path = os.path.join(self.image_dir, self.image_files[index])

            # Read the image
            self.current_image = cv2.imread(self.current_image_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

            # Update info
            self.image_info.config(
                text=f"Image: {index+1}/{len(self.image_files)} - {self.image_files[index]}"
            )

            # Calculate display scale to fit image within the canvas while maintaining aspect ratio
            h, w, _ = self.current_image.shape
            # Get canvas size (accounting for padding and scrollbars)
            canvas_width = self.image_panel.winfo_width() - self.scrollbar_width
            canvas_height = self.image_panel.winfo_height() - self.scrollbar_height

            # If canvas hasn't been drawn yet, use reasonable defaults
            if canvas_width <= self.default_canvas_width or canvas_height <= self.default_canvas_height:
                canvas_width = self.screen_width - self.control_panel_width
                canvas_height = self.screen_height - self.control_panel_height

            # Calculate scale to fit within canvas
            self.display_scale = min(canvas_width / w, canvas_height / h)

            # Set up proper scroll region for the original image size
            self.canvas.config(scrollregion=(0, 0, w, h))

            # Update display
            self.update_display()
            
    def next_image(self):
        """
        Go to next image
        """
    
        if self.current_image_index < len(self.image_files) - 1:
            self.load_image(self.current_image_index + 1)

    def previous_image(self):
        """
        Go to previous image
        """

        if self.current_image_index > 0:
            self.load_image(self.current_image_index - 1)
    
    def update_display(self):
        """
        Update the canvas with the current image and annotations
        """

        if self.current_image is None:
            return

        # Create a copy of the current image for display
        display_image = self.current_image.copy()

        # Draw masks and bounding boxes
        for i, (mask, bbox, label) in enumerate(zip(self.masks, self.bboxes, self.labels)):
            color = (0, 255, 0) if i != self.selected_mask_index else (0, 200, 255)
            mask_rgb = np.zeros_like(display_image)
            mask_rgb[mask] = color
            display_image = cv2.addWeighted(display_image, 1.0, mask_rgb, self.mask_overlay_alpha, 0)

            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if label:
                font_scale = 0.7
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                text_bg_y1 = max(0, y1 - text_size[1] - 10)
                text_bg_y2 = y1
                cv2.rectangle(
                    display_image, 
                    (x1, text_bg_y1), 
                    (x1 + text_size[0] + 10, text_bg_y2), 
                    color, 
                    -1
                )
                cv2.putText(
                    display_image, 
                    label, 
                    (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (0, 0, 0), 
                    2
                )

        # If currently drawing a bounding box, draw it
        if self.is_drawing and self.tool_mode == "bbox" and self.current_bbox:
            x1, y1, x2, y2 = self.current_bbox
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Store the unresized display image for saving
        self.display_image = Image.fromarray(display_image)

        # Resize image to fit the canvas while maintaining aspect ratio
        h, w, _ = display_image.shape
        display_w, display_h = int(w * self.display_scale), int(h * self.display_scale)

        # Resize only if the image is larger than the canvas
        if self.display_scale < 1.0:
            display_resized = cv2.resize(display_image, (display_w, display_h), interpolation=cv2.INTER_AREA)
            display_pil = Image.fromarray(display_resized)
        else:
            # Use original size if image is smaller than canvas
            display_pil = self.display_image

        # Convert PIL image to Tkinter PhotoImage
        self.photo_image = ImageTk.PhotoImage(image=display_pil)

        # Clear canvas and draw image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        # Update scroll region to full image size
        self.canvas.config(scrollregion=(0, 0, w, h))

        # Draw brush cursor if in brush or eraser mode
        if self.tool_mode in ["brush", "eraser"] and hasattr(self, 'cursor_x') and hasattr(self, 'cursor_y'):
            # Get the real size based on brush/eraser size
            size = self.brush_size if self.tool_mode == "brush" else self.eraser_size

            # Scale the size to match the display scale
            display_size = max(1, int(size * self.display_scale))

            # Use the cursor position directly (already in display coordinates)
            color = "green" if self.tool_mode == "brush" else "red"
            self.canvas.create_oval(
                self.cursor_x - display_size, self.cursor_y - display_size,
                self.cursor_x + display_size, self.cursor_y + display_size,
                outline=color, width=2
            )

    def _display_to_image_coords(self, display_x, display_y):
        """
        Convert display coordinates to original image coordinates
        """

        # Get the canvas scroll position
        canvas_x = self.canvas.canvasx(display_x)
        canvas_y = self.canvas.canvasy(display_y)

        # If displaying a scaled image, convert back to original image coordinates
        if self.display_scale != 1.0:
            image_x = int(canvas_x / self.display_scale)
            image_y = int(canvas_y / self.display_scale)
        else:
            image_x = int(canvas_x)
            image_y = int(canvas_y)
            
        return image_x, image_y

    def on_mouse_down(self, event):
        """
        Handle mouse button press event
        """

        self.is_drawing = True
        self.start_x, self.start_y = self._display_to_image_coords(event.x, event.y)

        if self.tool_mode == "select":
            # Check if clicked on an existing mask
            for i, mask in enumerate(self.masks):
                x1, y1, x2, y2 = self.bboxes[i]
                image_x, image_y = self.start_x, self.start_y

                if x1 <= image_x <= x2 and y1 <= image_y <= y2:
                    # Check if the point is actually inside the mask
                    try:
                        if image_y < mask.shape[0] and image_x < mask.shape[1] and mask[image_y, image_x]:
                            self.selected_mask_index = i
                            self.label_entry.delete(0, tk.END)
                            self.label_entry.insert(0, self.labels[i])
                            self.update_display()
                            return
                    except IndexError:
                        # Skip if coordinates are out of bounds
                        pass

            # No mask was clicked
            self.selected_mask_index = None
            self.update_display()

        elif self.tool_mode == "bbox":
            self.current_bbox = [self.start_x, self.start_y, self.start_x, self.start_y]

        elif self.tool_mode in ["brush", "eraser"]:
            self.apply_brush(self.start_x, self.start_y)

    def on_mouse_move(self, event):
        """
        Handle mouse movement event
        """

        image_x, image_y = self._display_to_image_coords(event.x, event.y)
        self.cursor_x, self.cursor_y = event.x, event.y

        if self.is_drawing:
            if self.tool_mode == "bbox":
                self.current_bbox = [self.start_x, self.start_y, image_x, image_y]
                self.update_display()

            elif self.tool_mode in ["brush", "eraser"]:
                self.apply_brush(image_x, image_y)

        if self.tool_mode in ["brush", "eraser"]:
            self.update_display()

    def on_mouse_up(self, event):
        """
        Handle mouse button release event
        """

        image_x, image_y = self._display_to_image_coords(event.x, event.y)
        self.is_drawing = False

        if self.tool_mode == "bbox" and self.current_bbox:
            x1, y1, x2, y2 = self.current_bbox

            # Normalize bbox (ensure x1 < x2, y1 < y2)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Ensure bbox has some area
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                # Get mask from SAM
                mask = self.get_mask_from_bbox([x1, y1, x2, y2])

                # Add to lists
                self.masks.append(mask)
                self.bboxes.append([x1, y1, x2, y2])
                self.labels.append("")

                # Select the new mask
                self.selected_mask_index = len(self.masks) - 1
                self.label_entry.delete(0, tk.END)

            self.current_bbox = None
            self.update_display()

    def apply_brush(self, x, y):
        """
        Apply brush or eraser at the specified location
        """

        # Check if we're within image bounds
        if self.current_image is None:
            return

        h, w, _ = self.current_image.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return

        if self.selected_mask_index is None and self.tool_mode == "brush":
            # Create a new mask if we're using the brush and no mask is selected
            mask = np.zeros((h, w), dtype=bool)
            self.masks.append(mask)
            self.bboxes.append([x, y, x, y])
            self.labels.append("")
            self.selected_mask_index = len(self.masks) - 1

        if self.selected_mask_index is not None:
            size = self.brush_size if self.tool_mode == "brush" else self.eraser_size
            y1, x1 = max(0, y - size), max(0, x - size)
            y2, x2 = min(h, y + size), min(w, x + size)

            # Create circular brush mask
            brush_mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
            for i in range(brush_mask.shape[0]):
                for j in range(brush_mask.shape[1]):
                    if (i - (y - y1))**2 + (j - (x - x1))**2 <= size**2:
                        brush_mask[i, j] = True

            try:
                if self.tool_mode == "brush":
                    # Apply brush
                    self.masks[self.selected_mask_index][y1:y2, x1:x2] |= brush_mask

                    # Update bounding box
                    mask = self.masks[self.selected_mask_index]
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        min_x, max_x = np.min(x_indices), np.max(x_indices)
                        min_y, max_y = np.min(y_indices), np.max(y_indices)
                        self.bboxes[self.selected_mask_index] = [min_x, min_y, max_x, max_y]
                else:
                    # Apply eraser
                    self.masks[self.selected_mask_index][y1:y2, x1:x2] &= ~brush_mask

                    # Check if mask is empty
                    if not np.any(self.masks[self.selected_mask_index]):
                        # Remove this mask
                        self.masks.pop(self.selected_mask_index)
                        self.bboxes.pop(self.selected_mask_index)
                        self.labels.pop(self.selected_mask_index)
                        self.selected_mask_index = None
                    else:
                        # Update bounding box
                        mask = self.masks[self.selected_mask_index]
                        y_indices, x_indices = np.where(mask)
                        min_x, max_x = np.min(x_indices), np.max(x_indices)
                        min_y, max_y = np.min(y_indices), np.max(y_indices)
                        self.bboxes[self.selected_mask_index] = [min_x, min_y, max_x, max_y]

                self.update_display()
            except Exception as e:
                print(f"Error applying brush: {e}")

    def get_mask_from_bbox(self, bbox):
        """
        Get mask from SAM model using a bounding box
        """

        if self.sam is None:
            # Return an empty mask if SAM model is not loaded
            h, w, _ = self.current_image.shape
            return np.zeros((h, w), dtype=bool)

        try:
            x1, y1, x2, y2 = bbox
            results = self.sam.predict(
                self.current_image,
                bboxes=np.array([[x1, y1, x2, y2]])
            )

            # Convert mask to boolean numpy array
            mask = results[0].masks.data[0].cpu().numpy().astype(bool)
            return mask
        except Exception as e:
            print(f"Error getting mask from bbox: {e}")
            # Return an empty mask
            h, w, _ = self.current_image.shape
            return np.zeros((h, w), dtype=bool)

    def get_mask_from_point(self, point):
        """
        Get mask from SAM model using a point prompt
        """

        if self.sam is None:
            # Return an empty mask if SAM model is not loaded
            h, w, _ = self.current_image.shape
            return np.zeros((h, w), dtype=bool)

        try:
            x, y = point
            results = self.sam.predict(
                self.current_image,
                points=np.array([[x, y]]),
                labels=np.array([1])
            )

            # Convert mask to boolean numpy array
            mask = results[0].masks.data[0].cpu().numpy().astype(bool)
            return mask
        except Exception as e:
            print(f"Error getting mask from point: {e}")
            # Return an empty mask
            h, w, _ = self.current_image.shape
            return np.zeros((h, w), dtype=bool)

    def save_objects(self):
        """
        Save all labeled objects as individual images
        """

        if not self.current_image_path or not self.masks:
            self.status_message.config(text="No objects to save.", foreground="orange")
            return

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Get base filename
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]

        # Save each object
        saved_count = 0
        for i, (mask, label) in enumerate(zip(self.masks, self.labels)):
            # Skip if no label
            if not label:
                continue
  
            # Create label directory
            label_dir = os.path.join(self.output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            # Extract object using mask
            object_image = self.current_image.copy()

            # Make non-mask area transparent (black)
            object_image[~mask] = 0

            # Create filename
            filename = f"{base_name}_{i}.png"
            save_path = os.path.join(label_dir, filename)

            # Get bounding box
            y_indices, x_indices = np.where(mask)
            x1, y1 = np.min(x_indices), np.min(y_indices)
            x2, y2 = np.max(x_indices), np.max(y_indices)

            # Crop and save
            cropped = object_image[y1:y2+1, x1:x2+1]
            Image.fromarray(cropped).save(save_path)
            saved_count += 1

        # Also save annotated image - directly save the PIL Image without converting again
        annotated_path = os.path.join(self.output_dir, f"annotated_{base_name}.png")
        try:
            self.display_image.save(annotated_path)

            print(f"Saved {saved_count} objects from {self.current_image_path}")
            # Display message in the status frame instead of canvas
            self.status_message.config(
                text=f"Saved {saved_count} objects for the current image!",
                foreground="green"
            )
        except Exception as e:
            print(f"Error saving annotated image: {e}")
            # Display error message in the status frame
            self.status_message.config(
                text="Failed to save annotated image, please check the output directory!",
                foreground="red"
            )

    def quit(self):
        """
        Save current objects and quit
        """

        if self.current_image is not None and self.masks:
            self.save_objects()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """
        Run the application
        """

        self.root.mainloop()


if __name__ == "__main__":
    # Create and run labeler
    labeler = EnhancedLabelingTool(
        screen_width=1400,
        screen_height=900
    )
    labeler.run()
