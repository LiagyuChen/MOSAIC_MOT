import os
import sys
import cv2
import time
import pickle
import numpy as np
import requests
import threading
from pathlib import Path
from tqdm import tqdm
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                QLabel, QPushButton, QFileDialog, QSlider, QComboBox, 
                                QProgressBar, QTabWidget, QGroupBox, QRadioButton, QButtonGroup,
                                QScrollArea, QSplitter, QFrame, QSpacerItem, QSizePolicy)
    from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
    from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, QThread, QUrl
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
    from PyQt5.QtMultimediaWidgets import QVideoWidget
    print("Successfully imported PyQt5 modules")
except ImportError as e:
    print(f"Error importing PyQt5: {e}")
    print("Please install PyQt5 using: pip install PyQt5")
    sys.exit(1)

try:
    from mosaic_mot import mosaic_mot
    print("Successfully imported mosaic_mot")
except ImportError as e:
    print(f"Error importing mosaic_mot: {e}")
    sys.exit(1)

try:
    from mosaic_det_seg import MosaicDetSeg
    print("Successfully imported MosaicDetSeg")
except ImportError as e:
    print(f"Error importing MosaicDetSeg: {e}")
    sys.exit(1)

from mosaic_mot_interface import adjustable_configs

# Constants for UI
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #2D2D30;
    color: #FFFFFF;
}
QLabel {
    color: #FFFFFF;
}
QPushButton {
    background-color: #0078D7;
    color: white;
    border: none;
    padding: 5px 15px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #1C97EA;
}
QPushButton:pressed {
    background-color: #00559B;
}
QSlider::groove:horizontal {
    height: 8px;
    background: #555555;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #0078D7;
    width: 16px;
    margin: -4px 0;
    border-radius: 8px;
}
QComboBox {
    background-color: #3E3E42;
    color: white;
    border: 1px solid #555555;
    padding: 5px;
    border-radius: 3px;
}
QProgressBar {
    border: 1px solid #555555;
    border-radius: 5px;
    text-align: center;
    background-color: #3E3E42;
}
QProgressBar::chunk {
    background-color: #0078D7;
    border-radius: 5px;
}
QTabWidget::pane {
    border: 1px solid #555555;
    background-color: #2D2D30;
}
QTabBar::tab {
    background-color: #3E3E42;
    color: white;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #0078D7;
}
QGroupBox {
    border: 1px solid #555555;
    border-radius: 5px;
    margin-top: 10px;
    font-weight: bold;
    color: white;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
}
QRadioButton {
    color: white;
}
QScrollArea {
    border: none;
}
"""

# Model URLs
MODEL_URLS = {
    "yoloe-11l-seg-pf.pt": "https://github.com/fudan-zvg/SOFT/releases/download/v0.1.0/yoloe-11l-seg-pf.pt",
    "FastSAM-s.pt": "https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/FastSAM-s/FastSAM-s.pt"
}

class VideoPlayer(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Title
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(400, 300)
        
        # Media player
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        
        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.setFixedWidth(100)
        self.play_button.clicked.connect(self.toggle_play)
        
        # Timeline slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        
        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        
        # Add widgets to layout
        layout.addWidget(self.title_label)
        layout.addWidget(self.video_widget)
        
        # Timeline controls layout
        timeline_layout = QHBoxLayout()
        timeline_layout.addWidget(self.play_button)
        timeline_layout.addWidget(self.position_slider)
        timeline_layout.addWidget(self.time_label)
        
        layout.addLayout(timeline_layout)
        
        # Connect media player signals
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.stateChanged.connect(self.state_changed)
        
        self.setLayout(layout)
        
    def load_video(self, file_path):
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.play_button.setText("Play")
        
    def toggle_play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
            
    def set_position(self, position):
        self.media_player.setPosition(position)
        
    def position_changed(self, position):
        self.position_slider.setValue(position)
        
        # Update time display
        seconds = position // 1000
        minutes = seconds // 60
        seconds %= 60
        
        duration_seconds = self.media_player.duration() // 1000
        duration_minutes = duration_seconds // 60
        duration_seconds %= 60
        
        self.time_label.setText(f"{minutes:02d}:{seconds:02d} / {duration_minutes:02d}:{duration_seconds:02d}")
        
    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)
        
    def state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")
            
    def reset(self):
        self.media_player.stop()
        self.media_player.setMedia(QMediaContent())
        self.position_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")
        self.play_button.setText("Play")


class ImageViewer(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Title
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #1E1E1E;")
        
        # Placeholder text
        self.placeholder_text = QLabel("No image selected")
        self.placeholder_text.setAlignment(Qt.AlignCenter)
        self.placeholder_text.setStyleSheet("color: #999999; font-size: 16px;")
        
        # Add widgets to layout
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label)
        
        self.setLayout(layout)
        self.current_image = None
        
    def load_image(self, file_path):
        if not file_path or not os.path.exists(file_path):
            self.image_label.clear()
            self.placeholder_text.setVisible(True)
            return
            
        image = cv2.imread(file_path)
        if image is None:
            self.image_label.clear()
            self.placeholder_text.setVisible(True)
            return
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.display_image(image)
        
    def display_image(self, cv_image):
        if cv_image is None:
            self.image_label.clear()
            self.placeholder_text.setVisible(True)
            return
            
        self.current_image = cv_image
        h, w, ch = cv_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale pixmap to fit in the label while maintaining aspect ratio
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), 
                                               Qt.KeepAspectRatio, 
                                               Qt.SmoothTransformation))
        self.placeholder_text.setVisible(False)
        
    def resizeEvent(self, event):
        if self.current_image is not None:
            h, w, ch = self.current_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.current_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), 
                                                  Qt.KeepAspectRatio, 
                                                  Qt.SmoothTransformation))
        super().resizeEvent(event)
        
    def reset(self):
        self.image_label.clear()
        self.current_image = None
        self.placeholder_text.setVisible(True)


class DownloadThread(QThread):
    progress_update = pyqtSignal(str, int)
    download_complete = pyqtSignal(str, str)
    download_error = pyqtSignal(str, str)
    
    def __init__(self, url, save_path):
        super().__init__()
        self.url = url
        self.save_path = save_path
        
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
                    self.progress_update.emit(filename, progress)
            
            self.download_complete.emit(filename, self.save_path)
            
        except Exception as e:
            self.download_error.emit(filename, str(e))


class TrackingThread(QThread):
    progress_update = pyqtSignal(int)
    process_complete = pyqtSignal(str)
    process_error = pyqtSignal(str)
    
    def __init__(self, config, video_path, features_path, output_path):
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.features_path = features_path
        self.output_path = output_path
        
    def run(self):
        try:
            # Create examples_root from features_path
            examples_root = os.path.dirname(self.features_path)
            
            # Get the weights folder path
            weights_folder = os.path.join(os.getcwd(), "weights")
            yoloe_model = os.path.join(weights_folder, "yoloe-11l-seg-pf.pt")
            sam_model = os.path.join(weights_folder, "FastSAM-s.pt")
            
            # Extract directory from features path to use as examples_root
            # Call mosaic_mot function
            mosaic_mot(
                bytetrack_cfg_path="bytetrack.yaml",
                video_path=self.video_path,
                output_path=self.output_path,
                yoloe_model=yoloe_model,
                sam_model=sam_model,
                examples_root=examples_root,
                cache_file=self.features_path,
                dinov2_model="facebook/dinov2-base",
                config=self.config
            )
            
            self.process_complete.emit(self.output_path)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.process_error.emit(str(e))


class DetectionThread(QThread):
    process_complete = pyqtSignal(object)
    process_error = pyqtSignal(str)
    
    def __init__(self, config, image_path, features_path):
        super().__init__()
        self.config = config
        self.image_path = image_path
        self.features_path = features_path
        
    def run(self):
        try:
            # Create examples_root from features_path
            examples_root = os.path.dirname(self.features_path)
            
            # Get the weights folder path
            weights_folder = os.path.join(os.getcwd(), "weights")
            yoloe_model = os.path.join(weights_folder, "yoloe-11l-seg-pf.pt")
            sam_model = os.path.join(weights_folder, "FastSAM-s.pt")
            
            # Initialize the detector
            detector = MosaicDetSeg(
                yoloe_model=yoloe_model,
                sam_model=sam_model,
                examples_root=examples_root,
                cache_file=self.features_path,
                dinov2_model="facebook/dinov2-base",
                config=self.config
            )
            
            # Load and process the image
            image = cv2.imread(self.image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect and segment
            detection_results = detector.detect_and_segment(image_rgb)
            
            # Visualize results
            annotated_image = image_rgb.copy()
            detector.visualize_results(annotated_image, detection_results)
            
            self.process_complete.emit(annotated_image)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.process_error.emit(str(e))


class ConfigSlider(QWidget):
    value_changed = pyqtSignal(str, float)
    
    def __init__(self, name, min_val, max_val, default_val, parent=None):
        super().__init__(parent)
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Config name
        self.name_label = QLabel(name)
        self.name_label.setMinimumWidth(150)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        
        # Value label
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(60)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        layout.addWidget(self.name_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)
        
        # Connect signals
        self.slider.valueChanged.connect(self.update_value)
        
        # Set default value
        self.set_value(default_val)
        
    def update_value(self):
        slider_val = self.slider.value() / 100.0
        actual_val = self.min_val + slider_val * (self.max_val - self.min_val)
        
        # Format the display value
        if self.max_val <= 1:
            self.value_label.setText(f"{actual_val:.2f}")
        else:
            self.value_label.setText(f"{int(actual_val)}")
            
        self.value_changed.emit(self.name, actual_val)
        
    def set_value(self, value):
        normalized_val = (value - self.min_val) / (self.max_val - self.min_val)
        self.slider.setValue(int(normalized_val * 100))
        self.update_value()
        
    def get_value(self):
        slider_val = self.slider.value() / 100.0
        return self.min_val + slider_val * (self.max_val - self.min_val)


class MosaicInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MOSAIC-MOT Interface")
        self.setMinimumSize(1200, 800)
        
        # Initialize config with default values
        self.config = adjustable_configs.copy()
        
        # Setup UI
        self.init_ui()
        
        # Check for required models
        self.check_required_models()
        
    def init_ui(self):
        # Apply dark style
        self.setStyleSheet(DARK_STYLE)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Mode selection (Track/Detect)
        mode_selection = QWidget()
        mode_layout = QHBoxLayout(mode_selection)
        
        self.mode_group = QButtonGroup(self)
        self.track_radio = QRadioButton("Track Video")
        self.detect_radio = QRadioButton("Detect Image")
        self.track_radio.setChecked(True)
        
        self.mode_group.addButton(self.track_radio, 1)
        self.mode_group.addButton(self.detect_radio, 2)
        
        mode_layout.addWidget(self.track_radio)
        mode_layout.addWidget(self.detect_radio)
        mode_layout.addStretch()
        
        # Connect signals
        self.track_radio.toggled.connect(self.toggle_mode)
        
        # Stacked widget for different modes
        self.stack_layout = QTabWidget()
        self.stack_layout.setTabPosition(QTabWidget.North)
        self.stack_layout.setDocumentMode(True)
        
        # Create track and detect tabs
        self.track_tab = self.create_track_tab()
        self.detect_tab = self.create_detect_tab()
        
        self.stack_layout.addTab(self.track_tab, "Track Video")
        self.stack_layout.addTab(self.detect_tab, "Detect Image")
        self.stack_layout.setCurrentIndex(0)
        
        # Add widgets to main layout
        main_layout.addWidget(mode_selection)
        main_layout.addWidget(self.stack_layout)
        
        self.setCentralWidget(main_widget)
        
    def create_track_tab(self):
        track_widget = QWidget()
        track_layout = QVBoxLayout(track_widget)
        
        # File selection area
        file_selection = QWidget()
        file_layout = QHBoxLayout(file_selection)
        
        # Video path selection
        video_layout = QHBoxLayout()
        self.video_path_label = QLabel("Video Path:")
        self.video_path_edit = QLabel("No file selected")
        self.video_path_edit.setStyleSheet("background-color: #3E3E42; padding: 5px; border-radius: 3px;")
        self.video_path_button = QPushButton("Browse")
        self.video_path_button.clicked.connect(self.select_video_file)
        
        video_layout.addWidget(self.video_path_label)
        video_layout.addWidget(self.video_path_edit)
        video_layout.addWidget(self.video_path_button)
        
        # Features path selection
        features_layout = QHBoxLayout()
        self.features_path_label = QLabel("Features Path:")
        self.features_path_edit = QLabel("No file selected")
        self.features_path_edit.setStyleSheet("background-color: #3E3E42; padding: 5px; border-radius: 3px;")
        self.features_path_button = QPushButton("Browse")
        self.features_path_button.clicked.connect(self.select_features_file)
        
        features_layout.addWidget(self.features_path_label)
        features_layout.addWidget(self.features_path_edit)
        features_layout.addWidget(self.features_path_button)
        
        file_layout.addLayout(video_layout)
        file_layout.addLayout(features_layout)
        
        # Main content area with video players
        content_layout = QHBoxLayout()
        
        # Create video players
        self.input_player = VideoPlayer("Input Video")
        self.output_player = VideoPlayer("Tracked Video")
        
        content_layout.addWidget(self.input_player)
        content_layout.addWidget(self.output_player)
        
        # Config area
        config_layout = QVBoxLayout()
        
        # Create config sliders
        self.config_sliders = {}
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setFrameShape(QFrame.NoFrame)
        
        config_widget = QWidget()
        config_grid = QVBoxLayout(config_widget)
        
        # Mode selector
        mode_box = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["accurate", "fast"])
        self.mode_combo.currentTextChanged.connect(lambda text: self.update_config("mode", text))
        
        mode_box.addWidget(mode_label)
        mode_box.addWidget(self.mode_combo)
        mode_box.addStretch()
        
        config_grid.addLayout(mode_box)
        
        # Add sliders for numeric parameters
        for param, value in adjustable_configs.items():
            if param != "mode":
                if param == "similarity_threshold" or param == "embedding_similarity_threshold":
                    slider = ConfigSlider(param, 0.0, 1.0, value)
                else:
                    slider = ConfigSlider(param, 0, 100, value)
                slider.value_changed.connect(self.update_config)
                self.config_sliders[param] = slider
                config_grid.addWidget(slider)
        
        config_scroll.setWidget(config_widget)
        config_layout.addWidget(config_scroll)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Tracking")
        self.start_button.setFixedHeight(40)
        self.start_button.clicked.connect(self.start_tracking)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Ready")
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.status_label)
        
        # Add all layouts to main track layout
        track_layout.addWidget(file_selection)
        track_layout.addLayout(content_layout, 1)
        track_layout.addLayout(config_layout)
        track_layout.addLayout(control_layout)
        
        return track_widget
        
    def create_detect_tab(self):
        detect_widget = QWidget()
        detect_layout = QVBoxLayout(detect_widget)
        
        # File selection area
        file_selection = QWidget()
        file_layout = QHBoxLayout(file_selection)
        
        # Image path selection
        image_layout = QHBoxLayout()
        self.image_path_label = QLabel("Image Path:")
        self.image_path_edit = QLabel("No file selected")
        self.image_path_edit.setStyleSheet("background-color: #3E3E42; padding: 5px; border-radius: 3px;")
        self.image_path_button = QPushButton("Browse")
        self.image_path_button.clicked.connect(self.select_image_file)
        
        image_layout.addWidget(self.image_path_label)
        image_layout.addWidget(self.image_path_edit)
        image_layout.addWidget(self.image_path_button)
        
        # Features path selection for detection
        features_layout = QHBoxLayout()
        self.det_features_path_label = QLabel("Features Path:")
        self.det_features_path_edit = QLabel("No file selected")
        self.det_features_path_edit.setStyleSheet("background-color: #3E3E42; padding: 5px; border-radius: 3px;")
        self.det_features_path_button = QPushButton("Browse")
        self.det_features_path_button.clicked.connect(self.select_det_features_file)
        
        features_layout.addWidget(self.det_features_path_label)
        features_layout.addWidget(self.det_features_path_edit)
        features_layout.addWidget(self.det_features_path_button)
        
        file_layout.addLayout(image_layout)
        file_layout.addLayout(features_layout)
        
        # Main content area with image viewers
        content_layout = QHBoxLayout()
        
        # Create image viewers
        self.input_viewer = ImageViewer("Input Image")
        self.output_viewer = ImageViewer("Detected Image")
        
        content_layout.addWidget(self.input_viewer)
        content_layout.addWidget(self.output_viewer)
        
        # Config area (reuse the same config from track tab)
        config_layout = QVBoxLayout()
        
        # Create config sliders
        detect_config_scroll = QScrollArea()
        detect_config_scroll.setWidgetResizable(True)
        detect_config_scroll.setFrameShape(QFrame.NoFrame)
        
        detect_config_widget = QWidget()
        detect_config_grid = QVBoxLayout(detect_config_widget)
        
        # Mode selector
        det_mode_box = QHBoxLayout()
        det_mode_label = QLabel("Mode:")
        self.det_mode_combo = QComboBox()
        self.det_mode_combo.addItems(["accurate", "fast"])
        self.det_mode_combo.currentTextChanged.connect(lambda text: self.update_config("mode", text))
        
        det_mode_box.addWidget(det_mode_label)
        det_mode_box.addWidget(self.det_mode_combo)
        det_mode_box.addStretch()
        
        detect_config_grid.addLayout(det_mode_box)
        
        # Add sliders for numeric parameters
        self.det_config_sliders = {}
        for param, value in adjustable_configs.items():
            if param != "mode":
                if param == "similarity_threshold":
                    slider = ConfigSlider(param, 0.0, 1.0, value)
                else:
                    slider = ConfigSlider(param, 0, 100, value)
                slider.value_changed.connect(self.update_config)
                self.det_config_sliders[param] = slider
                detect_config_grid.addWidget(slider)
        
        detect_config_scroll.setWidget(detect_config_widget)
        config_layout.addWidget(detect_config_scroll)
        
        # Control buttons
        detect_control_layout = QHBoxLayout()
        
        self.detect_button = QPushButton("Start Detection")
        self.detect_button.setFixedHeight(40)
        self.detect_button.clicked.connect(self.start_detection)
        
        self.detect_status_label = QLabel("Ready")
        
        detect_control_layout.addWidget(self.detect_button)
        detect_control_layout.addWidget(self.detect_status_label)
        
        # Add all layouts to main detect layout
        detect_layout.addWidget(file_selection)
        detect_layout.addLayout(content_layout, 1)
        detect_layout.addLayout(config_layout)
        detect_layout.addLayout(detect_control_layout)
        
        return detect_widget
        
    def toggle_mode(self, checked):
        if checked:
            self.stack_layout.setCurrentIndex(0)  # Track tab
        else:
            self.stack_layout.setCurrentIndex(1)  # Detect tab
            
    def select_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            self.video_path_edit.setText(file_path)
            self.input_player.load_video(file_path)
            self.output_player.reset()
            
    def select_features_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Features File", "", "Pickle Files (*.pkl)"
        )
        if file_path:
            self.features_path_edit.setText(file_path)
            
    def select_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.image_path_edit.setText(file_path)
            self.input_viewer.load_image(file_path)
            self.output_viewer.reset()
            
    def select_det_features_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Features File", "", "Pickle Files (*.pkl)"
        )
        if file_path:
            self.det_features_path_edit.setText(file_path)
            
    def update_config(self, param_name, value):
        self.config[param_name] = value
        
    def check_required_models(self):
        """Check if required models exist in the weights folder, download if not."""
        weights_folder = os.path.join(os.getcwd(), "weights")
        os.makedirs(weights_folder, exist_ok=True)
        
        # Check each required model
        self.download_threads = []
        
        for model_name, url in MODEL_URLS.items():
            model_path = os.path.join(weights_folder, model_name)
            if not os.path.exists(model_path):
                self.download_model(url, model_path)
    
    def download_model(self, url, save_path):
        """Download a model file."""
        # Create progress dialog
        filename = os.path.basename(save_path)
        self.status_label.setText(f"Downloading {filename}...")
        self.progress_bar.setValue(0)
        
        # Start download thread
        download_thread = DownloadThread(url, save_path)
        download_thread.progress_update.connect(self.update_download_progress)
        download_thread.download_complete.connect(self.download_complete)
        download_thread.download_error.connect(self.download_error)
        
        self.download_threads.append(download_thread)
        download_thread.start()
        
    def update_download_progress(self, filename, progress):
        """Update download progress."""
        self.status_label.setText(f"Downloading {filename}: {progress}%")
        self.progress_bar.setValue(progress)
        
    def download_complete(self, filename, path):
        """Handle download completion."""
        self.status_label.setText(f"Downloaded {filename}")
        self.progress_bar.setValue(100)
        
    def download_error(self, filename, error):
        """Handle download error."""
        self.status_label.setText(f"Error downloading {filename}: {error}")
        
    def start_tracking(self):
        """Start the tracking process."""
        video_path = self.video_path_edit.text()
        features_path = self.features_path_edit.text()
        
        if video_path == "No file selected" or features_path == "No file selected":
            self.status_label.setText("Error: Please select both video and features files")
            return
            
        # Generate output path
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
        
        # Update UI
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Tracking in progress...")
        
        # Start tracking thread
        self.tracking_thread = TrackingThread(self.config, video_path, features_path, output_path)
        self.tracking_thread.progress_update.connect(self.update_tracking_progress)
        self.tracking_thread.process_complete.connect(self.tracking_complete)
        self.tracking_thread.process_error.connect(self.tracking_error)
        
        self.tracking_thread.start()
        
    def update_tracking_progress(self, progress):
        """Update tracking progress."""
        self.progress_bar.setValue(progress)
        
    def tracking_complete(self, output_path):
        """Handle tracking completion."""
        self.start_button.setEnabled(True)
        self.status_label.setText("Tracking complete")
        self.progress_bar.setValue(100)
        
        # Load the tracked video
        self.output_player.load_video(output_path)
        
    def tracking_error(self, error):
        """Handle tracking error."""
        self.start_button.setEnabled(True)
        self.status_label.setText(f"Error: {error}")
        
    def start_detection(self):
        """Start the detection process."""
        image_path = self.image_path_edit.text()
        features_path = self.det_features_path_edit.text()
        
        if image_path == "No file selected" or features_path == "No file selected":
            self.detect_status_label.setText("Error: Please select both image and features files")
            return
            
        # Update UI
        self.detect_button.setEnabled(False)
        self.detect_status_label.setText("Detection in progress...")
        
        # Start detection thread
        self.detection_thread = DetectionThread(self.config, image_path, features_path)
        self.detection_thread.process_complete.connect(self.detection_complete)
        self.detection_thread.process_error.connect(self.detection_error)
        
        self.detection_thread.start()
        
    def detection_complete(self, annotated_image):
        """Handle detection completion."""
        self.detect_button.setEnabled(True)
        self.detect_status_label.setText("Detection complete")
        
        # Display the detected image
        self.output_viewer.display_image(annotated_image)
        
    def detection_error(self, error):
        """Handle detection error."""
        self.detect_button.setEnabled(True)
        self.detect_status_label.setText(f"Error: {error}")
        

if __name__ == "__main__":
    try:
        print("Starting application...")
        app = QApplication(sys.argv)
        print("Created QApplication")
        window = MosaicInterface()
        print("Created MosaicInterface")
        window.show()
        print("Called show() on window")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error running application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 