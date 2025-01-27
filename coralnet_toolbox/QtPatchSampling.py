import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import random

from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QColor, QPen, QBrush
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QLabel, QDialog, QHBoxLayout, QPushButton,
                             QComboBox, QSpinBox, QFormLayout, QButtonGroup, QCheckBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.Tile.QtCommon import MarginInput


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PatchSamplingDialog(QDialog):
    annotationsSampled = pyqtSignal(list, bool)  # Signal to emit the sampled annotations and apply to all flag

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.annotation_window = main_window.annotation_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window

        self.setWindowTitle("Sample Annotations")
        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowState(Qt.WindowMaximized)  # Ensure the dialog is maximized

        self.layout = QVBoxLayout(self)

        # Sampling Method
        self.method_label = QLabel("Sampling Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Random", "Stratified Random", "Uniform"])
        self.method_combo.currentIndexChanged.connect(self.preview_annotations)  # Connect to preview
        self.layout.addWidget(self.method_label)
        self.layout.addWidget(self.method_combo)

        # Number of Annotations
        self.num_annotations_label = QLabel("Number of Annotations:")
        self.num_annotations_spinbox = QSpinBox()
        self.num_annotations_spinbox.setMinimum(1)
        self.num_annotations_spinbox.setMaximum(10000)
        self.num_annotations_spinbox.setValue(10)
        self.num_annotations_spinbox.valueChanged.connect(self.preview_annotations)  # Connect to preview
        self.layout.addWidget(self.num_annotations_label)
        self.layout.addWidget(self.num_annotations_spinbox)

        # Annotation Size
        self.annotation_size_label = QLabel("Annotation Size:")
        self.annotation_size_spinbox = QSpinBox()
        self.annotation_size_spinbox.setMinimum(32)
        self.annotation_size_spinbox.setMaximum(10000)  # Arbitrary large number for "infinite"
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.annotation_size_spinbox.valueChanged.connect(self.preview_annotations)  # Connect to preview
        self.layout.addWidget(self.annotation_size_label)
        self.layout.addWidget(self.annotation_size_spinbox)

        # Margin Offsets using MarginInput
        self.margin_input = MarginInput()
        self.margin_input.value_type.currentIndexChanged.connect(self.preview_annotations)  # Connect to preview
        self.margin_input.type_combo.currentIndexChanged.connect(self.preview_annotations)  # Connect to preview
        for spin in self.margin_input.margin_spins:
            spin.valueChanged.connect(self.preview_annotations)  # Connect to preview
        for double in self.margin_input.margin_doubles:
            double.valueChanged.connect(self.preview_annotations)  # Connect to preview
        self.layout.addWidget(self.margin_input)

        # Apply to Filtered Images Checkbox
        self.apply_filtered_checkbox = QCheckBox("Apply to filtered images")
        self.layout.addWidget(self.apply_filtered_checkbox)
        # Apply to Previous Images Checkbox
        self.apply_prev_checkbox = QCheckBox("Apply to previous images")
        self.layout.addWidget(self.apply_prev_checkbox)
        # Apply to Next Images Checkbox
        self.apply_next_checkbox = QCheckBox("Apply to next images")
        self.layout.addWidget(self.apply_next_checkbox)
        # Apply to All Images Checkbox
        self.layout.addWidget(self.apply_all_checkbox)

        # Ensure only one of the apply checkboxes can be selected at a time
        self.apply_group = QButtonGroup(self)
        self.apply_group.addButton(self.apply_filtered_checkbox)
        self.apply_group.addButton(self.apply_prev_checkbox)
        self.apply_group.addButton(self.apply_next_checkbox)
        self.apply_group.addButton(self.apply_all_checkbox)
        self.apply_group.setExclusive(False)

        # Preview Button
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview_annotations)
        self.layout.addWidget(self.preview_button)

        # Accept/Cancel Buttons
        self.button_box = QHBoxLayout()
        self.accept_button = QPushButton("Accept")
        self.accept_button.clicked.connect(self.accept_annotations)
        self.button_box.addWidget(self.accept_button)
        self.layout.addLayout(self.button_box)

        self.sampled_annotations = []

    def showEvent(self, event):
        # Call update_margin_spinbox when the dialog is shown
        self.annotation_window.imageLoaded.connect(self.update_margin_spinbox)
        self.update_margin_spinbox()

    def hideEvent(self, event):
        # Disconnect the signal when the dialog is hidden
        self.annotation_window.imageLoaded.disconnect(self.update_margin_spinbox)

    def update_margin_spinbox(self):
        if self.annotation_window.image_pixmap:
            width = self.annotation_window.image_pixmap.width()
            height = self.annotation_window.image_pixmap.height()
            # Set the margin spinboxes to the image dimensions
            annotation_size = self.annotation_size_spinbox.value()
            self.margin_input.single_spin.setMaximum(min(width, height) // 2 - annotation_size)
            for spin in self.margin_input.margin_spins:
                spin.setMaximum(min(width, height) // 2 - annotation_size)

    def sample_annotations(self, method, num_annotations, annotation_size, margins, image_width, image_height):
        # Extract the margins
        margin_x_min, margin_y_min, margin_x_max, margin_y_max = margins

        annotations = []

        if method == "Random":
            for _ in range(num_annotations):
                x = random.randint(margin_x_min, image_width - annotation_size - margin_x_max)
                y = random.randint(margin_y_min, image_height - annotation_size - margin_y_max)
                annotations.append((x, y, annotation_size))

        if method == "Uniform":
            grid_size = int(num_annotations ** 0.5)
            x_step = (image_width - margin_x_min - margin_x_max - annotation_size) / grid_size
            y_step = (image_height - margin_y_min - margin_y_max - annotation_size) / grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    x = margin_x_min + int(i * x_step + annotation_size / 2)
                    y = margin_y_min + int(j * y_step + annotation_size / 2)
                    annotations.append((x, y, annotation_size))

        if method == "Stratified Random":
            grid_size = int(num_annotations ** 0.5)
            x_step = (image_width - margin_x_min - margin_x_max - annotation_size) / grid_size
            y_step = (image_height - margin_y_min - margin_y_max - annotation_size) / grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    x = margin_x_min + int(
                        i * x_step + random.uniform(annotation_size / 2, x_step - annotation_size / 2))
                    y = margin_y_min + int(
                        j * y_step + random.uniform(annotation_size / 2, y_step - annotation_size / 2))
                    annotations.append((x, y, annotation_size))

        return annotations

    def preview_annotations(self):
        method = self.method_combo.currentText()
        num_annotations = self.num_annotations_spinbox.value()
        annotation_size = self.annotation_size_spinbox.value()
        margins = self.margin_input.get_value()

        self.sampled_annotations = self.sample_annotations(method,
                                                           num_annotations,
                                                           annotation_size,
                                                           margins,
                                                           self.annotation_window.image_pixmap.width(),
                                                           self.annotation_window.image_pixmap.height())

        self.draw_annotation_previews(margins)

    def draw_annotation_previews(self, margins):

        margin_x_min, margin_y_min, margin_x_max, margin_y_max = margins

        self.annotation_window.unselect_annotations()
        for annotation in self.sampled_annotations:
            x, y, size = annotation
            new_annotation = PatchAnnotation(QPointF(x + size // 2, y + size // 2),
                                             size,
                                             self.label_window.active_label.short_label_code,
                                             self.label_window.active_label.long_label_code,
                                             self.label_window.active_label.color,
                                             self.annotation_window.current_image_path,
                                             self.label_window.active_label.id,
                                             transparency=self.annotation_window.transparency)
            new_annotation.create_graphics_item(self.annotation_window.scene)
            self.annotation_window.annotations_dict[new_annotation.id] = new_annotation

    def accept_annotations(self):

        margins = self.margin_input.get_value()

        self.add_sampled_annotations(self.method_combo.currentText(),
                                     self.num_annotations_spinbox.value(),
                                     self.annotation_size_spinbox.value(),
                                     margins)

    def add_sampled_annotations(self, method, num_annotations, annotation_size, margins):

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.apply_to_filtered = self.apply_filtered_checkbox.isChecked()
        self.apply_to_prev = self.apply_prev_checkbox.isChecked()
        self.apply_to_next = self.apply_next_checkbox.isChecked()
        self.apply_to_all = self.apply_all_checkbox.isChecked()

        # Sets the LabelWindow and AnnotationWindow to Review
        review_label = self.label_window.get_label_by_id("-1")

        # Current image path showing
        current_image_path = self.annotation_window.current_image_path

        if self.apply_to_filtered:
            image_paths = self.image_window.filtered_image_paths
        elif self.apply_to_prev:
            current_image_index = self.image_window.image_paths.index(current_image_path)
            image_paths = self.image_window.image_paths[:current_image_index + 1]
        elif self.apply_to_next:
            current_image_index = self.image_window.image_paths.index(current_image_path)
            image_paths = self.image_window.image_paths[current_image_index:]
        elif self.apply_to_all:
            image_paths = self.image_window.image_paths
        else:
            # Only apply to the current image
            image_paths = [current_image_path]

        # Create and show the progress bar
        progress_bar = ProgressBar(self, title="Sampling Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths) * num_annotations)
        
        for image_path in image_paths:
            sampled_annotations = []
            
            # Load the rasterio representation
            rasterio_image = self.image_window.rasterio_open(image_path)
            height, width = rasterio_image.shape[0:2]

            # Sample the annotation, given params
            annotations = self.sample_annotations(method,
                                                  num_annotations,
                                                  annotation_size,
                                                  margins,
                                                  width,
                                                  height)

            for annotation in annotations:
                x, y, size = annotation
                new_annotation = PatchAnnotation(QPointF(x + size // 2, y + size // 2),
                                                 size,
                                                 review_label.short_label_code,
                                                 review_label.long_label_code,
                                                 review_label.color,
                                                 image_path,
                                                 review_label.id,
                                                 transparency=self.annotation_window.transparency)

                # Add annotation to the dict
                self.annotation_window.annotations_dict[new_annotation.id] = new_annotation
                sampled_annotations.append(new_annotation)
                progress_bar.update_progress()

            # Update the image window's image dict
            self.image_window.update_image_annotations(image_path)

        # Load the annotations for current image
        self.annotation_window.load_these_annotations(image_path=image_path, annotations=sampled_annotations)

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

        self.accept()
