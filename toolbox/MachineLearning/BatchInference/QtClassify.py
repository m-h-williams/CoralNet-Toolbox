import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from operator import attrgetter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QSlider, QButtonGroup)

from toolbox.MachineLearning.BatchInference.QtBase import Base

from toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setup_generic_layout("Classify Batch Inference")

    def apply(self):
        """
        Apply batch inference for image classification.
        """
        # Get the Review Annotations
        if self.review_checkbox.isChecked():
            for image_path in self.get_selected_image_paths():
                self.annotations.extend(self.annotation_window.get_image_review_annotations(image_path))
        else:
            # Get all the annotations
            for image_path in self.get_selected_image_paths():
                self.annotations.extend(self.annotation_window.get_image_annotations(image_path))

        # Crop them, if not already cropped
        self.preprocess_patch_annotations()
        self.batch_inference()
        
    def setup_generic_layout(self):
        """
        Set up the layout with both generic and classification-specific options.
        """
        # Call parent implementation first
        super().setup_generic_layout()
        
        # Create a group box for annotation options
        annotation_group_box = QGroupBox("Annotation Options")
        annotation_layout = QVBoxLayout()

        # Create a button group for the annotation checkboxes
        self.annotation_options_group = QButtonGroup(self)

        self.review_checkbox = QCheckBox("Predict Review Annotation")
        self.all_checkbox = QCheckBox("Predict All Annotations")

        # Add the checkboxes to the button group
        self.annotation_options_group.addButton(self.review_checkbox)
        self.annotation_options_group.addButton(self.all_checkbox)

        # Ensure only one checkbox can be checked at a time
        self.annotation_options_group.setExclusive(True)

        # Set the default checkbox
        self.review_checkbox.setChecked(True)

        # Build the annotation layout
        annotation_layout.addWidget(self.review_checkbox)
        annotation_layout.addWidget(self.all_checkbox)
        annotation_group_box.setLayout(annotation_layout)

        # Add to main layout
        self.layout.addWidget(annotation_group_box)
        
        # Move button box to the bottom (it was added in parent class)
        self.layout.addWidget(self.button_box)
        
    def preprocess_patch_annotations(self):
        """
        Preprocess patch annotations by cropping the images based on the annotations.
        """
        # Get unique image paths
        self.image_paths = list(set(a.image_path for a in self.annotations))
        if not self.image_paths:
            return

        progress_bar = ProgressBar(self, title="Cropping Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        def crop(image_path, image_annotations):
            # Crop the image based on the annotations
            return self.annotation_window.crop_these_image_annotations(image_path, image_annotations)

        # Group annotations by image path
        groups = groupby(sorted(self.annotations, key=attrgetter('image_path')), key=attrgetter('image_path'))

        with ThreadPoolExecutor() as executor:
            future_to_image = {}
            for path, group in groups:
                future = executor.submit(crop, path, list(group))
                future_to_image[future] = path

            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    self.prepared_patches.extend(future.result())
                except Exception as exc:
                    print(f'{image_path} generated an exception: {exc}')
                finally:
                    progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()
        
    def batch_inference(self):
        """
        Perform batch inference on the selected images and annotations.
        """
        # Make predictions on each image's annotations
        progress_bar = ProgressBar(self, title="Batch Inference")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        if self.loaded_models['classify'] is not None:
            # Group annotations by image path
            groups = groupby(sorted(self.prepared_patches, key=attrgetter('image_path')), key=attrgetter('image_path'))

            # Make predictions on each image's annotations
            for path, patches in groups:
                self.deploy_model_dialog.predict_classification(annotations=list(patches))
                progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()
