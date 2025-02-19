import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import json

from PyQt5.QtWidgets import (QDialog, QFileDialog, QVBoxLayout, QPushButton, QLabel, 
                             QMessageBox, QGroupBox, QFormLayout, QHBoxLayout, QLineEdit,
                             QApplication)

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SaveProject(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        self.setWindowTitle("Save Project")
        self.resize(400, 100)

        # Setup the save file layout
        self.setup_save_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()

    def setup_save_layout(self):
        group_box = QGroupBox("Save Project")
        layout = QFormLayout()

        self.file_path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file_path)
        browse_layout = QHBoxLayout()
        browse_layout.addWidget(self.file_path_edit)
        browse_layout.addWidget(self.browse_button)
        layout.addRow("File Path:", browse_layout)

        group_box.setLayout(layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(group_box)
        self.setLayout(main_layout)

    def setup_buttons_layout(self):
        layout = self.layout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_project)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.save_button)
        layout.addWidget(self.cancel_button)

    def browse_file_path(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project JSON", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            self.file_path_edit.setText(file_path)

    def save_project(self):
        file_path = self.file_path_edit.text()
        if file_path:
            self.save_project_data(file_path)

    def save_project_data(self, file_path):

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            project_data = {
                'image_paths': self.get_images(),
                'labels': self.get_labels(),
                'annotations': self.get_annotations()
            }

            with open(file_path, 'w') as file:
                json.dump(project_data, file, indent=4)

            # Update current project label in status bar
            self.main_window.current_project_label.setText(f"{os.path.basename(file_path)}")

            QMessageBox.information(self.annotation_window, "Project Saved", "Project has been successfully saved.")

        except Exception as e:
            QMessageBox.warning(self.annotation_window, "Error Saving Project", f"An error occurred while saving the project: {str(e)}")

        finally:
            # Restor cursor to normal
            QApplication.restoreOverrideCursor()

        # Exit
        self.accept()

    def get_images(self):
        # Start the progress bar
        total_images = len(self.image_window.images)
        progress_bar = ProgressBar(self.label_window, "Exporting Images")
        progress_bar.show()
        progress_bar.start_progress(total_images)

        try:
            export_images = []
            
            # Loop through all of the image paths
            for image_path in list(self.image_window.images.keys()):
                export_images.append(image_path)
                progress_bar.update_progress()

        except Exception as e:
            QMessageBox.warning(self.label_window,
                    "Error Exporting Images",
                    f"An error occurred while exporting images: {str(e)}")
        finally:
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()

        return export_images

    def get_labels(self):
        # Start the progress bar
        total_labels = len(self.label_window.labels)
        progress_bar = ProgressBar(self.label_window, "Exporting Labels")
        progress_bar.show()
        progress_bar.start_progress(total_labels)

        try:
            export_labels = []

            # Loop through all the labels in label list
            for i, label in enumerate(self.label_window.labels):
                labels_data.append(label.to_dict())
                progress_bar.update_progress()

        except Exception as e:
            QMessageBox.warning(self.label_window,
                                "Error Exporting Labels",
                                f"An error occurred while exporting labels: {str(e)}")
        finally:
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()

        return export_labels

    def get_annotations(self):
        # Start progress bar
        total_annotations = len(list(self.annotation_window.annotations_dict.values()))
        progress_bar = ProgressBar(self.annotation_window, title="Exporting Annotations")
        progress_bar.show()
        progress_bar.start_progress(total_annotations)

        try:
            export_annotations = {}

            # Loop through all the annotations in the annotations dict
            for annotation in self.annotation_window.annotations_dict.values():
                image_path = annotation.image_path
                if image_path not in export_dict:
                    export_dict[image_path] = []

                # Convert annotation to dictionary based on its type
                if isinstance(annotation, PatchAnnotation):
                    annotation_dict = {
                        'type': 'PatchAnnotation',
                        **annotation.to_dict()
                    }
                elif isinstance(annotation, PolygonAnnotation):
                    annotation_dict = {
                        'type': 'PolygonAnnotation',
                        **annotation.to_dict()
                    }
                elif isinstance(annotation, RectangleAnnotation):
                    annotation_dict = {
                        'type': 'RectangleAnnotation',
                        **annotation.to_dict()
                    }
                else:
                    raise ValueError(f"Unknown annotation type: {type(annotation)}")

                export_annotations[image_path].append(annotation_dict)
                progress_bar.update_progress()

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Exporting Annotations",
                                f"An error occurred while exporting annotations: {str(e)}")

        finally:
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()

        return export_annotations
