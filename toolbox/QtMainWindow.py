import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import re

from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QIcon, QMouseEvent
from PyQt5.QtWidgets import (QDoubleSpinBox, QListWidget, QCheckBox)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QToolBar, QAction, QSizePolicy, QMessageBox,
                             QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSpinBox, QSlider, QDialog, 
                             QPushButton)

from toolbox.QtAnnotationWindow import AnnotationWindow
from toolbox.QtConfidenceWindow import ConfidenceWindow
from toolbox.QtEventFilter import GlobalEventFilter
from toolbox.QtImageWindow import ImageWindow
from toolbox.QtLabelWindow import LabelWindow
from toolbox.QtPatchSampling import PatchSamplingDialog

from toolbox.IO.QtImportImages import ImportImages
from toolbox.IO.QtImportLabels import ImportLabels
from toolbox.IO.QtImportAnnotations import ImportAnnotations
from toolbox.IO.QtImportCoralNetAnnotations import ImportCoralNetAnnotations
from toolbox.IO.QtImportViscoreAnnotations import ImportViscoreAnnotations
from toolbox.IO.QtImportTagLabAnnotations import ImportTagLabAnnotations
from toolbox.IO.QtExportLabels import ExportLabels
from toolbox.IO.QtExportAnnotations import ExportAnnotations
from toolbox.IO.QtExportCoralNetAnnotations import ExportCoralNetAnnotations
from toolbox.IO.QtExportViscoreAnnotations import ExportViscoreAnnotations
from toolbox.IO.QtExportTagLabAnnotations import ExportTagLabAnnotations

from toolbox.MachineLearning.TrainModel.QtClassify import Classify as ClassifyTrainModelDialog
from toolbox.MachineLearning.TrainModel.QtDetect import Detect as DetectTrainModelDialog
from toolbox.MachineLearning.TrainModel.QtSegment import Segment as SegmentTrainModelDialog

from toolbox.MachineLearning.DeployModel.QtClassify import Classify as ClassifyDeployModelDialog
from toolbox.MachineLearning.DeployModel.QtDetect import Detect as DetectDeployModelDialog
from toolbox.MachineLearning.DeployModel.QtSegment import Segment as SegmentDeployModelDialog

from toolbox.MachineLearning.BatchInference.QtClassify import Classify as ClassifyBatchInferenceDialog
from toolbox.MachineLearning.BatchInference.QtDetect import Detect as DetectBatchInferenceDialog
from toolbox.MachineLearning.BatchInference.QtSegment import Segment as SegmentBatchInferenceDialog

from toolbox.MachineLearning.ImportDataset.QtDetect import Detect as DetectImportDatasetDialog
from toolbox.MachineLearning.ImportDataset.QtSegment import Segment as SegmentImportDatasetDialog

from toolbox.MachineLearning.EvaluateModel.QtClassify import Classify as ClassifyEvaluateModelDialog
from toolbox.MachineLearning.EvaluateModel.QtDetect import Detect as DetectEvaluateModelDialog
from toolbox.MachineLearning.EvaluateModel.QtSegment import Segment as SegmentEvaluateModelDialog

from toolbox.MachineLearning.MergeDatasets.QtClassify import Classify as ClassifyMergeDatasetsDialog

from toolbox.MachineLearning.OptimizeModel.QtBase import Base as OptimizeModelDialog

from toolbox.MachineLearning.QtExportDataset import ExportDatasetDialog

from toolbox.SAM.QtDeployPredictor import DeployPredictorDialog as SAMDeployPredictorDialog
from toolbox.SAM.QtDeployGenerator import DeployGeneratorDialog as SAMDeployGeneratorDialog
from toolbox.SAM.QtBatchInference import BatchInferenceDialog as SAMBatchInferenceDialog

from toolbox.AutoDistill.QtDeployModel import DeployModelDialog as AutoDistillDeployModelDialog
from toolbox.AutoDistill.QtBatchInference import BatchInferenceDialog as AutoDistillBatchInferenceDialog

from toolbox.utilities import get_available_device
from toolbox.utilities import get_icon_path


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MainWindow(QMainWindow):
    toolChanged = pyqtSignal(str)  # Signal to emit the current tool state
    uncertaintyChanged = pyqtSignal(float)  # Signal to emit the current uncertainty threshold
    iouChanged = pyqtSignal(float)  # Signal to emit the current IoU threshold

    def __init__(self):
        super().__init__()

        # Define the icon path
        self.setWindowTitle("CoralNet-Toolbox")
        # Set the window icon
        main_window_icon_path = get_icon_path("coral.png")
        self.setWindowIcon(QIcon(main_window_icon_path))

        # Set window flags for resizing, minimize, maximize, and customizing
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        self.annotation_window = AnnotationWindow(self)
        self.label_window = LabelWindow(self)
        self.image_window = ImageWindow(self)
        self.confidence_window = ConfidenceWindow(self)

        self.import_images = ImportImages(self)
        self.import_labels = ImportLabels(self)
        self.import_annotations = ImportAnnotations(self)
        self.import_coralnet_annotations = ImportCoralNetAnnotations(self)
        self.import_viscore_annotations = ImportViscoreAnnotations(self)
        self.import_taglab_annotations = ImportTagLabAnnotations(self)
        self.export_labels = ExportLabels(self)
        self.export_annotations = ExportAnnotations(self)
        self.export_coralnet_annotations = ExportCoralNetAnnotations(self)
        self.export_viscore_annotations = ExportViscoreAnnotations(self)
        self.export_taglab_annotations = ExportTagLabAnnotations(self)

        # Set the default uncertainty threshold for Deploy Model and Batch Inference
        self.iou_thresh = 0.70
        self.uncertainty_thresh = 0.30

        # Create dialogs
        self.patch_annotation_sampling_dialog = PatchSamplingDialog(self)
        self.detect_import_dataset_dialog = DetectImportDatasetDialog(self)
        self.segment_import_dataset_dialog = SegmentImportDatasetDialog(self)
        self.export_dataset_dialog = ExportDatasetDialog(self)
        self.classify_merge_datasets_dialog = ClassifyMergeDatasetsDialog(self)
        self.classify_train_model_dialog = ClassifyTrainModelDialog(self)
        self.detect_train_model_dialog = DetectTrainModelDialog(self)
        self.segment_train_model_dialog = SegmentTrainModelDialog(self)
        self.classify_evaluate_model_dialog = ClassifyEvaluateModelDialog(self)
        self.detect_evaluate_model_dialog = DetectEvaluateModelDialog(self)
        self.segment_evaluate_model_dialog = SegmentEvaluateModelDialog(self)
        self.optimize_model_dialog = OptimizeModelDialog(self)
        self.classify_deploy_model_dialog = ClassifyDeployModelDialog(self)
        self.detect_deploy_model_dialog = DetectDeployModelDialog(self)
        self.segment_deploy_model_dialog = SegmentDeployModelDialog(self)
        self.classify_batch_inference_dialog = ClassifyBatchInferenceDialog(self)
        self.detect_batch_inference_dialog = DetectBatchInferenceDialog(self)
        self.segment_batch_inference_dialog = SegmentBatchInferenceDialog(self)
        self.sam_deploy_model_dialog = SAMDeployPredictorDialog(self)  # TODO
        self.sam_deploy_generator_dialog = SAMDeployGeneratorDialog(self)
        self.sam_batch_inference_dialog = SAMBatchInferenceDialog(self)
        self.auto_distill_deploy_model_dialog = AutoDistillDeployModelDialog(self)
        self.auto_distill_batch_inference_dialog = AutoDistillBatchInferenceDialog(self)

        # Connect signals to update status bar
        self.annotation_window.imageLoaded.connect(self.update_image_dimensions)
        self.annotation_window.mouseMoved.connect(self.update_mouse_position)
        self.annotation_window.viewChanged.connect(self.update_view_dimensions)

        # Connect the hover_point signal from AnnotationWindow to the methods in SAMTool
        self.annotation_window.hover_point.connect(self.annotation_window.tools["sam"].start_hover_timer)
        self.annotation_window.hover_point.connect(self.annotation_window.tools["sam"].stop_hover_timer)

        # Connect the toolChanged signal (to the AnnotationWindow)
        self.toolChanged.connect(self.annotation_window.set_selected_tool)
        # Connect the toolChanged signal (to the Toolbar)
        self.annotation_window.toolChanged.connect(self.handle_tool_changed)
        # Connect the selectedLabel signal to the LabelWindow's set_selected_label method
        self.annotation_window.labelSelected.connect(self.label_window.set_selected_label)
        # Connect the labelSelected signal from LabelWindow to update the selected label in AnnotationWindow
        self.label_window.labelSelected.connect(self.annotation_window.set_selected_label)
        # Connect the labelSelected signal from LabelWindow to update the transparency slider
        self.label_window.transparencyChanged.connect(self.update_label_transparency)
        # Connect the imageSelected signal to update_current_image_path in AnnotationWindow
        self.image_window.imageSelected.connect(self.annotation_window.update_current_image_path)
        # Connect the imageChanged signal from ImageWindow to cancel SAM working area
        self.image_window.imageChanged.connect(self.handle_image_changed)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # Menu bar
        self.menu_bar = self.menuBar()

        # Import menu
        self.import_menu = self.menu_bar.addMenu("Import")

        # Raster submenu
        self.import_rasters_menu = self.import_menu.addMenu("Rasters")

        self.import_images_action = QAction("Images", self)
        self.import_images_action.triggered.connect(self.import_images.import_images)
        self.import_rasters_menu.addAction(self.import_images_action)

        # Labels submenu
        self.import_labels_menu = self.import_menu.addMenu("Labels")

        self.import_labels_action = QAction("Labels (JSON)", self)
        self.import_labels_action.triggered.connect(self.import_labels.import_labels)
        self.import_labels_menu.addAction(self.import_labels_action)

        # Annotations submenu
        self.import_annotations_menu = self.import_menu.addMenu("Annotations")

        self.import_annotations_action = QAction("Annotations (JSON)", self)
        self.import_annotations_action.triggered.connect(self.import_annotations.import_annotations)
        self.import_annotations_menu.addAction(self.import_annotations_action)

        self.import_coralnet_annotations_action = QAction("CoralNet (CSV)", self)
        self.import_coralnet_annotations_action.triggered.connect(self.import_coralnet_annotations.import_annotations)
        self.import_annotations_menu.addAction(self.import_coralnet_annotations_action)

        self.import_viscore_annotations_action = QAction("Viscore (CSV)", self)
        self.import_viscore_annotations_action.triggered.connect(self.import_viscore_annotations.import_annotations)
        self.import_annotations_menu.addAction(self.import_viscore_annotations_action)

        self.import_taglab_annotations_action = QAction("TagLab (JSON)", self)
        self.import_taglab_annotations_action.triggered.connect(self.import_taglab_annotations.import_annotations)
        self.import_annotations_menu.addAction(self.import_taglab_annotations_action)

        # Dataset submenu
        self.import_dataset_menu = self.import_menu.addMenu("Dataset")
        
        # Import Detection Dataset submenu
        self.import_detect_dataset_action = QAction("Detect", self)
        self.import_detect_dataset_action.triggered.connect(self.detect_import_dataset_dialog.exec_)
        self.import_dataset_menu.addAction(self.import_detect_dataset_action)

        # Import Segmentation Dataset submenu
        self.import_segment_dataset_action = QAction("Segment", self)
        self.import_segment_dataset_action.triggered.connect(self.segment_import_dataset_dialog.exec_)
        self.import_dataset_menu.addAction(self.import_segment_dataset_action)

        # Export menu
        self.export_menu = self.menu_bar.addMenu("Export")

        # Labels submenu
        self.export_labels_menu = self.export_menu.addMenu("Labels")

        self.export_labels_action = QAction("Labels (JSON)", self)
        self.export_labels_action.triggered.connect(self.export_labels.export_labels)
        self.export_labels_menu.addAction(self.export_labels_action)

        # Annotations submenu
        self.export_annotations_menu = self.export_menu.addMenu("Annotations")

        self.export_annotations_action = QAction("Annotations (JSON)", self)
        self.export_annotations_action.triggered.connect(self.export_annotations.export_annotations)
        self.export_annotations_menu.addAction(self.export_annotations_action)

        self.export_coralnet_annotations_action = QAction("CoralNet (CSV)", self)
        self.export_coralnet_annotations_action.triggered.connect(self.export_coralnet_annotations.export_annotations)
        self.export_annotations_menu.addAction(self.export_coralnet_annotations_action)

        self.export_viscore_annotations_action = QAction("Viscore (CSV)", self)
        self.export_viscore_annotations_action.triggered.connect(self.export_viscore_annotations.export_annotations)
        self.export_annotations_menu.addAction(self.export_viscore_annotations_action)

        self.export_taglab_annotations_action = QAction("TagLab (JSON)", self)
        self.export_taglab_annotations_action.triggered.connect(self.export_taglab_annotations.export_annotations)
        self.export_annotations_menu.addAction(self.export_taglab_annotations_action)

        # Dataset submenu
        self.export_dataset_menu = self.export_menu.addMenu("Dataset")

        # Export YOLO Dataset menu
        self.export_dataset_action = QAction("YOLO (TXT)", self)
        self.export_dataset_action.triggered.connect(self.open_export_dataset_dialog)
        self.export_dataset_menu.addAction(self.export_dataset_action)

        # Sampling Annotations menu
        self.annotation_sampling_action = QAction("Sample", self)
        self.annotation_sampling_action.triggered.connect(self.open_patch_annotation_sampling_dialog)
        self.menu_bar.addAction(self.annotation_sampling_action)

        # CoralNet menu
        # self.coralnet_menu = self.menu_bar.addMenu("CoralNet")

        # self.coralnet_authenticate_action = QAction("Authenticate", self)
        # self.coralnet_authenticate_action.triggered.connect(
        #     lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        # self.coralnet_menu.addAction(self.coralnet_authenticate_action)

        # self.coralnet_upload_action = QAction("Upload", self)
        # self.coralnet_upload_action.triggered.connect(
        #     lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        # self.coralnet_menu.addAction(self.coralnet_upload_action)

        # self.coralnet_download_action = QAction("Download", self)
        # self.coralnet_download_action.triggered.connect(
        #     lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        # self.coralnet_menu.addAction(self.coralnet_download_action)

        # self.coralnet_model_api_action = QAction("Model API", self)
        # self.coralnet_model_api_action.triggered.connect(
        #     lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        # self.coralnet_menu.addAction(self.coralnet_model_api_action)

        # Ultralytics menu
        self.ml_menu = self.menu_bar.addMenu("Ultralytics")

        # Merge Datasets submenu
        self.ml_merge_datasets_menu = self.ml_menu.addMenu("Merge Datasets")
        
        self.ml_classify_merge_datasets_action = QAction("Classify", self)
        self.ml_classify_merge_datasets_action.triggered.connect(self.open_classify_merge_datasets_dialog)
        self.ml_merge_datasets_menu.addAction(self.ml_classify_merge_datasets_action)

        # Train Model submenu
        self.ml_train_model_menu = self.ml_menu.addMenu("Train Model")

        self.ml_classify_train_model_action = QAction("Classify", self)
        self.ml_classify_train_model_action.triggered.connect(self.open_classify_train_model_dialog)
        self.ml_train_model_menu.addAction(self.ml_classify_train_model_action)

        self.ml_detect_train_model_action = QAction("Detect", self)
        self.ml_detect_train_model_action.triggered.connect(self.open_detect_train_model_dialog)
        self.ml_train_model_menu.addAction(self.ml_detect_train_model_action)

        self.ml_segment_train_model_action = QAction("Segment", self)
        self.ml_segment_train_model_action.triggered.connect(self.open_segment_train_model_dialog)
        self.ml_train_model_menu.addAction(self.ml_segment_train_model_action)

        # Evaluate Model submenu 
        self.ml_evaluate_model_menu = self.ml_menu.addMenu("Evaluate Model")
        
        self.ml_classify_evaluate_model_action = QAction("Classify", self)
        self.ml_classify_evaluate_model_action.triggered.connect(self.open_classify_evaluate_model_dialog)
        self.ml_evaluate_model_menu.addAction(self.ml_classify_evaluate_model_action)
        
        self.ml_detect_evaluate_model_action = QAction("Detect", self)
        self.ml_detect_evaluate_model_action.triggered.connect(self.open_detect_evaluate_model_dialog)
        self.ml_evaluate_model_menu.addAction(self.ml_detect_evaluate_model_action)
        
        self.ml_segment_evaluate_model_action = QAction("Segment", self)
        self.ml_segment_evaluate_model_action.triggered.connect(self.open_segment_evaluate_model_dialog)
        self.ml_evaluate_model_menu.addAction(self.ml_segment_evaluate_model_action)
        
        # Optimize Model action
        self.ml_optimize_model_action = QAction("Optimize Model", self)
        self.ml_optimize_model_action.triggered.connect(self.open_optimize_model_dialog)
        self.ml_menu.addAction(self.ml_optimize_model_action)

        # Deploy Model submenu
        self.ml_deploy_model_menu = self.ml_menu.addMenu("Deploy Model")

        self.ml_classify_deploy_model_action = QAction("Classify", self)
        self.ml_classify_deploy_model_action.triggered.connect(self.open_classify_deploy_model_dialog)
        self.ml_deploy_model_menu.addAction(self.ml_classify_deploy_model_action)

        self.ml_detect_deploy_model_action = QAction("Detect", self)
        self.ml_detect_deploy_model_action.triggered.connect(self.open_detect_deploy_model_dialog)
        self.ml_deploy_model_menu.addAction(self.ml_detect_deploy_model_action)

        self.ml_segment_deploy_model_action = QAction("Segment", self)
        self.ml_segment_deploy_model_action.triggered.connect(self.open_segment_deploy_model_dialog)
        self.ml_deploy_model_menu.addAction(self.ml_segment_deploy_model_action)

        # Batch Inference submenu
        self.ml_batch_inference_menu = self.ml_menu.addMenu("Batch Inference")

        self.ml_classify_batch_inference_action = QAction("Classify", self)
        self.ml_classify_batch_inference_action.triggered.connect(self.open_classify_batch_inference_dialog)
        self.ml_batch_inference_menu.addAction(self.ml_classify_batch_inference_action)

        self.ml_detect_batch_inference_action = QAction("Detect", self)
        self.ml_detect_batch_inference_action.triggered.connect(self.open_detect_batch_inference_dialog)
        self.ml_batch_inference_menu.addAction(self.ml_detect_batch_inference_action)

        self.ml_segment_batch_inference_action = QAction("Segment", self)
        self.ml_segment_batch_inference_action.triggered.connect(self.open_segment_batch_inference_dialog)
        self.ml_batch_inference_menu.addAction(self.ml_segment_batch_inference_action)

        # SAM menu
        self.sam_menu = self.menu_bar.addMenu("SAM")
        
        # Deploy Model submenu
        self.sam_deploy_model_menu = self.sam_menu.addMenu("Deploy Model")
        
        self.sam_deploy_model_action = QAction("Predictor", self)
        self.sam_deploy_model_action.triggered.connect(self.open_sam_deploy_model_dialog)
        self.sam_deploy_model_menu.addAction(self.sam_deploy_model_action)
        
        self.sam_deploy_generator_action = QAction("Generator", self)
        self.sam_deploy_generator_action.triggered.connect(self.open_sam_deploy_generator_dialog)
        self.sam_deploy_model_menu.addAction(self.sam_deploy_generator_action)
        
        self.sam_batch_inference_action = QAction("Batch Inference", self)
        self.sam_batch_inference_action.triggered.connect(self.open_sam_batch_inference_dialog)
        self.sam_menu.addAction(self.sam_batch_inference_action)

        # Auto Distill menu
        self.auto_distill_menu = self.menu_bar.addMenu("AutoDistill")

        self.auto_distill_deploy_model_action = QAction("Deploy Model", self)
        self.auto_distill_deploy_model_action.triggered.connect(self.open_auto_distill_deploy_model_dialog)
        self.auto_distill_menu.addAction(self.auto_distill_deploy_model_action)
        
        self.auto_distill_batch_inference_action = QAction("Batch Inference", self)
        self.auto_distill_batch_inference_action.triggered.connect(self.open_auto_distill_batch_inference_dialog)
        self.auto_distill_menu.addAction(self.auto_distill_batch_inference_action)

        # Create and add the toolbar
        self.toolbar = QToolBar("Tools", self)
        self.toolbar.setOrientation(Qt.Vertical)
        self.toolbar.setFixedWidth(40)
        self.toolbar.setMovable(False)  # Lock the toolbar in place
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)

        # Define spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        spacer.setFixedHeight(10)  # Set a fixed height for the spacer
        
        # Define line separator
        separator = QWidget()
        separator.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        separator.setFixedHeight(1)  # Set a fixed height for the line separator
        
        # Add a spacer before the first tool with a fixed height
        self.toolbar.addWidget(spacer)

        # Define icon paths
        self.select_icon_path = get_icon_path("select.png")
        self.patch_icon_path = get_icon_path("patch.png")
        self.rectangle_icon_path = get_icon_path("rectangle.png")
        self.polygon_icon_path = get_icon_path("polygon.png")
        self.sam_icon_path = get_icon_path("sam.png")
        self.slicer_icon_path = get_icon_path("slicer.png")
        self.turtle_icon_path = get_icon_path("turtle.png")
        self.rabbit_icon_path = get_icon_path("rabbit.png")
        self.rocket_icon_path = get_icon_path("rocket.png")
        self.apple_icon_path = get_icon_path("apple.png")

        # Add tools here with icons
        self.select_tool_action = QAction(QIcon(self.select_icon_path), "Select", self)
        self.select_tool_action.setCheckable(True)
        self.select_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.select_tool_action)
        
        self.toolbar.addSeparator()
        
        self.patch_tool_action = QAction(QIcon(self.patch_icon_path), "Patch", self)
        self.patch_tool_action.setCheckable(True)
        self.patch_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.patch_tool_action)

        self.rectangle_tool_action = QAction(QIcon(self.rectangle_icon_path), "Rectangle", self)
        self.rectangle_tool_action.setCheckable(True)
        self.rectangle_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.rectangle_tool_action)

        self.polygon_tool_action = QAction(QIcon(self.polygon_icon_path), "Polygon", self)
        self.polygon_tool_action.setCheckable(True)
        self.polygon_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.polygon_tool_action)
        
        self.toolbar.addSeparator()
        
        self.sam_tool_action = QAction(QIcon(self.sam_icon_path), "SAM", self)
        self.sam_tool_action.setCheckable(True)
        self.sam_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.sam_tool_action)
        
        self.toolbar.addSeparator()
        
        self.slicer_tool_action = QAction(QIcon(self.slicer_icon_path), "Slicer", self)
        self.slicer_tool_action.setCheckable(False)
        self.slicer_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.slicer_tool_action)
        
        self.toolbar.addSeparator()

        # Add a spacer to push the device label to the bottom
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.toolbar.addWidget(spacer)

        # Add the device label widget as an action in the toolbar
        self.devices = get_available_device()
        self.current_device_index = 0
        self.device = self.devices[self.current_device_index]

        if self.device.startswith('cuda'):
            if len(self.devices) == 1:
                device_icon = QIcon(self.rabbit_icon_path)
            else:
                device_icon = QIcon(self.rocket_icon_path)
            device_tooltip = self.device
        elif self.device == 'mps':
            device_icon = QIcon(self.apple_icon_path)
            device_tooltip = 'mps'
        else:
            device_icon = QIcon(self.turtle_icon_path)
            device_tooltip = 'cpu'

        # Create the device action with the appropriate icon
        device_action = ClickableAction(device_icon, "", self)  # Empty string for the text
        self.device_tool_action = device_action
        self.device_tool_action.setCheckable(False)
        # Set the tooltip to show the value of self.device
        self.device_tool_action.setToolTip(device_tooltip)
        self.device_tool_action.triggered.connect(self.toggle_device)
        self.toolbar.addAction(self.device_tool_action)

        # Create status bar layout
        self.status_bar_layout = QHBoxLayout()

        # Labels for image dimensions and mouse position
        self.image_dimensions_label = QLabel("Image: 0 x 0")
        self.mouse_position_label = QLabel("Mouse: X: 0, Y: 0")
        self.view_dimensions_label = QLabel("View: 0 x 0")  # Add QLabel for view dimensions

        # Set fixed width for labels to prevent them from resizing
        self.image_dimensions_label.setFixedWidth(150)
        self.mouse_position_label.setFixedWidth(150)
        self.view_dimensions_label.setFixedWidth(150)  # Set fixed width for view dimensions label

        # Transparency slider
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 128)
        self.transparency_slider.setValue(128)  # Default transparency
        self.transparency_slider.valueChanged.connect(self.update_label_transparency)

        # Add a checkbox labeled "All" next to the transparency slider
        self.all_labels_checkbox = QCheckBox("All")
        self.all_labels_checkbox.setCheckState(Qt.Checked)
        self.all_labels_checkbox.stateChanged.connect(self.update_all_labels_transparency)

        # Spin box for IoU threshold control
        self.iou_thresh_spinbox = QDoubleSpinBox()
        self.iou_thresh_spinbox.setRange(0.0, 1.0)  # Range is 0.0 to 1.0
        self.iou_thresh_spinbox.setSingleStep(0.05)  # Step size for the spinbox
        self.iou_thresh_spinbox.setValue(self.iou_thresh)
        self.iou_thresh_spinbox.valueChanged.connect(self.update_iou_thresh)

        # Spin box for Uncertainty threshold control
        self.uncertainty_thresh_spinbox = QDoubleSpinBox()
        self.uncertainty_thresh_spinbox.setRange(0.0, 1.0)  # Range is 0.0 to 1.0
        self.uncertainty_thresh_spinbox.setSingleStep(0.05)  # Step size for the spinbox
        self.uncertainty_thresh_spinbox.setValue(self.uncertainty_thresh)
        self.uncertainty_thresh_spinbox.valueChanged.connect(self.update_uncertainty_thresh)

        # Spin box for annotation size control
        self.annotation_size_spinbox = QSpinBox()
        self.annotation_size_spinbox.setMinimum(1)
        self.annotation_size_spinbox.setMaximum(5000)
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.annotation_size_spinbox.valueChanged.connect(self.annotation_window.set_annotation_size)
        self.annotation_window.annotationSizeChanged.connect(self.annotation_size_spinbox.setValue)

        # Add widgets to status bar layout
        self.status_bar_layout.addWidget(self.image_dimensions_label)
        self.status_bar_layout.addWidget(self.mouse_position_label)
        self.status_bar_layout.addWidget(self.view_dimensions_label)
        self.status_bar_layout.addStretch()
        self.status_bar_layout.addWidget(QLabel("Transparency:"))
        self.status_bar_layout.addWidget(self.transparency_slider)
        self.status_bar_layout.addWidget(self.all_labels_checkbox)
        self.status_bar_layout.addStretch()
        self.status_bar_layout.addWidget(QLabel("IoU Threshold:"))
        self.status_bar_layout.addWidget(self.iou_thresh_spinbox)
        self.status_bar_layout.addWidget(QLabel("Uncertainty Threshold:"))
        self.status_bar_layout.addWidget(self.uncertainty_thresh_spinbox)
        self.status_bar_layout.addWidget(QLabel("Annotation Size:"))
        self.status_bar_layout.addWidget(self.annotation_size_spinbox)

        # --------------------------------------------------
        # Create the main layout
        # --------------------------------------------------
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create left and right layouts
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # Add status bar layout to left layout above the AnnotationWindow
        self.left_layout.addLayout(self.status_bar_layout)
        self.left_layout.addWidget(self.annotation_window, 85)
        self.left_layout.addWidget(self.label_window, 15)

        # Add widgets to right layout
        self.right_layout.addWidget(self.image_window, 54)
        self.right_layout.addWidget(self.confidence_window, 46)

        # Add left and right layouts to main layout
        self.main_layout.addLayout(self.left_layout, 85)
        self.main_layout.addLayout(self.right_layout, 15)

        # Set up global event filter
        self.global_event_filter = GlobalEventFilter(self)
        QApplication.instance().installEventFilter(self.global_event_filter)

    def showEvent(self, event):
        super().showEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() & Qt.WindowMinimized:
                # Allow minimizing
                pass
            elif self.windowState() & Qt.WindowMaximized:
                # Window is maximized, do nothing
                pass
            else:
                # Restore to normal state
                pass  # Do nothing, let the OS handle the restore

    def toggle_tool(self, state):
        action = self.sender()
        if action == self.select_tool_action:
            if state:
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.toolChanged.emit("select")
            else:
                self.toolChanged.emit(None)
        elif action == self.patch_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.toolChanged.emit("patch")
            else:
                self.toolChanged.emit(None)
        elif action == self.rectangle_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.toolChanged.emit("rectangle")
            else:
                self.toolChanged.emit(None)
        elif action == self.polygon_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.toolChanged.emit("polygon")
            else:
                self.toolChanged.emit(None)
        elif action == self.sam_tool_action:
            if not self.sam_deploy_model_dialog.loaded_model:
                self.sam_tool_action.setChecked(False)
                QMessageBox.warning(self, 
                                    "SAM Deploy Predictor", 
                                    "You must deploy a Predictor before using the SAM tool.")
                return
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.toolChanged.emit("sam")
            else:
                self.toolChanged.emit(None)

    def untoggle_all_tools(self):
        self.select_tool_action.setChecked(False)
        self.patch_tool_action.setChecked(False)
        self.rectangle_tool_action.setChecked(False)
        self.polygon_tool_action.setChecked(False)
        self.sam_tool_action.setChecked(False)
        self.toolChanged.emit(None)

    def handle_tool_changed(self, tool):
        if tool == "select":
            self.select_tool_action.setChecked(True)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
        elif tool == "patch":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(True)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
        elif tool == "rectangle":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(True)
            self.polygon_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
        elif tool == "polygon":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(True)
            self.sam_tool_action.setChecked(False)
        elif tool == "sam":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(True)
        else:
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)

    def toggle_device(self):
        dialog = DeviceSelectionDialog(self.devices, self)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_devices = dialog.selected_devices

            if not self.selected_devices:
                return

            # Convert the string to multi-CUDA format: "cuda:0,1,2"
            if len(self.selected_devices) == 1:
                self.device = self.selected_devices[0]
            else:
                # Get only the numerical values for cuda
                cuda_devices = [re.search(r'\d+', device).group() for device in self.selected_devices]
                self.device = f"{','.join(cuda_devices)} "

            # Update the icon and tooltip
            if self.device.startswith('cuda'):
                if len(self.selected_devices) == 1:
                    if self.device == 'cuda:0':
                        device_icon = QIcon(self.rabbit_icon_path)
                    else:
                        # Use a different icon for other CUDA devices
                        device_icon = QIcon(self.rocket_icon_path)
                    device_tooltip = self.device
                else:
                    # Use a different icon for multiple devices
                    device_icon = QIcon(self.rocket_icon_path)
                    device_tooltip = f"Multiple CUDA Devices: {self.selected_devices}"

            elif self.device == 'mps':
                device_icon = QIcon(self.apple_icon_path)
                device_tooltip = 'mps'
            else:
                device_icon = QIcon(self.turtle_icon_path)
                device_tooltip = 'cpu'

            self.device_tool_action.setIcon(device_icon)
            self.device_tool_action.setToolTip(device_tooltip)

    def handle_image_changed(self):
        if self.annotation_window.selected_tool == 'sam':
            self.annotation_window.tools['sam'].cancel_working_area()

    def update_image_dimensions(self, width, height):
        self.image_dimensions_label.setText(f"Image: {height} x {width}")

    def update_mouse_position(self, x, y):
        self.mouse_position_label.setText(f"Mouse: X: {x}, Y: {y}")

    def update_view_dimensions(self, original_width, original_height):
        # Current extent (view)
        extent = self.annotation_window.viewportToScene()

        top = round(extent.top())
        left = round(extent.left())
        width = round(extent.width())
        height = round(extent.height())

        bottom = top + height
        right = left + width

        # If the current extent includes areas outside the
        # original image, reduce it to be only the original image
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if bottom > original_height:
            bottom = original_height
        if right > original_width:
            right = original_width

        width = right - left
        height = bottom - top

        self.view_dimensions_label.setText(f"View: {height} x {width}")

    def get_transparency_value(self):
        return self.transparency_slider.value()

    def update_label_transparency(self, value):
        if self.all_labels_checkbox.isChecked():
            self.label_window.set_all_labels_transparency(value)
        else:
            self.label_window.set_label_transparency(value)
        self.update_transparency_slider(value)

    def update_transparency_slider(self, transparency):
        self.transparency_slider.setValue(transparency)

    def update_all_labels_transparency(self, state):
        if state == Qt.Checked:
            self.label_window.set_all_labels_transparency(self.transparency_slider.value())
        else:
            self.label_window.set_label_transparency(self.transparency_slider.value())

    def get_uncertainty_thresh(self):
        return self.uncertainty_thresh

    def update_uncertainty_thresh(self, value):
        if self.uncertainty_thresh != value:
            self.uncertainty_thresh = value
            self.uncertainty_thresh_spinbox.setValue(value)
            self.uncertaintyChanged.emit(value)

    def get_iou_thresh(self):
        return self.iou_thresh

    def update_iou_thresh(self, value):
        if self.iou_thresh != value:
            self.iou_thresh = value
            self.iou_thresh_spinbox.setValue(value)
            self.iouChanged.emit(value)

    def open_patch_annotation_sampling_dialog(self):

        if not self.image_window.image_paths:
            # Check if there are any images in the project
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before sampling annotations.")
            return

        try:
            # Proceed to open the dialog if images are loaded
            self.untoggle_all_tools()
            self.patch_annotation_sampling_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

        self.patch_annotation_sampling_dialog = None
        self.patch_annotation_sampling_dialog = PatchSamplingDialog(self)

    def open_import_dataset_dialog(self):
        try:
            self.untoggle_all_tools()
            self.import_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_export_dataset_dialog(self):
        # Check if there are loaded images
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Export Dataset",
                                "No images are present in the project.")
            return

        # Check if there are annotations
        if not len(self.annotation_window.annotations_dict):
            QMessageBox.warning(self,
                                "Export Dataset",
                                "No annotations are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.export_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_merge_datasets_dialog(self):
        try:
            self.untoggle_all_tools()
            self.classify_merge_datasets_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_train_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.classify_train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_detect_train_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.detect_train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_segment_train_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.segment_train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_evaluate_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.classify_evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_detect_evaluate_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.detect_evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_segment_evaluate_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.segment_evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_optimize_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.optimize_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_deploy_model_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Classify Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.classify_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_detect_deploy_model_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Detect Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.detect_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_segment_deploy_model_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Segment Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.segment_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "No images are present in the project.")
            return

        if not any(list(self.deploy_model_dialog.loaded_models.values())):
            QMessageBox.warning(self,
                                "Batch Inference",
                                "Please deploy a model before running batch inference.")
            return

        try:
            self.untoggle_all_tools()
            self.batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "No images are present in the project.")
            return

        if not self.classify_deploy_model_dialog.loaded_model:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "Please deploy a model before running batch inference.")
            return

        try:
            self.untoggle_all_tools()
            self.classify_batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_detect_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "No images are present in the project.")
            return

        if not self.detect_deploy_model_dialog.loaded_model:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "Please deploy a model before running batch inference.")
            return

        try:
            self.untoggle_all_tools()
            self.detect_batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_segment_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "No images are present in the project.")
            return

        if not self.segment_deploy_model_dialog.loaded_model:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "Please deploy a model before running batch inference.")
            return

        try:
            self.untoggle_all_tools()
            self.segment_batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_sam_deploy_model_dialog(self):  # TODO 
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "SAM Deploy Predictor",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.sam_deploy_model_dialog.exec_()  
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_sam_deploy_generator_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "SAM Deploy Generator",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.sam_deploy_generator_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_sam_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "SAM Batch Inference",
                                "No images are present in the project.")
            return

        if not self.sam_deploy_generator_dialog.loaded_model:
            QMessageBox.warning(self,
                                "SAM Batch Inference",
                                "Please deploy a generator before running batch inference.")
            return

        try:
            self.untoggle_all_tools()
            self.sam_batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_auto_distill_deploy_model_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "AutoDistill Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.auto_distill_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_auto_distill_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "AutoDistill Batch Inference",
                                "No images are present in the project.")
            return

        if not self.auto_distill_deploy_model_dialog.loaded_model:
            QMessageBox.warning(self,
                                "AutoDistill Batch Inference",
                                "Please deploy a model before running batch inference.")
            return

        try:
            self.untoggle_all_tools()
            self.auto_distill_batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")


class DeviceSelectionDialog(QDialog):
    def __init__(self, devices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Device")
        self.devices = devices
        self.selected_devices = []

        layout = QVBoxLayout()

        self.device_list = QListWidget()
        self.device_list.addItems(self.devices)
        self.device_list.setSelectionMode(QListWidget.SingleSelection)  # Allow only single selection
        layout.addWidget(self.device_list)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def accept(self):
        self.selected_devices = [item.text() for item in self.device_list.selectedItems()]
        if self.validate_selection():
            super().accept()
        else:
            QMessageBox.warning(self, "Invalid Selection", "Cannot mix CUDA devices with CPU or MPS.")

    def validate_selection(self):
        cuda_selected = any(device.startswith('cuda') for device in self.selected_devices)
        cpu_selected = 'cpu' in self.selected_devices
        mps_selected = 'mps' in self.selected_devices

        if cuda_selected and (cpu_selected or mps_selected):
            return False
        return True


class ClickableAction(QAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.trigger()
        super().mousePressEvent(event)