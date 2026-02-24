import sys
import time
import numpy as np
import SimpleITK as sitk
import ants  # 新增
import tempfile  # 新增
import os  # 新增
import subprocess  # 新增
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, QFileDialog, 
                             QGroupBox, QProgressBar, QRadioButton, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QImage, QIntValidator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from brain_extraction import *
from antspynet.utilities import brain_extraction
import torch
from models.cnn3d import CNN3D
from pytorch_grad_cam import GradCAM

class RegressionTarget:
    def __init__(self):
        self.category = None
    
    def __call__(self, model_output):
        return model_output
    
class BrainAgePredictor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Brain Age Predictor")
        self.setGeometry(100, 100, 1600, 900)

        # 主要資料變數
        self.img_array = None
        self.img_min = None
        self.img_max = None
        self.is_sidebar_expanded = True
        self.original_file_path = None
        self.c1_array = None
        self.c2_array = None
        self.current_segmentation_map = 'c1'
        self.zoom_factors = {"axial": 1.0, "coronal": 1.0, "sagittal": 1.0}
        self.segmentation_zoom_factors = {
            "c1_axial": 1.0, "c1_coronal": 1.0, "c1_sagittal": 1.0,
            "c2_axial": 1.0, "c2_coronal": 1.0, "c2_sagittal": 1.0
        }

        # 設定中央元件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左側側邊欄
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setAlignment(Qt.AlignTop)
        self.sidebar_widget.setStyleSheet("background-color: #2c3e50; padding: 10px;")

        self.toggle_button = QPushButton("選單")
        self.toggle_button.setIcon(QIcon.fromTheme("menu"))
        self.toggle_button.setStyleSheet("color: white; background-color: #34495e; border: none; padding: 5px;")
        self.toggle_button.clicked.connect(self.toggle_sidebar)
        self.sidebar_layout.addWidget(self.toggle_button)

        self.original_file_button = QPushButton("原始檔案：未選擇")
        self.original_file_button.setStyleSheet("color: white; background-color: #34495e; border: 1px solid #555; padding: 10px; text-align: left;")
        self.original_file_button.clicked.connect(self.show_original_slices)
        self.original_file_button.setToolTip("未選擇檔案")
        self.sidebar_layout.addWidget(self.original_file_button)

        self.preprocess_button = QPushButton("預處理")
        self.preprocess_button.setStyleSheet("color: white; background-color: #34495e; border: 1px solid #555; padding: 10px; text-align: left;")
        self.preprocess_button.clicked.connect(self.show_preprocessing_view)
        self.sidebar_layout.addWidget(self.preprocess_button)

        self.model_button = QPushButton("模型")
        self.model_button.setStyleSheet("color: white; background-color: #34495e; border: 1px solid #555; padding: 10px; text-align: left;")
        self.model_button.clicked.connect(self.show_model_options)
        self.sidebar_layout.addWidget(self.model_button)

        self.gradcam_button = QPushButton("Grad-CAM熱力圖")
        self.gradcam_button.setStyleSheet("color: white; background-color: #34495e; border: 1px solid #555; padding: 10px; text-align: left;")
        self.sidebar_layout.addWidget(self.gradcam_button)

        main_layout.addWidget(self.sidebar_widget)
        self.content_layout = QVBoxLayout()

        # 標題始終置中且一致
        title_label = QLabel("🧠 Brain Age Predictor")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.content_layout.addWidget(title_label)

        self.run_all_button = QPushButton("執行所有模型預測")
        self.run_all_button.clicked.connect(self.run_all_predictions)
        self.run_all_button.setVisible(False)

        self.upload_button = QPushButton("📤 上傳 NIfTI MRI 檔案 (.nii 或 .nii.gz)")
        self.upload_button.clicked.connect(self.upload_file)
        self.content_layout.addWidget(self.upload_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        self.content_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(self.status_label)

        # 模型選項區域（右側，初始隱藏）
        self.model_options_widget = QWidget()
        self.model_options_layout = QVBoxLayout(self.model_options_widget)
        self.model_options_widget.setVisible(False)
        
        self.models = ["自訂 CNN", "3D DenseNet121", "3D ResNet50"]
        self.result_widgets = {} # <--- 新增這行：初始化 result_widgets

        # --- 將模型選項佈局的創建移到 __init__ ---
        # 將「執行所有模型預測」按鈕和狀態文字放在同一水平佈局
        run_status_layout = QHBoxLayout()
        run_status_layout.addStretch()
        # run_all_button 在 __init__ 中創建，但先不添加到佈局，因為它需要由 content_layout 管理
        # self.run_all_button = QPushButton("執行所有模型預測") # 在前面已經創建
        # self.run_all_button.clicked.connect(self.run_all_predictions)
        self.run_all_button.setStyleSheet("font-size: 16px; padding: 10px 20px;")
        # self.run_all_button.setVisible(False) # 初始隱藏
        run_status_layout.addWidget(self.run_all_button)
        # status_label 在 __init__ 中創建，但先不添加到佈局
        # self.status_label = QLabel("") # 在前面已經創建
        # self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; padding: 10px;")
        run_status_layout.addWidget(self.status_label)
        run_status_layout.addStretch()
        self.model_options_layout.addLayout(run_status_layout)

        # --- 新增：實際年齡輸入區域 ---
        true_age_layout = QHBoxLayout()
        true_age_label_input = QLabel("輸入實際年齡：")
        true_age_label_input.setStyleSheet("font-size: 16px;")
        self.true_age_input = QLineEdit()
        self.true_age_input.setPlaceholderText("例如：65")
        self.true_age_input.setValidator(QIntValidator(0, 150)) # 限制輸入為 0-150 的整數
        self.true_age_input.setFixedWidth(100) # 設定固定寬度
        self.true_age_input.setStyleSheet("font-size: 16px; padding: 5px;")

        true_age_layout.addStretch()
        true_age_layout.addWidget(true_age_label_input)
        true_age_layout.addWidget(self.true_age_input)
        true_age_layout.addStretch()
        self.model_options_layout.addLayout(true_age_layout)
        # --------------------------

        self.model_options_layout.addSpacing(30)

        # 添加模型標題和預測結果區域 (只創建一次)
        model_result_layout = QHBoxLayout()
        self.model_labels = {} # 用於存儲模型名稱標籤
        for model in self.models:
            model_container = QWidget()
            model_container_layout = QVBoxLayout(model_container)
            label = QLabel(model)
            label.setStyleSheet("font-size: 16px; font-weight: bold; padding-bottom: 10px;")
            label.setAlignment(Qt.AlignLeft)
            self.model_labels[model] = label # 儲存標籤引用
            model_container_layout.addWidget(label)

            # 創建結果顯示區域 (僅在 __init__ 中創建一次)
            result_widget = QWidget()
            result_layout = QVBoxLayout(result_widget)
            true_age_label = QLabel("實際年齡：")
            pred_age_label = QLabel("預測腦齡：")
            gap_label = QLabel("Brain Age Gap：")
            true_age_label.setStyleSheet("font-size: 18px;")
            pred_age_label.setStyleSheet("font-size: 18px;")
            gap_label.setStyleSheet("font-size: 18px;")
            result_layout.addWidget(true_age_label)
            result_layout.addWidget(pred_age_label)
            result_layout.addWidget(gap_label)
            result_layout.addWidget(true_age_label)
            result_layout.addWidget(pred_age_label)
            result_layout.addWidget(gap_label)
            result_widget.setVisible(False) # 初始隱藏結果

            # 儲存引用到 self.result_widgets (現在 self.result_widgets 已經存在)
            self.result_widgets[model] = {
                "widget": result_widget,
                "true_age": true_age_label,
                "pred_age": pred_age_label,
                "gap": gap_label
            }

            # 將結果區域添加到模型容器
            model_container_layout.addWidget(result_widget)
            model_result_layout.addWidget(model_container)

        self.model_options_layout.addLayout(model_result_layout)
        self.model_options_layout.addStretch()
        # --------------------------------------------

        # 將 model_options_widget 添加到主內容佈局
        self.content_layout.addWidget(self.model_options_widget)

        # 顯示區域
        self.display_widget = QWidget()
        self.display_layout = QVBoxLayout(self.display_widget)

        # 切片顯示區域
        self.slices_widget = QWidget()
        slices_layout = QHBoxLayout(self.slices_widget)
        slices_layout.setContentsMargins(0, 0, 0, 0)

        # 軸向切片組
        axial_group = QGroupBox("Axial (橫切)")
        axial_group.setStyleSheet("background-color: black; color: white; border: 1px solid #555;")
        axial_layout = QVBoxLayout()
        self.axial_figure = plt.figure(facecolor='black')
        self.axial_canvas = FigureCanvas(self.axial_figure)
        self.axial_ax = self.axial_figure.add_subplot(111)
        self.axial_ax.axis('off')
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setMinimum(0)
        self.axial_slider.valueChanged.connect(self.update_axial_slice)
        axial_layout.addWidget(self.axial_canvas)
        axial_layout.addWidget(self.axial_slider)
        axial_zoom_layout = QHBoxLayout()
        self.axial_zoom_in_btn = QPushButton("Zoom In")
        self.axial_zoom_in_btn.clicked.connect(lambda: self.zoom("axial", 1.2))
        self.axial_zoom_out_btn = QPushButton("Zoom Out")
        self.axial_zoom_out_btn.clicked.connect(lambda: self.zoom("axial", 0.8))
        axial_zoom_layout.addWidget(self.axial_zoom_in_btn)
        axial_zoom_layout.addWidget(self.axial_zoom_out_btn)
        axial_layout.addLayout(axial_zoom_layout)
        axial_group.setLayout(axial_layout)
        slices_layout.addWidget(axial_group)

        # 冠狀切片組
        coronal_group = QGroupBox("Coronal (冠狀)")
        coronal_group.setStyleSheet("background-color: black; color: white; border: 1px solid #555;")
        coronal_layout = QVBoxLayout()
        self.coronal_figure = plt.figure(facecolor='black')
        self.coronal_canvas = FigureCanvas(self.coronal_figure)
        self.coronal_ax = self.coronal_figure.add_subplot(111)
        self.coronal_ax.axis('off')
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.setMinimum(0)
        self.coronal_slider.valueChanged.connect(self.update_coronal_slice)
        coronal_layout.addWidget(self.coronal_canvas)
        coronal_layout.addWidget(self.coronal_slider)
        coronal_zoom_layout = QHBoxLayout()
        self.coronal_zoom_in_btn = QPushButton("Zoom In")
        self.coronal_zoom_in_btn.clicked.connect(lambda: self.zoom("coronal", 1.2))
        self.coronal_zoom_out_btn = QPushButton("Zoom Out")
        self.coronal_zoom_out_btn.clicked.connect(lambda: self.zoom("coronal", 0.8))
        coronal_zoom_layout.addWidget(self.coronal_zoom_in_btn)
        coronal_zoom_layout.addWidget(self.coronal_zoom_out_btn)
        coronal_layout.addLayout(coronal_zoom_layout)
        coronal_group.setLayout(coronal_layout)
        slices_layout.addWidget(coronal_group)

        # 矢狀切片組
        sagittal_group = QGroupBox("Sagittal (矢狀)")
        sagittal_group.setStyleSheet("background-color: black; color: white; border: 1px solid #555;")
        sagittal_layout = QVBoxLayout()
        self.sagittal_figure = plt.figure(facecolor='black')
        self.sagittal_canvas = FigureCanvas(self.sagittal_figure)
        self.sagittal_ax = self.sagittal_figure.add_subplot(111)
        self.sagittal_ax.axis('off')
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.setMinimum(0)
        self.sagittal_slider.valueChanged.connect(self.update_sagittal_slice)
        sagittal_layout.addWidget(self.sagittal_canvas)
        sagittal_layout.addWidget(self.sagittal_slider)
        sagittal_zoom_layout = QHBoxLayout()
        self.sagittal_zoom_in_btn = QPushButton("Zoom In")
        self.sagittal_zoom_in_btn.clicked.connect(lambda: self.zoom("sagittal", 1.2))
        self.sagittal_zoom_out_btn = QPushButton("Zoom Out")
        self.sagittal_zoom_out_btn.clicked.connect(lambda: self.zoom("sagittal", 0.8))
        sagittal_zoom_layout.addWidget(self.sagittal_zoom_in_btn)
        sagittal_zoom_layout.addWidget(self.sagittal_zoom_out_btn)
        sagittal_layout.addLayout(sagittal_zoom_layout)
        sagittal_group.setLayout(sagittal_layout)
        slices_layout.addWidget(sagittal_group)

        self.slices_widget.setVisible(False)
        self.display_layout.addWidget(self.slices_widget)

        # --- 预处理视图 ---
        self.preprocessing_widget = QWidget()
        preprocessing_layout = QVBoxLayout(self.preprocessing_widget)
        
        # SPM 分割结果区域
        processed_group = QGroupBox("SPM12 分割结果")
        processed_main_layout = QVBoxLayout(processed_group)

        # 上方：切换 c1/c2 的按钮
        switch_layout = QHBoxLayout()
        self.c1_button = QRadioButton("顯示 c1 (灰質)")
        self.c1_button.setChecked(True)
        self.c1_button.toggled.connect(lambda checked: self.switch_segmentation_map('c1') if checked else None)
        self.c2_button = QRadioButton("顯示 c2 (白質)")
        self.c2_button.toggled.connect(lambda checked: self.switch_segmentation_map('c2') if checked else None)
        switch_layout.addWidget(self.c1_button)
        switch_layout.addWidget(self.c2_button)
        switch_layout.addStretch()
        processed_main_layout.addLayout(switch_layout)

        # 下方：c1 和 c2 的三视图容器 (水平排列)
        self.c1_views_widget = self._create_segmentation_view_widget('c1')
        self.c2_views_widget = self._create_segmentation_view_widget('c2')
        self.c2_views_widget.setVisible(False)

        segmentation_views_container = QWidget()
        segmentation_views_layout = QHBoxLayout(segmentation_views_container)
        segmentation_views_layout.setContentsMargins(0,0,0,0)
        segmentation_views_layout.addWidget(self.c1_views_widget)
        segmentation_views_layout.addWidget(self.c2_views_widget)

        processed_main_layout.addWidget(segmentation_views_container)

        preprocessing_layout.addWidget(processed_group)

        self.preprocessing_widget.setVisible(False)
        self.display_layout.addWidget(self.preprocessing_widget)

        self.content_layout.addWidget(self.display_widget)
        main_layout.addLayout(self.content_layout)

        # --- GradCAM 視圖 ---
        self.gradcam_widget = QWidget()
        gradcam_layout = QVBoxLayout(self.gradcam_widget)

        # 模型切換按鈕
        model_switch_layout = QHBoxLayout()
        self.gradcam_cnn_btn = QRadioButton("CNN")
        self.gradcam_resnet_btn = QRadioButton("ResNet18")
        self.gradcam_densenet_btn = QRadioButton("DenseNet121")
        self.gradcam_cnn_btn.setChecked(True)
        model_switch_layout.addWidget(self.gradcam_cnn_btn)
        model_switch_layout.addWidget(self.gradcam_resnet_btn)
        model_switch_layout.addWidget(self.gradcam_densenet_btn)
        model_switch_layout.addStretch()
        gradcam_layout.addLayout(model_switch_layout)

        # 三視圖
        self.gradcam_views = {}
        views_layout = QHBoxLayout()
        self.gradcam_sliders = {}
        for view in ["axial", "coronal", "sagittal"]:
            vbox = QVBoxLayout()
            fig = plt.figure(facecolor='black')
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.axis('off')
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.valueChanged.connect(lambda value, v=view: self.update_gradcam_display(v, value))
            vbox.addWidget(canvas)
            vbox.addWidget(slider)
            widget = QWidget()
            widget.setLayout(vbox)
            views_layout.addWidget(widget)
            self.gradcam_views[view] = {"fig": fig, "canvas": canvas, "ax": ax}
            self.gradcam_sliders[view] = slider
        gradcam_layout.addLayout(views_layout)

        self.gradcam_widget.setVisible(False)
        self.display_layout.addWidget(self.gradcam_widget)

        # 連接按鈕
        self.gradcam_button.clicked.connect(self.show_gradcam_view)
        self.gradcam_cnn_btn.toggled.connect(lambda checked: self.set_gradcam_model("cnn") if checked else None)
        self.gradcam_resnet_btn.toggled.connect(lambda checked: self.set_gradcam_model("resnet18") if checked else None)
        self.gradcam_densenet_btn.toggled.connect(lambda checked: self.set_gradcam_model("densenet121") if checked else None)
        self.current_gradcam_model = "cnn"

        self.update_sidebar()

        # 新增：快取不同模型的 Grad-CAM 結果
        self.gradcam_cache = {}

    def toggle_sidebar(self):
        self.is_sidebar_expanded = not self.is_sidebar_expanded
        self.update_sidebar()

    def update_sidebar(self):
        if self.is_sidebar_expanded:
            self.sidebar_widget.setFixedWidth(200)
            for widget in self.sidebar_widget.findChildren(QWidget):
                widget.setVisible(True)
        else:
            self.sidebar_widget.setFixedWidth(50)
            for widget in self.sidebar_widget.findChildren(QWidget):
                if widget != self.toggle_button:
                    widget.setVisible(False)

    def clear_layout(self, layout):
        """安全地清除佈局中的所有項目，但不刪除 Widget"""
        # 這個函數現在可能不再需要，或者至少不再需要用於 model_options_layout
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    # 從佈局移除，但不刪除 C++ 物件
                    widget.setParent(None)
                else:
                    # 遞迴清除子佈局
                    self.clear_layout(item.layout())

    def show_model_options(self):
        """顯示模型選項 (修改後)"""
        if self.img_array is None:
            self.status_label.setText("❌ 請先上傳 MRI 檔案！")
            return

        # 隱藏其他主視圖
        self.slices_widget.setVisible(False)
        self.preprocessing_widget.setVisible(False)
        self.upload_button.setVisible(False) # 模型選項時隱藏上傳按鈕
        self.gradcam_widget.setVisible(False)

        self.run_all_button.setVisible(True)
        self.status_label.setText("請輸入實際年齡並執行模型預測") # 更新提示

        # 重設所有模型的結果顯示為不可見 (以防上次預測結果殘留)
        for model in self.models:
             # 檢查 widget 是否仍然有效 (雖然理論上不應再被刪除)
             if self.result_widgets[model]["widget"]:
                 self.result_widgets[model]["widget"].setVisible(False)

        # 直接顯示包含預建佈局的 model_options_widget
        self.model_options_widget.setVisible(True)

    def show_original_slices(self):
        self.model_options_widget.setVisible(False)
        self.preprocessing_widget.setVisible(False)  # 確保預處理視圖被隱藏
        self.run_all_button.setVisible(False)
        self.upload_button.setVisible(True)
        self.gradcam_widget.setVisible(False)

        if self.img_array is not None:
            for model in self.models:
                self.result_widgets[model]["widget"].setVisible(False)
            self.slices_widget.setVisible(True)
            self.update_axial_slice()
            self.update_coronal_slice()
            self.update_sagittal_slice()
            self.status_label.setText("顯示原始 MRI 切片")
        else:
            self.slices_widget.setVisible(False)
            self.status_label.setText("請上傳 MRI 檔案以顯示切片")

    @pyqtSlot()
    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "上傳 NIfTI 檔案", "", "NIfTI Files (*.nii *.nii.gz)"
        )
        if file_path:
            self.original_file_path = file_path
            # 显示简短路径（只显示文件名）
            short_path = os.path.basename(file_path)
            self.original_file_button.setText(f"原始檔案：{short_path}")
            # 设置完整路径作为提示
            self.original_file_button.setToolTip(file_path)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("⏳ 正在處理 MRI 檔案...")
            QApplication.processEvents()
            try:
                self.progress_bar.setValue(10)
                QApplication.processEvents()
                time.sleep(0.5)
                img = sitk.ReadImage(file_path)
                
                self.progress_bar.setValue(30)
                QApplication.processEvents()

                time.sleep(0.5)
                img_ras = self.reorient_to_ras(img)
                
                self.progress_bar.setValue(60)
                QApplication.processEvents()

                time.sleep(0.5)
                self.img_array = sitk.GetArrayFromImage(img_ras)
                self.img_min = np.min(self.img_array)
                self.img_max = np.max(self.img_array)
                self.axial_slider.setMaximum(self.img_array.shape[0] - 1)
                self.axial_slider.setValue(self.img_array.shape[0] // 2)
                self.coronal_slider.setMaximum(self.img_array.shape[1] - 1)
                self.coronal_slider.setValue(self.img_array.shape[1] // 2)
                self.sagittal_slider.setMaximum(self.img_array.shape[2] - 1)
                self.sagittal_slider.setValue(self.img_array.shape[2] // 2)
                self.progress_bar.setValue(90)
                QApplication.processEvents()

                # --- 新增：清空實際年齡輸入框 ---
                if hasattr(self, 'true_age_input'): # 確保輸入框已創建
                    self.true_age_input.clear()
                # -----------------------------

                time.sleep(0.5)
                self.slices_widget.setVisible(True)
                self.preprocessing_widget.setVisible(False)
                self.model_options_widget.setVisible(False)
                self.status_label.setText("✅ MRI 檔案已成功讀取並轉正向座標 (RAS)！")
                self.progress_bar.setValue(100)
                QApplication.processEvents()

                self.update_axial_slice()
                self.update_coronal_slice()
                self.update_sagittal_slice()
                self.progress_bar.setVisible(False)
                self.c1_array = None
                self.c2_array = None
                self.current_segmentation_map = 'c1'
                self.gradcam_cache = {}  # 新增：清空 Grad-CAM 快取

                # 預處理完成，初始化分割視圖
                self.initialize_segmentation_views()

            except Exception as e:
                self.status_label.setText(f"❌ 錯誤: {str(e)}")
                self.progress_bar.setVisible(False)

    def reorient_to_ras(self, image):
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation("RAS")
        return orienter.Execute(image)

    def zoom(self, slice_type, factor):
        self.zoom_factors[slice_type] = max(0.1, min(5.0, self.zoom_factors[slice_type] * factor))
        if slice_type == "axial":
            self.update_axial_slice()
        elif slice_type == "coronal":
            self.update_coronal_slice()
        elif slice_type == "sagittal":
            self.update_sagittal_slice()

    def update_axial_slice(self):
        if self.img_array is not None:
            index = self.axial_slider.value()
            self.axial_ax.clear()
            slice_data = np.rot90(self.img_array[index, :, :], k=2)
            zoom = self.zoom_factors["axial"]
            h, w = slice_data.shape
            center_h, center_w = h // 2, w // 2
            new_h, new_w = int(h / zoom), int(w / zoom)
            h_start = max(0, center_h - new_h // 2)
            h_end = min(h, h_start + new_h)
            w_start = max(0, center_w - new_w // 2)
            w_end = min(w, w_start + new_w)
            
            if h_start >= h_end or w_start >= w_end:
                zoomed_slice = slice_data
            else:
                zoomed_slice = slice_data[h_start:h_end, w_start:w_end]
                
            self.axial_ax.imshow(zoomed_slice, cmap="gray", vmin=self.img_min, vmax=self.img_max)
            # --- 修改標題，從1開始顯示切片索引 ---
            self.axial_ax.set_title(f"Axial Slice: {index + 1}", color='white', fontsize=10)
            # ---------------
            self.axial_ax.axis('off')
            self.axial_canvas.draw()

    def update_coronal_slice(self):
        if self.img_array is not None:
            index = self.coronal_slider.value()
            self.coronal_ax.clear()
            slice_data = np.rot90(self.img_array[:, index, :], k=2)
            zoom = self.zoom_factors["coronal"]
            h, w = slice_data.shape
            center_h, center_w = h // 2, w // 2
            new_h, new_w = int(h / zoom), int(w / zoom)
            h_start = max(0, center_h - new_h // 2)
            h_end = min(h, h_start + new_h)
            w_start = max(0, center_w - new_w // 2)
            w_end = min(w, w_start + new_w)
            
            if h_start >= h_end or w_start >= w_end:
                zoomed_slice = slice_data
            else:
                zoomed_slice = slice_data[h_start:h_end, w_start:w_end]
                
            self.coronal_ax.imshow(zoomed_slice, cmap="gray", vmin=self.img_min, vmax=self.img_max)
            # --- 修改標題，從1開始顯示切片索引 ---
            self.coronal_ax.set_title(f"Coronal Slice: {index + 1}", color='white', fontsize=10)
            # ---------------
            self.coronal_ax.axis('off')
            self.coronal_canvas.draw()

    def update_sagittal_slice(self):
        if self.img_array is not None:
            index = self.sagittal_slider.value()
            self.sagittal_ax.clear()
            slice_data = np.rot90(self.img_array[:, :, index], k=2)
            zoom = self.zoom_factors["sagittal"]
            h, w = slice_data.shape
            center_h, center_w = h // 2, w // 2
            new_h, new_w = int(h / zoom), int(w / zoom)
            h_start = max(0, center_h - new_h // 2)
            h_end = min(h, h_start + new_h)
            w_start = max(0, center_w - new_w // 2)
            w_end = min(w, w_start + new_w)
            
            if h_start >= h_end or w_start >= w_end:
                 zoomed_slice = slice_data
            else:
                 zoomed_slice = slice_data[h_start:h_end, w_start:w_end]
                 
            self.sagittal_ax.imshow(zoomed_slice, cmap="gray", vmin=self.img_min, vmax=self.img_max)
            # --- 修改標題，從1開始顯示切片索引 ---
            self.sagittal_ax.set_title(f"Sagittal Slice: {index + 1}", color='white', fontsize=10)
            # ---------------
            self.sagittal_ax.axis('off')
            self.sagittal_canvas.draw()

    @pyqtSlot()
    def run_all_predictions(self):
        if self.img_array is None:
            self.status_label.setText("❌ 請先上傳 MRI 檔案！")
            return

        # 檢查是否已經進行了預處理
        if self.c1_array is None:
            self.status_label.setText("❌ 請先進行預處理！")
            return

        # --- 獲取並驗證實際年齡輸入 ---
        true_age_text = self.true_age_input.text()
        if not true_age_text:
            self.status_label.setText("❌ 請輸入實際年齡！")
            return
        try:
            true_age = int(true_age_text)
        except ValueError:
            self.status_label.setText("❌ 實際年齡必須是有效的數字！")
            return

        self.status_label.setText("⏳ 正在進行所有模型預測...")
        QApplication.processEvents()

        input_tensor = torch.from_numpy(self.c1_array).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 維度

        # 模型路徑字典
        model_paths = {
            "自訂 CNN": ("C:/Users/user/Desktop/test/cnn.pth", CNN3D),
            "3D ResNet50": ("C:/Users/user/Desktop/test/resnet18.pth", resnet18),
            "3D DenseNet121": ("C:/Users/user/Desktop/test/densenet121.pth", densenet121_3d)
        }

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tensor = input_tensor.to(device)

            for model_name in self.models:
                if model_name in model_paths:
                    model_path, model_class = model_paths[model_name]
                    
                    # 載入模型和權重
                    model = model_class(num_classes=1)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()

                    # 進行預測
                    with torch.no_grad():
                        pred_age = model(input_tensor).item()

                    # 計算 Brain Age Gap
                    gap = pred_age - true_age

                    # 更新顯示結果
                    if model_name in self.result_widgets:
                        result_widget_data = self.result_widgets[model_name]
                        result_widget_data["true_age"].setText(f"✅ 實際年齡：{true_age} 歲")
                        result_widget_data["pred_age"].setText(f"🎯 預測腦齡：{pred_age:.2f} 歲")
                        result_widget_data["gap"].setText(f"🧮 Brain Age Gap：{gap:+.2f} 歲")
                        result_widget_data["widget"].setVisible(True)

            self.status_label.setText("✅ 所有模型預測完成！")

        except Exception as e:
            self.status_label.setText(f"❌ 預測過程發生錯誤：{str(e)}")
            print(f"預測錯誤：{e}")

    def preprocess_mri(self):
        if self.original_file_path is None:
            self.status_label.setText("❌ 請先上傳MRI檔案！")
            return
        self.gradcam_cache = {}  # 新增：清空 Grad-CAM 快取
        self.status_label.setText("⏳ 正在進行預處理 ( SPM 分割)...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        try:
            matlab_script_dir = r"C:\Users\user\Desktop"
            template_path = r"D:\Colab_Notebooks\python\skull-stripping-3D-brain-mri-main\assets\templates\mni_icbm152_t1_tal_nlin_sym_09a.nii"
            if not os.path.exists(matlab_script_dir):
                raise FileNotFoundError(f"MATLAB 脚本目錄未找到: {matlab_script_dir}")
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"模板文件未找到: {template_path}")

            with tempfile.TemporaryDirectory() as tmpdirname:
                print(f"臨時目錄: {tmpdirname}")
                self.progress_bar.setValue(10)
                QApplication.processEvents()
                raw_img_ants = ants.image_read(self.original_file_path, reorient='IAL')
                self.progress_bar.setValue(40)
                QApplication.processEvents()
                prob_brain_mask = brain_extraction(raw_img_ants, modality='bold', verbose=True)
                brain_mask = ants.get_mask(prob_brain_mask, low_thresh=0.5)
                masked = ants.mask_image(raw_img_ants, brain_mask)
                self.progress_bar.setValue(60)
                QApplication.processEvents()
                template_img_ants = ants.image_read(template_path, reorient='IAL')
                transformation = ants.registration(fixed=template_img_ants, 
                                                moving=masked,
                                                type_of_transform='Rigid',
                                                verbose=True)
                registered_img_ants = transformation['warpedmovout']
                self.progress_bar.setValue(80)
                QApplication.processEvents()
                registered_output_path = os.path.join(tmpdirname, 'registered.nii')
                registered_img_ants.to_file(registered_output_path)
                print("運行 MATLAB SPM 分割...")
                matlab_command = f'matlab -wait -nosplash -nodesktop -r "try, addpath(\'{matlab_script_dir}\'); run_spm_seg(\'{registered_output_path}\'); catch e, disp(getReport(e)); exit(1); end; exit(0);"'
                print(f"執行: {matlab_command}")
                result = subprocess.run(matlab_command, shell=True, capture_output=True, text=True)
                self.progress_bar.setValue(90)
                QApplication.processEvents()

                # 读取分割结果并进行裁切
                c1_path = os.path.join(os.path.dirname(registered_output_path), 'c1registered.nii')
                c2_path = os.path.join(os.path.dirname(registered_output_path), 'c2registered.nii')
                
                if os.path.exists(c1_path) and os.path.exists(c2_path):
                    c1_img = sitk.ReadImage(c1_path)
                    c2_img = sitk.ReadImage(c2_path)
                    
                    # 转换为 numpy 数组
                    c1_array = sitk.GetArrayFromImage(c1_img)
                    c2_array = sitk.GetArrayFromImage(c2_img)
                    
                    # 进行裁切 (x:17:141, y:29:200, z:30:166)
                    self.c1_array = c1_array[30:166, 29:200, 17:141]
                    self.c2_array = c2_array[30:166, 29:200, 17:141]
                    
                    # 保存裁切后的结果
                    c1_cropped = sitk.GetImageFromArray(self.c1_array)
                    c2_cropped = sitk.GetImageFromArray(self.c2_array)
                    
                    # 复制原始图像的属性
                    c1_cropped.SetOrigin(c1_img.GetOrigin())
                    c1_cropped.SetSpacing(c1_img.GetSpacing())
                    c1_cropped.SetDirection(c1_img.GetDirection())
                    
                    c2_cropped.SetOrigin(c2_img.GetOrigin())
                    c2_cropped.SetSpacing(c2_img.GetSpacing())
                    c2_cropped.SetDirection(c2_img.GetDirection())
                    
                    # 获取桌面路径
                    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
                    c1_cropped_path = os.path.join(desktop_path, 'c1_cropped.nii')
                    c2_cropped_path = os.path.join(desktop_path, 'c2_cropped.nii')
                    
                    # sitk.WriteImage(c1_cropped, c1_cropped_path)
                    # sitk.WriteImage(c2_cropped, c2_cropped_path)
                    
                    self.status_label.setText(f"✅ 預處理完成！结果已保存到桌面：\nc1_cropped.nii\nc2_cropped.nii")
                else:
                    raise FileNotFoundError("分割结果檔案未找到")

                # 預處理完成，初始化分割視圖
                self.initialize_segmentation_views()

                self.progress_bar.setValue(100)
                self.status_label.setText("✅ 預處理完成！结果已保存到桌面。")
                QApplication.processEvents()

                # 新增：快取不同模型的 Grad-CAM 結果
                self.gradcam_cache = {}

        except Exception as e:
            self.status_label.setText(f"❌ 預處理錯誤: {str(e)}")
            print(f"預處理錯誤: {e}")
        finally:
            self.progress_bar.setVisible(False)

    def initialize_segmentation_views(self):
        """初始化分割视图的滑块范围和初始显示"""
        if self.c1_array is None or self.c2_array is None:
            return

        shape = self.c1_array.shape
        # 轴向切片数量是 shape[2]，因为轴向切片是 array[:, :, index]
        max_axial = shape[2] - 1
        max_coronal = shape[1] - 1
        max_sagittal = shape[0] - 1
        mid_axial = shape[2] // 2
        mid_coronal = shape[1] // 2
        mid_sagittal = shape[0] // 2

        for map_type in ['c1', 'c2']:
            views = getattr(self, f"{map_type}_views")
            views['axial']['slider'].setMaximum(max_axial)
            views['axial']['slider'].setValue(mid_axial)
            views['coronal']['slider'].setMaximum(max_coronal)
            views['coronal']['slider'].setValue(mid_coronal)
            views['sagittal']['slider'].setMaximum(max_sagittal)
            views['sagittal']['slider'].setValue(mid_sagittal)

        # 更新当前选中视图的显示 (默认为 c1)
        self.update_segmentation_slice('axial', self.current_segmentation_map)
        self.update_segmentation_slice('coronal', self.current_segmentation_map)
        self.update_segmentation_slice('sagittal', self.current_segmentation_map)

    def switch_segmentation_map(self, map_type):
        if self.c1_array is None or self.c2_array is None:
            print("分割檔案尚未加載，无法切換視角")
            if map_type == 'c1' and not self.c1_button.isChecked():
                 self.c1_button.blockSignals(True)
                 self.c1_button.setChecked(True)
                 self.c1_button.blockSignals(False)
                 self.c2_button.blockSignals(True)
                 self.c2_button.setChecked(False)
                 self.c2_button.blockSignals(False)

            elif map_type == 'c2' and not self.c2_button.isChecked():
                 self.c2_button.blockSignals(True)
                 self.c2_button.setChecked(True)
                 self.c2_button.blockSignals(False)
                 self.c1_button.blockSignals(True)
                 self.c1_button.setChecked(False)
                 self.c1_button.blockSignals(False)

            return

        print(f"切換到顯示: {map_type}")
        self.current_segmentation_map = map_type
        is_c1 = (map_type == 'c1')

        # 直接设置可见性
        self.c1_views_widget.setVisible(is_c1)
        self.c2_views_widget.setVisible(not is_c1)

    def update_segmentation_slice(self, view_type, map_type):
        """更新指定分割图的指定视图切片"""
        if map_type == 'c1' and self.c1_array is None: return
        if map_type == 'c2' and self.c2_array is None: return

        array = self.c1_array if map_type == 'c1' else self.c2_array
        views = self.c1_views if map_type == 'c1' else self.c2_views
        view_info = views.get(view_type.lower()) 
        if not view_info:
            print(f"錯誤：找不到視角資訊 for {view_type}")
            return

        slider = view_info['slider']
        ax = view_info['ax']
        canvas = view_info['canvas']
        index = slider.value()

        ax.clear()
        slice_data = None
        title = f"{map_type.upper()} {view_type.capitalize()} Slice: {index + 1}"

        # 预处理视图的切片方向 (LPS)
        if view_type == 'axial':
            if 0 <= index < array.shape[2]:  # 使用 z 轴索引
                slice_data = np.rot90(array[:, :, index], k=1)  # LPS 方向
                slice_data = np.rot90(slice_data, k=2)  # 再旋转180度
        elif view_type == 'coronal':
            if 0 <= index < array.shape[1]:
                slice_data = np.rot90(array[:, index, :], k=1)  # LPS 方向
        elif view_type == 'sagittal':
            if 0 <= index < array.shape[0]:  # 使用 x 轴索引
                slice_data = np.rot90(array[index, :, :], k=1)  # LPS 方向

        if slice_data is not None:
            zoom = self.segmentation_zoom_factors[f"{map_type}_{view_type}"]
            h, w = slice_data.shape
            center_h, center_w = h // 2, w // 2
            new_h, new_w = int(h / zoom), int(w / zoom)
            h_start = max(0, center_h - new_h // 2)
            h_end = min(h, h_start + new_h)
            w_start = max(0, center_w - new_w // 2)
            w_end = min(w, w_start + new_w)

            if h_start >= h_end or w_start >= w_end:
                zoomed_slice = slice_data
            else:
                zoomed_slice = slice_data[h_start:h_end, w_start:w_end]

            vmin, vmax = 0, np.max(array) if np.max(array) > 0 else 1
            ax.imshow(zoomed_slice, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title, color='white', fontsize=10)
            ax.axis('off')
        else:
            ax.set_title(f"索引無效 ({view_type})", color='red')
            ax.axis('off')

        canvas.draw()

    def show_preprocessing_view(self):
        """显示预处理视图，如果未处理则执行预处理"""
        if self.img_array is None:
            self.status_label.setText("❌ 请先上傳 MRI 檔案！")
            return

        # 隐藏其他主要视图
        self.model_options_widget.setVisible(False)
        self.slices_widget.setVisible(False)
        self.upload_button.setVisible(False) # 预处理时也隐藏上传按钮
        self.run_all_button.setVisible(False) # 隐藏运行模型按钮
        self.gradcam_widget.setVisible(False)
        # 显示预处理视图
        self.preprocessing_widget.setVisible(True)
        self.status_label.setText("顯示預處理視角")

        # 如果 c1/c2 数据还不存在，则执行预处理
        if self.c1_array is None or self.c2_array is None:
            self.preprocess_mri()
        else:
            # 如果数据已存在，确保视图状态正确并更新显示
            print(f"預處理檔案已存在，顯示 {self.current_segmentation_map} 視角")
            # 确保正确的 RadioButton 被选中
            if self.current_segmentation_map == 'c1':
                if not self.c1_button.isChecked(): self.c1_button.setChecked(True)
                self.c1_views_widget.setVisible(True)
                self.c2_views_widget.setVisible(False)
            else:
                if not self.c2_button.isChecked(): self.c2_button.setChecked(True)
                self.c1_views_widget.setVisible(False)
                self.c2_views_widget.setVisible(True)

            # 更新当前视图的显示内容
            self.update_segmentation_slice('axial', self.current_segmentation_map)
            self.update_segmentation_slice('coronal', self.current_segmentation_map)
            self.update_segmentation_slice('sagittal', self.current_segmentation_map)

    def _create_segmentation_view_widget(self, map_type):
        """辅助函数：创建 c1 或 c2 的三视图小部件 (内部为水平布局)"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        views = {}
        # 視圖順序：axial, coronal, sagittal
        for view_name in ["Axial", "Coronal", "Sagittal"]:
            group = QGroupBox(f"{map_type.upper()} - {view_name}")
            group.setStyleSheet("background-color: black; color: white; border: 1px solid #555;")
            view_layout = QVBoxLayout()

            figure = plt.figure(facecolor='black')
            canvas = FigureCanvas(figure)
            ax = figure.add_subplot(111)
            ax.axis('off')

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            # 確保傳遞小寫的 view_name.lower()
            slider.valueChanged.connect(lambda value, m=map_type, v=view_name.lower(): self.update_segmentation_slice(v, m))

            # Zoom buttons
            zoom_layout = QHBoxLayout()
            zoom_in_btn = QPushButton("Zoom In")
            # 確保傳遞小寫的 view_name.lower()
            zoom_in_btn.clicked.connect(lambda checked, m=map_type, v=view_name.lower(): self.zoom_segmentation(f"{m}_{v}", 1.2))
            zoom_out_btn = QPushButton("Zoom Out")
            # 確保傳遞小寫的 view_name.lower()
            zoom_out_btn.clicked.connect(lambda checked, m=map_type, v=view_name.lower(): self.zoom_segmentation(f"{m}_{v}", 0.8))
            zoom_layout.addWidget(zoom_in_btn)
            zoom_layout.addWidget(zoom_out_btn)

            view_layout.addWidget(canvas)
            view_layout.addWidget(slider)
            view_layout.addLayout(zoom_layout)
            group.setLayout(view_layout)
            layout.addWidget(group)

            # 確保使用小寫的 view_name.lower() 作為 key
            views[view_name.lower()] = {'figure': figure, 'canvas': canvas, 'ax': ax, 'slider': slider}

        # Store references
        setattr(self, f"{map_type}_views", views)
        return widget

    def zoom_segmentation(self, view_key, factor):
        """缩放分割视图"""
        self.segmentation_zoom_factors[view_key] = max(0.1, min(5.0, self.segmentation_zoom_factors[view_key] * factor))
        map_type, view_type = view_key.split('_')
        self.update_segmentation_slice(view_type, map_type)

    def compute_gradcam(self, model_type):
        """
        根據目前的 c1_array 與選擇的模型，動態計算 Grad-CAM 結果
        """
        if self.c1_array is None:
            return None
        # 新增：先查快取
        if model_type in self.gradcam_cache:
            return self.gradcam_cache[model_type]
        input_tensor = torch.from_numpy(self.c1_array).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)

        # 選擇模型與 target layer
        if model_type == "cnn":
            model = CNN3D(num_classes=1).to(device)
            model_path = "C:/Users/user/Desktop/test/cnn.pth"
            target_layers = [model.conv6]
        elif model_type == "resnet18":
            model = resnet18(num_classes=1).to(device)
            model_path = "C:/Users/user/Desktop/test/resnet18.pth"
            target_layers = [model.layer4[-2]]
        elif model_type == "densenet121":
            model = densenet121_3d(num_classes=1).to(device)
            model_path = "C:/Users/user/Desktop/test/densenet121.pth"
            target_layers = [model.features[-4]]
        else:
            return None

        # 載入權重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 設定 GradCAM
        targets = [RegressionTarget()]
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]  # shape: [D, H, W]
            self.gradcam_cache[model_type] = grayscale_cam  # 新增：存入快取
            return grayscale_cam

    def update_gradcam_display(self, view_type=None, index=None):
        """
        切換顯示不同模型的 Grad-CAM 結果，並與原始影像疊加
        """
        if self.img_array is None:
            self.status_label.setText("❌ 請先上傳 MRI 檔案！")
            return

        gradcam_map = self.compute_gradcam(self.current_gradcam_model)
        if gradcam_map is None:
            self.status_label.setText(f"❌ 無法產生 {self.current_gradcam_model} 的 Grad-CAM 結果！")
            return

        shape = self.c1_array.shape
        for view in ["axial", "coronal", "sagittal"]:
            ax = self.gradcam_views[view]["ax"]
            canvas = self.gradcam_views[view]["canvas"]
            ax.clear()
            if view_type == view and index is not None:
                idx = index
            else:
                idx = self.gradcam_sliders[view].value()
            # 邊界檢查
            if view == "axial":
                if not (0 <= idx < shape[2]):
                    continue
                c1_slice = np.rot90(self.c1_array[:, :, idx], k=1)
                cam_slice = np.rot90(gradcam_map[:, :, idx], k=1)
                c1_slice = np.rot90(c1_slice, k=2)
                cam_slice = np.rot90(cam_slice, k=2)
            elif view == "coronal":
                if not (0 <= idx < shape[1]):
                    continue
                c1_slice = np.rot90(self.c1_array[:, idx, :], k=1)
                cam_slice = np.rot90(gradcam_map[:, idx, :], k=1)
            elif view == "sagittal":
                if not (0 <= idx < shape[0]):
                    continue
                c1_slice = np.rot90(self.c1_array[idx, :, :], k=1)
                cam_slice = np.rot90(gradcam_map[idx, :, :], k=1)
            orig_norm = (c1_slice - c1_slice.min()) / (c1_slice.max() - c1_slice.min() + 1e-8)
            cam_norm = (cam_slice - cam_slice.min()) / (cam_slice.max() - cam_slice.min() + 1e-8)
            ax.imshow(orig_norm, cmap="gray", alpha=0.5)
            ax.imshow(cam_norm, cmap="jet", alpha=0.5)
            ax.set_title(f"{self.current_gradcam_model} {view.capitalize()} Slice: {idx + 1}", color='white', fontsize=10)
            ax.axis('off')
            canvas.draw()

    def show_gradcam_view(self):
        if self.c1_array is None:
            self.status_label.setText("❌ 請先進行預處理！")
            return
        self.model_options_widget.setVisible(False)
        self.preprocessing_widget.setVisible(False)
        self.slices_widget.setVisible(False)
        self.upload_button.setVisible(False)
        self.run_all_button.setVisible(False)
        self.gradcam_widget.setVisible(True)
        self.status_label.setText("顯示 Grad-CAM 熱力圖")
        # 設定slider最大值，與 segmentation 一致
        shape = self.c1_array.shape
        # axial: z 軸 (shape[2])
        self.gradcam_sliders["axial"].setMaximum(shape[2] - 1)
        self.gradcam_sliders["axial"].setValue(shape[2] // 2)
        # coronal: y 軸 (shape[1])
        self.gradcam_sliders["coronal"].setMaximum(shape[1] - 1)
        self.gradcam_sliders["coronal"].setValue(shape[1] // 2)
        # sagittal: x 軸 (shape[0])
        self.gradcam_sliders["sagittal"].setMaximum(shape[0] - 1)
        self.gradcam_sliders["sagittal"].setValue(shape[0] // 2)
        self.set_gradcam_model("cnn")

    def set_gradcam_model(self, model_type):
        self.current_gradcam_model = model_type
        self.update_gradcam_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BrainAgePredictor()
    window.show()
    sys.exit(app.exec_()) 


