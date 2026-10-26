"""
Deep Learning Studio - Fully Configurable & Scalable Trainer
Auto-detects models in Models/ and datasets in Data sets/.
Offers flexible hyperparameters: Train/Test split %, Random Seed, and Algorithm-specific controls.
Outputs are saved into: output -> <dataset_name> -> <model_name> -> results.
"""

import os
import sys
import queue
import subprocess
import threading
import customtkinter as ctk
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset_loader import discover_datasets
from utils.model_discovery import discover_models

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class DeepLearningApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Deep Learning Studio - Flexible Model Trainer & Evaluator")
        self.geometry("1240x840")
        self.minsize(1040, 740)

        # Process management & thread-safe log queue
        self.process = None
        self.log_queue = queue.Queue()
        self.is_training = False

        self.datasets_map = {}
        self.models_map = {}
        self.current_dataset_id = "iris"
        self.current_model_id = "knn"

        # Grid layout
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_panel()

        # Load available models and datasets dynamically
        self.refresh_discovery()

        # Start log polling
        self.after(50, self._process_log_queue)

    def refresh_discovery(self):
        """Auto-detects models in Models/ and datasets in Data sets/."""
        self.datasets_map = discover_datasets(os.path.join(PROJECT_ROOT, "Data sets"))
        self.models_map = discover_models(os.path.join(PROJECT_ROOT, "Models"))

        ds_names = list(self.datasets_map.keys()) if self.datasets_map else ["No datasets found"]
        model_names = list(self.models_map.keys()) if self.models_map else ["No models found"]

        self.dataset_menu.configure(values=ds_names)
        self.model_menu.configure(values=model_names)

        if ds_names:
            self.dataset_var.set(ds_names[0])
            self._on_dataset_change(ds_names[0])

        if model_names:
            self.model_var.set(model_names[0])
            self._on_model_change(model_names[0])

    def _build_sidebar(self):
        # Using CTkScrollableFrame for sidebar so all parameter sections fit comfortably
        sidebar = ctk.CTkScrollableFrame(self, width=350, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Header Title
        title_label = ctk.CTkLabel(
            sidebar, text="🧠 DL Studio",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(padx=16, pady=(15, 2), anchor="w")

        subtitle = ctk.CTkLabel(
            sidebar, text="Fully Configurable Deep Learning Workbench",
            font=ctk.CTkFont(size=11), text_color="gray70"
        )
        subtitle.pack(padx=16, pady=(0, 10), anchor="w")

        # Hardware Badge
        hw_frame = ctk.CTkFrame(sidebar, fg_color="#1e232a", corner_radius=8)
        hw_frame.pack(padx=16, pady=(0, 14), fill="x")

        gpu_text = "🟢 RTX 5060 Laptop GPU (CUDA 12.8)"
        hw_label = ctk.CTkLabel(
            hw_frame, text=gpu_text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#58a6ff"
        )
        hw_label.pack(padx=10, pady=6, anchor="w")

        # Rescan Assets Row
        refresh_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        refresh_row.pack(padx=16, pady=(0, 8), fill="x")

        disco_lbl = ctk.CTkLabel(refresh_row, text="Plugin Assets", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray60")
        disco_lbl.pack(side="left")

        refresh_btn = ctk.CTkButton(
            refresh_row, text="🔄 Rescan",
            width=80, height=22, font=ctk.CTkFont(size=11),
            fg_color="#21262d", hover_color="#30363d",
            command=self.refresh_discovery
        )
        refresh_btn.pack(side="right")

        # ================= SECTION 1: DATASET & SPLIT =================
        sec1_frame = ctk.CTkFrame(sidebar, fg_color="#181c24", corner_radius=8)
        sec1_frame.pack(padx=16, pady=(0, 12), fill="x")

        sec1_title = ctk.CTkLabel(sec1_frame, text="📁 1. Dataset & Splitting", font=ctk.CTkFont(size=12, weight="bold"))
        sec1_title.pack(padx=12, pady=(8, 4), anchor="w")

        self.dataset_var = ctk.StringVar(value="Loading...")
        self.dataset_menu = ctk.CTkOptionMenu(
            sec1_frame, values=["Loading..."], variable=self.dataset_var,
            command=self._on_dataset_change, height=32, font=ctk.CTkFont(size=12)
        )
        self.dataset_menu.pack(padx=12, pady=(0, 10), fill="x")

        # Train/Test Split Slider
        self.split_label = ctk.CTkLabel(sec1_frame, text="Split: 80% Train / 20% Test", font=ctk.CTkFont(size=11))
        self.split_label.pack(padx=12, pady=(2, 2), anchor="w")

        self.split_slider = ctk.CTkSlider(
            sec1_frame, from_=10, to=50, number_of_steps=40,
            command=self._on_split_slider
        )
        self.split_slider.set(20)
        self.split_slider.pack(padx=12, pady=(0, 8), fill="x")

        # Random Seed Control
        seed_row = ctk.CTkFrame(sec1_frame, fg_color="transparent")
        seed_row.pack(padx=12, pady=(0, 10), fill="x")
        seed_lbl = ctk.CTkLabel(seed_row, text="Random Seed:", font=ctk.CTkFont(size=11))
        seed_lbl.pack(side="left")
        self.seed_entry = ctk.CTkEntry(seed_row, width=70, height=26, font=ctk.CTkFont(size=11))
        self.seed_entry.insert(0, "42")
        self.seed_entry.pack(side="right")

        # ================= SECTION 2: MODEL SELECTION =================
        sec2_frame = ctk.CTkFrame(sidebar, fg_color="#181c24", corner_radius=8)
        sec2_frame.pack(padx=16, pady=(0, 12), fill="x")

        sec2_title = ctk.CTkLabel(sec2_frame, text="🤖 2. Algorithm / Model", font=ctk.CTkFont(size=12, weight="bold"))
        sec2_title.pack(padx=12, pady=(8, 4), anchor="w")

        self.model_var = ctk.StringVar(value="Loading...")
        self.model_menu = ctk.CTkOptionMenu(
            sec2_frame, values=["Loading..."], variable=self.model_var,
            command=self._on_model_change, height=32, font=ctk.CTkFont(size=12)
        )
        self.model_menu.pack(padx=12, pady=(0, 10), fill="x")

        # ================= SECTION 3: MODEL HYPERPARAMETERS =================
        self.param_frame = ctk.CTkFrame(sidebar, fg_color="#181c24", corner_radius=8)
        self.param_frame.pack(padx=16, pady=(0, 14), fill="x")

        self.param_title = ctk.CTkLabel(self.param_frame, text="⚙️ 3. Hyperparameters", font=ctk.CTkFont(size=12, weight="bold"))
        self.param_title.pack(padx=12, pady=(8, 4), anchor="w")

        # --- KNN Controls ---
        self.knn_container = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.k_label = ctk.CTkLabel(self.knn_container, text="K Neighbors: 5", font=ctk.CTkFont(size=11))
        self.k_label.pack(anchor="w", pady=(0, 2))
        self.k_slider = ctk.CTkSlider(self.knn_container, from_=1, to=25, number_of_steps=24, command=self._on_k_slider)
        self.k_slider.set(5)
        self.k_slider.pack(fill="x", pady=(0, 8))

        k_opt_row = ctk.CTkFrame(self.knn_container, fg_color="transparent")
        k_opt_row.pack(fill="x", pady=(0, 4))
        w_lbl = ctk.CTkLabel(k_opt_row, text="Weights:", font=ctk.CTkFont(size=11))
        w_lbl.pack(side="left")
        self.knn_weights_var = ctk.StringVar(value="distance")
        self.knn_weights_menu = ctk.CTkOptionMenu(k_opt_row, values=["distance", "uniform"], variable=self.knn_weights_var, width=95, height=26, font=ctk.CTkFont(size=11))
        self.knn_weights_menu.pack(side="right")

        m_opt_row = ctk.CTkFrame(self.knn_container, fg_color="transparent")
        m_opt_row.pack(fill="x", pady=(4, 8))
        m_lbl = ctk.CTkLabel(m_opt_row, text="Distance:", font=ctk.CTkFont(size=11))
        m_lbl.pack(side="left")
        self.knn_metric_var = ctk.StringVar(value="euclidean")
        self.knn_metric_menu = ctk.CTkOptionMenu(k_opt_row, values=["euclidean", "manhattan"], variable=self.knn_metric_var, width=95, height=26, font=ctk.CTkFont(size=11))
        self.knn_metric_menu.pack(side="right")

        # --- SVM Controls ---
        self.svm_container = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        svm_k_row = ctk.CTkFrame(self.svm_container, fg_color="transparent")
        svm_k_row.pack(fill="x", pady=(0, 8))
        sk_lbl = ctk.CTkLabel(svm_k_row, text="Kernel:", font=ctk.CTkFont(size=11))
        sk_lbl.pack(side="left")
        self.svm_kernel_var = ctk.StringVar(value="rbf")
        self.svm_kernel_menu = ctk.CTkOptionMenu(svm_k_row, values=["rbf", "linear", "poly"], variable=self.svm_kernel_var, width=100, height=26, font=ctk.CTkFont(size=11))
        self.svm_kernel_menu.pack(side="right")

        svm_c_row = ctk.CTkFrame(self.svm_container, fg_color="transparent")
        svm_c_row.pack(fill="x", pady=(0, 8))
        sc_lbl = ctk.CTkLabel(svm_c_row, text="C Regularization:", font=ctk.CTkFont(size=11))
        sc_lbl.pack(side="left")
        self.svm_c_var = ctk.StringVar(value="1.0")
        self.svm_c_menu = ctk.CTkOptionMenu(svm_c_row, values=["0.1", "0.5", "1.0", "5.0", "10.0"], variable=self.svm_c_var, width=100, height=26, font=ctk.CTkFont(size=11))
        self.svm_c_menu.pack(side="right")

        # --- ANN Controls ---
        self.ann_container = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        self.epoch_label = ctk.CTkLabel(self.ann_container, text="Epochs: 60", font=ctk.CTkFont(size=11))
        self.epoch_label.pack(anchor="w", pady=(0, 2))
        self.epoch_slider = ctk.CTkSlider(self.ann_container, from_=5, to=100, number_of_steps=19, command=self._on_epoch_slider)
        self.epoch_slider.set(60)
        self.epoch_slider.pack(fill="x", pady=(0, 8))

        lr_row = ctk.CTkFrame(self.ann_container, fg_color="transparent")
        lr_row.pack(fill="x", pady=(0, 6))
        lr_lbl = ctk.CTkLabel(lr_row, text="Learning Rate:", font=ctk.CTkFont(size=11))
        lr_lbl.pack(side="left")
        self.ann_lr_var = ctk.StringVar(value="0.001")
        self.ann_lr_menu = ctk.CTkOptionMenu(lr_row, values=["0.01", "0.005", "0.001", "0.0005", "0.0001"], variable=self.ann_lr_var, width=100, height=26, font=ctk.CTkFont(size=11))
        self.ann_lr_menu.pack(side="right")

        opt_row = ctk.CTkFrame(self.ann_container, fg_color="transparent")
        opt_row.pack(fill="x", pady=(0, 6))
        opt_lbl = ctk.CTkLabel(opt_row, text="Optimizer:", font=ctk.CTkFont(size=11))
        opt_lbl.pack(side="left")
        self.ann_opt_var = ctk.StringVar(value="adamw")
        self.ann_opt_menu = ctk.CTkOptionMenu(opt_row, values=["adamw", "adam", "sgd"], variable=self.ann_opt_var, width=100, height=26, font=ctk.CTkFont(size=11))
        self.ann_opt_menu.pack(side="right")

        bs_row = ctk.CTkFrame(self.ann_container, fg_color="transparent")
        bs_row.pack(fill="x", pady=(0, 6))
        bs_lbl = ctk.CTkLabel(bs_row, text="Batch Size:", font=ctk.CTkFont(size=11))
        bs_lbl.pack(side="left")
        self.ann_bs_var = ctk.StringVar(value="32")
        self.ann_bs_menu = ctk.CTkOptionMenu(bs_row, values=["16", "32", "64", "128", "256"], variable=self.ann_bs_var, width=100, height=26, font=ctk.CTkFont(size=11))
        self.ann_bs_menu.pack(side="right")

        # Action Buttons
        self.train_btn = ctk.CTkButton(
            sidebar, text="▶  Start Training",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2ea043", hover_color="#238636",
            height=42, corner_radius=8,
            command=self._start_training
        )
        self.train_btn.pack(padx=16, pady=(4, 6), fill="x")

        self.stop_btn = ctk.CTkButton(
            sidebar, text="⏹  Stop Execution",
            font=ctk.CTkFont(size=12),
            fg_color="#da3633", hover_color="#b62324",
            height=32, corner_radius=8, state="disabled",
            command=self._stop_training
        )
        self.stop_btn.pack(padx=16, pady=(0, 10), fill="x")

        # Bottom Utilities
        open_folder_btn = ctk.CTkButton(
            sidebar, text="📂 Open Result Directory",
            fg_color="#21262d", hover_color="#30363d",
            height=30, corner_radius=6, font=ctk.CTkFont(size=11),
            command=self._open_output_folder
        )
        open_folder_btn.pack(padx=16, pady=(0, 6), fill="x")

        clear_btn = ctk.CTkButton(
            sidebar, text="🧹 Clear Log Output",
            fg_color="#21262d", hover_color="#30363d",
            height=30, corner_radius=6, font=ctk.CTkFont(size=11),
            command=self._clear_terminal
        )
        clear_btn.pack(padx=16, pady=(0, 14), fill="x")

    def _build_main_panel(self):
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Top Bar
        top_bar = ctk.CTkFrame(main_frame, fg_color="#161b22", height=48, corner_radius=8)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        self.status_label = ctk.CTkLabel(
            top_bar, text="Status: Ready",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#58a6ff"
        )
        self.status_label.pack(side="left", padx=16, pady=8)

        self.acc_badge = ctk.CTkLabel(
            top_bar, text="Last Accuracy: --",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#7ee787"
        )
        self.acc_badge.pack(side="right", padx=16, pady=8)

        # Tabs
        self.tabview = ctk.CTkTabview(main_frame)
        self.tabview.grid(row=1, column=0, sticky="nsew")

        self.tab_term = self.tabview.add("💻 Live Terminal Stream")
        self.tab_viz = self.tabview.add("📊 Confusion Matrix & Results")

        # Terminal Tab
        self.tab_term.grid_columnconfigure(0, weight=1)
        self.tab_term.grid_rowconfigure(0, weight=1)

        self.terminal = ctk.CTkTextbox(
            self.tab_term,
            font=ctk.CTkFont(family="Consolas", size=13),
            fg_color="#0d1117",
            text_color="#e6edf3",
            corner_radius=8,
            wrap="word"
        )
        self.terminal.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self._append_terminal(
            "=================================================================\n"
            "  DEEP LEARNING STUDIO - FULLY CONFIGURABLE WORKBENCH\n"
            "  - Flexible Train/Test Split percentage and Random Seed.\n"
            "  - Granular Hyperparameters for KNN, SVM, and GPU Neural Networks.\n"
            "  - Real-time unbuffered training metrics.\n"
            "=================================================================\n\n"
        )

        # Visuals Tab
        self.tab_viz.grid_columnconfigure(0, weight=1)
        self.tab_viz.grid_rowconfigure(1, weight=1)

        viz_top = ctk.CTkFrame(self.tab_viz, fg_color="transparent")
        viz_top.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 10))

        reload_btn = ctk.CTkButton(
            viz_top, text="🔄 Refresh Plot",
            height=30, command=self._load_cm_image
        )
        reload_btn.pack(side="left", padx=(0, 10))

        open_res_btn = ctk.CTkButton(
            viz_top, text="📂 Open Current Result Directory",
            height=30, fg_color="#1f6feb", command=self._open_output_folder
        )
        open_res_btn.pack(side="left")

        self.img_label = ctk.CTkLabel(self.tab_viz, text="No confusion matrix generated yet. Select a model and train.")
        self.img_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    # ---------------- Parameter Controls ---------------- #
    def _on_split_slider(self, val):
        test_pct = int(val)
        train_pct = 100 - test_pct
        self.split_label.configure(text=f"Split: {train_pct}% Train / {test_pct}% Test")

    def _on_k_slider(self, val):
        self.k_label.configure(text=f"K Neighbors: {int(val)}")

    def _on_epoch_slider(self, val):
        self.epoch_label.configure(text=f"Epochs: {int(val)}")

    def _on_dataset_change(self, choice):
        info = self.datasets_map.get(choice)
        if info:
            self.current_dataset_id = info["id"]
            if info["type"] == "mnist" or info["type"] == "image_folder":
                self.epoch_slider.configure(from_=2, to=20)
                self.epoch_slider.set(6)
                self.epoch_label.configure(text="Epochs: 6")
                self.ann_bs_var.set("128")
            else:
                self.epoch_slider.configure(from_=10, to=100)
                self.epoch_slider.set(60)
                self.epoch_label.configure(text="Epochs: 60")
                self.ann_bs_var.set("32")

    def _on_model_change(self, choice):
        info = self.models_map.get(choice)
        if info:
            self.current_model_id = info["id"]
        self._update_parameter_visibility()

    def _update_parameter_visibility(self):
        self.knn_container.pack_forget()
        self.svm_container.pack_forget()
        self.ann_container.pack_forget()

        m_id = self.current_model_id.lower()
        if "knn" in m_id:
            self.knn_container.pack(padx=12, pady=(4, 8), fill="x")
        elif "svm" in m_id:
            self.svm_container.pack(padx=12, pady=(4, 8), fill="x")
        elif "ann" in m_id or "mlp" in m_id or "neural" in m_id:
            self.ann_container.pack(padx=12, pady=(4, 8), fill="x")

    # ---------------- Logging ---------------- #
    def _append_terminal(self, text):
        self.terminal.configure(state="normal")
        self.terminal.insert("end", text)
        self.terminal.see("end")
        self.terminal.configure(state="disabled")

    def _clear_terminal(self):
        self.terminal.configure(state="normal")
        self.terminal.delete("1.0", "end")
        self.terminal.configure(state="disabled")

    def _process_log_queue(self):
        while not self.log_queue.empty():
            line = self.log_queue.get_nowait()
            self._append_terminal(line)

            if "[RESULT]" in line and "Accuracy" in line:
                try:
                    acc_part = line.split("Accuracy")[-1].replace(":", "").strip()
                    self.acc_badge.configure(text=f"Accuracy: {acc_part}")
                except Exception:
                    pass

        self.after(50, self._process_log_queue)

    # ---------------- Subprocess Execution ---------------- #
    def _start_training(self):
        if self.is_training:
            return

        # Parse Split & Seed
        test_fraction = self.split_slider.get() / 100.0
        try:
            seed_val = int(self.seed_entry.get())
        except ValueError:
            seed_val = 42
            self.seed_entry.delete(0, "end")
            self.seed_entry.insert(0, "42")

        python_exe = sys.executable
        runner_script = os.path.join(PROJECT_ROOT, "utils", "training_engine.py")

        cmd = [
            python_exe, "-u", runner_script,
            "--dataset", self.current_dataset_id,
            "--model", self.current_model_id,
            "--test_size", f"{test_fraction:.2f}",
            "--random_state", str(seed_val)
        ]

        # Model specific args
        m_id = self.current_model_id.lower()
        if "knn" in m_id:
            cmd.extend([
                "--k", str(int(self.k_slider.get())),
                "--knn_weights", self.knn_weights_var.get(),
                "--knn_metric", self.knn_metric_var.get()
            ])
        elif "svm" in m_id:
            cmd.extend([
                "--svm_kernel", self.svm_kernel_var.get(),
                "--svm_c", self.svm_c_var.get()
            ])
        elif "ann" in m_id or "mlp" in m_id or "neural" in m_id:
            cmd.extend([
                "--epochs", str(int(self.epoch_slider.get())),
                "--lr", self.ann_lr_var.get(),
                "--optimizer", self.ann_opt_var.get(),
                "--batch_size", self.ann_bs_var.get()
            ])

        self.is_training = True
        self.train_btn.configure(state="disabled", fg_color="#555555")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(
            text=f"Status: Training {self.current_model_id.upper()} on {self.current_dataset_id.upper()}...",
            text_color="#e3b341"
        )
        self.tabview.set("💻 Live Terminal Stream")
        self._append_terminal(f"\n>>> Running: {' '.join(cmd)}\n\n")

        thread = threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True)
        thread.start()

    def _run_subprocess(self, cmd):
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            for line in iter(self.process.stdout.readline, ''):
                self.log_queue.put(line)

            self.process.stdout.close()
            return_code = self.process.wait()

        except Exception as e:
            self.log_queue.put(f"\n[ERROR] Subprocess execution failed: {e}\n")
            return_code = -1

        self.is_training = False
        self.after(0, self._on_training_finished, return_code)

    def _on_training_finished(self, return_code):
        self.train_btn.configure(state="normal", fg_color="#2ea043")
        self.stop_btn.configure(state="disabled")

        if return_code == 0:
            self.status_label.configure(text="Status: Completed Successfully! ✅", text_color="#7ee787")
            self._load_cm_image()
        else:
            self.status_label.configure(text=f"Status: Finished with code {return_code}", text_color="#f85149")

    def _stop_training(self):
        if self.process and self.is_training:
            try:
                self.process.terminate()
                self._append_terminal("\n[STOPPED] Training process terminated by user.\n")
                self.status_label.configure(text="Status: Terminated by User", text_color="#da3633")
            except Exception as e:
                self._append_terminal(f"\n[ERROR] Could not terminate: {e}\n")

    def _load_cm_image(self):
        specific_path = os.path.join(PROJECT_ROOT, "output", self.current_dataset_id, self.current_model_id, "confusion_matrix.png")
        latest_path = os.path.join(PROJECT_ROOT, "output", "last_run_cm.png")

        cm_path = specific_path if os.path.exists(specific_path) else latest_path

        if os.path.exists(cm_path):
            try:
                pil_img = Image.open(cm_path)
                max_w, max_h = 760, 530
                pil_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
                self.img_label.configure(image=ctk_img, text="")
            except Exception as e:
                self.img_label.configure(image=None, text=f"Error displaying image: {e}")
        else:
            self.img_label.configure(image=None, text="No confusion matrix generated yet.")

    def _open_output_folder(self):
        folder_path = os.path.join(PROJECT_ROOT, "output", self.current_dataset_id, self.current_model_id)
        if not os.path.exists(folder_path):
            folder_path = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(folder_path, exist_ok=True)
        os.startfile(os.path.abspath(folder_path))


if __name__ == "__main__":
    app = DeepLearningApp()
    app.mainloop()
