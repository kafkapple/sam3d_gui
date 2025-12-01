"""
SAM 3D GUI Application
Interactive GUI for video segmentation and 3D reconstruction
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
from typing import Optional, List
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sam3d_processor import SAM3DProcessor, TrackingResult
from image_cropper import (
    find_image_pairs, batch_crop_images, preview_crop,
    BatchCropResult, CropResult
)


class SAM3DGUI:
    """Main GUI application for SAM 3D object segmentation"""

    def __init__(self, root):
        self.root = root
        self.root.title("SAM 3D Object Segmentation & Reconstruction")
        self.root.geometry("1400x900")

        # Initialize processor
        self.processor = SAM3DProcessor()

        # State variables
        self.current_video_path = None
        self.current_frames = []
        self.current_frame_idx = 0
        self.video_info = None
        self.tracking_result: Optional[TrackingResult] = None
        self.reconstruction_3d = None

        # Image Crop tab state
        self.crop_input_dir = None
        self.crop_image_pairs = []
        self.crop_preview_idx = 0

        # Setup UI
        self.setup_ui()

        # Default data directory (one level above project root)
        project_root = Path(__file__).parent.parent
        self.data_dir = str(project_root.parent / "data" / "markerless_mouse")
        self.output_dir = str(project_root / "outputs")

    def setup_ui(self):
        """Setup the user interface"""
        # Main container with Notebook tabs
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Tab 1: Video Processing (original functionality)
        self.video_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.video_tab, text="Video Processing")

        # Tab 2: Image Crop
        self.crop_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.crop_tab, text="Image Crop")

        # Setup individual tabs
        self.setup_video_tab()
        self.setup_crop_tab()

    def setup_video_tab(self):
        """Setup the video processing tab (original UI)"""
        main_frame = self.video_tab
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # ===== Left Panel: Controls =====
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Data directory selection
        ttk.Label(control_frame, text="Data Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.data_dir_var = tk.StringVar(value=self.data_dir)
        ttk.Entry(control_frame, textvariable=self.data_dir_var, width=40).grid(row=1, column=0, pady=5)
        ttk.Button(control_frame, text="Browse...", command=self.browse_data_dir).grid(row=1, column=1, padx=5)

        # Video file selection
        ttk.Label(control_frame, text="Video Files:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.video_listbox = tk.Listbox(control_frame, height=8, width=50)
        self.video_listbox.grid(row=3, column=0, columnspan=2, pady=5)
        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_select)

        ttk.Button(control_frame, text="Refresh List", command=self.refresh_video_list).grid(row=4, column=0, pady=5)
        ttk.Button(control_frame, text="Load Video", command=self.load_selected_video).grid(row=4, column=1, pady=5)

        # Video info
        ttk.Label(control_frame, text="Video Info:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.info_text = scrolledtext.ScrolledText(control_frame, height=6, width=50)
        self.info_text.grid(row=6, column=0, columnspan=2, pady=5)

        # Processing parameters
        ttk.Separator(control_frame, orient='horizontal').grid(row=7, column=0, columnspan=2, sticky='ew', pady=10)
        ttk.Label(control_frame, text="Processing Parameters", font=('Arial', 10, 'bold')).grid(row=8, column=0, columnspan=2, pady=5)

        # Start time
        ttk.Label(control_frame, text="Start Time (s):").grid(row=9, column=0, sticky=tk.W)
        self.start_time_var = tk.DoubleVar(value=0.0)
        ttk.Entry(control_frame, textvariable=self.start_time_var, width=15).grid(row=9, column=1, sticky=tk.W)

        # Duration
        ttk.Label(control_frame, text="Duration (s):").grid(row=10, column=0, sticky=tk.W)
        self.duration_var = tk.DoubleVar(value=3.0)
        ttk.Entry(control_frame, textvariable=self.duration_var, width=15).grid(row=10, column=1, sticky=tk.W)

        # Motion threshold
        ttk.Label(control_frame, text="Motion Threshold:").grid(row=11, column=0, sticky=tk.W)
        self.motion_threshold_var = tk.DoubleVar(value=50.0)
        ttk.Entry(control_frame, textvariable=self.motion_threshold_var, width=15).grid(row=11, column=1, sticky=tk.W)

        # Segmentation method
        ttk.Label(control_frame, text="Segmentation:").grid(row=12, column=0, sticky=tk.W)
        self.seg_method_var = tk.StringVar(value='contour')
        seg_combo = ttk.Combobox(control_frame, textvariable=self.seg_method_var, width=15)
        seg_combo['values'] = ('contour', 'simple_threshold', 'grabcut')
        seg_combo.grid(row=12, column=1, sticky=tk.W)

        # Output directory
        ttk.Label(control_frame, text="Output Directory:").grid(row=13, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar(value=self.output_dir)
        ttk.Entry(control_frame, textvariable=self.output_dir_var, width=40).grid(row=14, column=0, pady=5)
        ttk.Button(control_frame, text="Browse...", command=self.browse_output_dir).grid(row=14, column=1, padx=5)

        # Action buttons
        ttk.Separator(control_frame, orient='horizontal').grid(row=15, column=0, columnspan=2, sticky='ew', pady=10)

        self.process_btn = ttk.Button(
            control_frame,
            text="Process Video Segment",
            command=self.process_video_segment,
            state='disabled'
        )
        self.process_btn.grid(row=16, column=0, columnspan=2, pady=10)

        self.reconstruct_btn = ttk.Button(
            control_frame,
            text="3D Reconstruction",
            command=self.reconstruct_3d,
            state='disabled'
        )
        self.reconstruct_btn.grid(row=17, column=0, columnspan=2, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=18, column=0, columnspan=2, pady=10, sticky='ew')

        # ===== Center Panel: Video Display =====
        video_frame = ttk.LabelFrame(main_frame, text="Video Preview", padding="10")
        video_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        main_frame.columnconfigure(1, weight=1)

        # Canvas for video display
        self.video_canvas = tk.Canvas(video_frame, width=640, height=480, bg='black')
        self.video_canvas.pack(pady=10)

        # Frame controls
        frame_control = ttk.Frame(video_frame)
        frame_control.pack(pady=5)

        self.prev_frame_btn = ttk.Button(frame_control, text="<< Prev", command=self.prev_frame, state='disabled')
        self.prev_frame_btn.grid(row=0, column=0, padx=5)

        self.frame_label = ttk.Label(frame_control, text="Frame: 0 / 0")
        self.frame_label.grid(row=0, column=1, padx=10)

        self.next_frame_btn = ttk.Button(frame_control, text="Next >>", command=self.next_frame, state='disabled')
        self.next_frame_btn.grid(row=0, column=2, padx=5)

        # ===== Right Panel: Results =====
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=0, column=2, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Results text
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20, width=50)
        self.results_text.pack(pady=10, fill='both', expand=True)

        # Export buttons
        export_frame = ttk.Frame(results_frame)
        export_frame.pack(pady=10)

        ttk.Button(export_frame, text="Export PLY", command=lambda: self.export_mesh('ply')).grid(row=0, column=0, padx=5)
        ttk.Button(export_frame, text="Export OBJ", command=lambda: self.export_mesh('obj')).grid(row=0, column=1, padx=5)
        ttk.Button(export_frame, text="View 3D", command=self.view_3d).grid(row=0, column=2, padx=5)

        # ===== Bottom Panel: Logs =====
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="10")
        log_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill='both', expand=True)

        # Initialize
        self.refresh_video_list()

    def log(self, message: str):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()

    def browse_data_dir(self):
        """Browse for data directory"""
        directory = filedialog.askdirectory(initialdir=self.data_dir_var.get())
        if directory:
            self.data_dir_var.set(directory)
            self.refresh_video_list()

    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)

    def refresh_video_list(self):
        """Refresh the list of available videos"""
        self.video_listbox.delete(0, tk.END)
        data_dir = Path(self.data_dir_var.get())

        if not data_dir.exists():
            self.log(f"Directory not found: {data_dir}")
            return

        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(data_dir.rglob(f'*{ext}'))

        # Add to listbox
        for video_file in sorted(video_files):
            rel_path = video_file.relative_to(data_dir)
            self.video_listbox.insert(tk.END, str(rel_path))

        self.log(f"Found {len(video_files)} video files")

    def on_video_select(self, event):
        """Handle video selection"""
        selection = self.video_listbox.curselection()
        if selection:
            self.process_btn['state'] = 'normal'

    def load_selected_video(self):
        """Load the selected video"""
        selection = self.video_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a video file")
            return

        rel_path = self.video_listbox.get(selection[0])
        video_path = Path(self.data_dir_var.get()) / rel_path
        self.current_video_path = str(video_path)

        # Get video info
        self.video_info = self.processor.get_video_info(self.current_video_path)

        # Display info
        info_str = f"Path: {video_path.name}\n"
        info_str += f"Resolution: {self.video_info['width']}x{self.video_info['height']}\n"
        info_str += f"FPS: {self.video_info['fps']:.2f}\n"
        info_str += f"Frames: {self.video_info['frame_count']}\n"
        info_str += f"Duration: {self.video_info['duration']:.2f}s\n"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_str)

        # Load preview frames
        self.load_preview_frames()

        self.log(f"Loaded video: {video_path.name}")

    def load_preview_frames(self, num_frames=10):
        """Load preview frames from current video"""
        if not self.current_video_path:
            return

        self.current_frames = self.processor.extract_frames(
            self.current_video_path,
            start_frame=int(self.start_time_var.get() * self.video_info['fps']),
            num_frames=num_frames,
            stride=1
        )

        self.current_frame_idx = 0

        if self.current_frames:
            self.display_current_frame()
            self.prev_frame_btn['state'] = 'normal'
            self.next_frame_btn['state'] = 'normal'
            self.frame_label['text'] = f"Frame: {self.current_frame_idx + 1} / {len(self.current_frames)}"

    def display_current_frame(self):
        """Display current frame on canvas"""
        if not self.current_frames:
            return

        frame = self.current_frames[self.current_frame_idx]

        # Resize to fit canvas
        canvas_width = 640
        canvas_height = 480
        h, w = frame.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        frame_resized = cv2.resize(frame, (new_w, new_h))

        # Convert to PhotoImage
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image)

        # Display on canvas
        self.video_canvas.delete("all")
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2
        self.video_canvas.create_image(x, y, anchor=tk.NW, image=photo)
        self.video_canvas.image = photo  # Keep reference

    def prev_frame(self):
        """Show previous frame"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.display_current_frame()
            self.frame_label['text'] = f"Frame: {self.current_frame_idx + 1} / {len(self.current_frames)}"

    def next_frame(self):
        """Show next frame"""
        if self.current_frame_idx < len(self.current_frames) - 1:
            self.current_frame_idx += 1
            self.display_current_frame()
            self.frame_label['text'] = f"Frame: {self.current_frame_idx + 1} / {len(self.current_frames)}"

    def process_video_segment(self):
        """Process video segment with tracking and motion detection"""
        if not self.current_video_path:
            messagebox.showwarning("No Video", "Please load a video first")
            return

        # Disable button and show progress
        self.process_btn['state'] = 'disabled'
        self.progress.start()

        def process_thread():
            try:
                self.log("Processing video segment...")

                # Process
                tracking_result, reconstruction = self.processor.process_video_segment(
                    video_path=self.current_video_path,
                    start_time=self.start_time_var.get(),
                    duration=self.duration_var.get(),
                    output_dir=self.output_dir_var.get(),
                    segmentation_method=self.seg_method_var.get(),
                    motion_threshold=self.motion_threshold_var.get()
                )

                self.tracking_result = tracking_result
                self.reconstruction_3d = reconstruction

                # Display results
                results = f"=== Tracking Results ===\n"
                results += f"Duration: {tracking_result.duration_seconds:.2f}s\n"
                results += f"Frames analyzed: {len(tracking_result.segments)}\n"
                results += f"Motion detected: {tracking_result.motion_detected}\n"

                if tracking_result.segments:
                    avg_area = np.mean([s.area for s in tracking_result.segments])
                    results += f"Average object area: {avg_area:.0f} pixels²\n"

                    # Motion statistics
                    if len(tracking_result.segments) > 1:
                        displacements = []
                        for i in range(1, len(tracking_result.segments)):
                            dx = tracking_result.segments[i].center[0] - tracking_result.segments[i-1].center[0]
                            dy = tracking_result.segments[i].center[1] - tracking_result.segments[i-1].center[1]
                            disp = np.sqrt(dx**2 + dy**2)
                            displacements.append(disp)

                        results += f"Max displacement: {max(displacements):.1f} pixels\n"
                        results += f"Avg displacement: {np.mean(displacements):.1f} pixels\n"

                if reconstruction:
                    results += f"\n=== 3D Reconstruction ===\n"
                    results += f"Status: Complete\n"
                    results += f"Output directory: {self.output_dir_var.get()}\n"
                    self.reconstruct_btn['state'] = 'normal'

                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(1.0, results)

                self.log("Processing complete!")

            except Exception as e:
                self.log(f"Error: {str(e)}")
                messagebox.showerror("Processing Error", str(e))

            finally:
                self.progress.stop()
                self.process_btn['state'] = 'normal'

        # Run in separate thread
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()

    def reconstruct_3d(self):
        """Perform 3D reconstruction on selected frame"""
        if not self.current_frames:
            messagebox.showwarning("No Frames", "Please load video frames first")
            return

        self.log("Starting 3D reconstruction...")
        self.reconstruct_btn['state'] = 'disabled'
        self.progress.start()

        def reconstruct_thread():
            try:
                # Use current frame
                frame = self.current_frames[self.current_frame_idx]

                # Segment
                mask = self.processor.segment_object_interactive(
                    frame,
                    method=self.seg_method_var.get()
                )

                # Reconstruct
                self.reconstruction_3d = self.processor.reconstruct_3d(frame, mask)

                # Save
                output_dir = self.output_dir_var.get()
                os.makedirs(output_dir, exist_ok=True)

                # Export mesh
                mesh_path = os.path.join(output_dir, f'reconstruction_frame_{self.current_frame_idx}')
                self.processor.export_mesh(self.reconstruction_3d, mesh_path, format='ply')

                # Save visualization
                vis = self.processor.visualize_mask_overlay(frame, mask)
                Image.fromarray(vis).save(os.path.join(output_dir, f'mask_frame_{self.current_frame_idx}.png'))

                self.log(f"3D reconstruction saved to {output_dir}")

            except Exception as e:
                self.log(f"Reconstruction error: {str(e)}")
                messagebox.showerror("Reconstruction Error", str(e))

            finally:
                self.progress.stop()
                self.reconstruct_btn['state'] = 'normal'

        thread = threading.Thread(target=reconstruct_thread, daemon=True)
        thread.start()

    def export_mesh(self, format='ply'):
        """Export mesh in specified format"""
        if not self.reconstruction_3d:
            messagebox.showwarning("No Reconstruction", "Please perform 3D reconstruction first")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=f'.{format}',
            filetypes=[(f"{format.upper()} files", f"*.{format}")]
        )

        if file_path:
            try:
                self.processor.export_mesh(self.reconstruction_3d, file_path, format=format)
                self.log(f"Mesh exported to {file_path}")
                messagebox.showinfo("Export Complete", f"Mesh saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def view_3d(self):
        """Open 3D visualization"""
        if not self.reconstruction_3d:
            messagebox.showwarning("No Reconstruction", "Please perform 3D reconstruction first")
            return

        # Save temporary PLY
        temp_path = "/tmp/temp_reconstruction.ply"
        self.processor.export_mesh(self.reconstruction_3d, temp_path, format='ply')

        self.log(f"3D model saved to {temp_path}")
        self.log("To view: Use a 3D viewer like MeshLab or online viewer")
        messagebox.showinfo(
            "3D Visualization",
            f"3D model saved to:\n{temp_path}\n\nUse MeshLab, CloudCompare, or online viewers to visualize."
        )

    # ========== Image Crop Tab Methods ==========

    def setup_crop_tab(self):
        """Setup the Image Crop tab UI"""
        main_frame = self.crop_tab
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # ===== Left Panel: Controls =====
        control_frame = ttk.LabelFrame(main_frame, text="Crop Settings", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Input directory
        ttk.Label(control_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.crop_input_dir_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.crop_input_dir_var, width=40).grid(row=1, column=0, pady=2)
        ttk.Button(control_frame, text="Browse...", command=self.browse_crop_input_dir).grid(row=1, column=1, padx=5)

        # Output directory
        ttk.Label(control_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.crop_output_dir_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.crop_output_dir_var, width=40).grid(row=3, column=0, pady=2)
        ttk.Button(control_frame, text="Browse...", command=self.browse_crop_output_dir).grid(row=3, column=1, padx=5)

        # Pattern settings
        ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky='ew', pady=10)
        ttk.Label(control_frame, text="Pattern Settings", font=('Arial', 10, 'bold')).grid(row=5, column=0, columnspan=2, pady=5)

        ttk.Label(control_frame, text="RGB Pattern:").grid(row=6, column=0, sticky=tk.W)
        self.crop_rgb_pattern_var = tk.StringVar(value="rgb")
        ttk.Entry(control_frame, textvariable=self.crop_rgb_pattern_var, width=20).grid(row=6, column=1, sticky=tk.W)

        ttk.Label(control_frame, text="Mask Pattern:").grid(row=7, column=0, sticky=tk.W)
        self.crop_mask_pattern_var = tk.StringVar(value="mask")
        ttk.Entry(control_frame, textvariable=self.crop_mask_pattern_var, width=20).grid(row=7, column=1, sticky=tk.W)

        # Crop settings
        ttk.Separator(control_frame, orient='horizontal').grid(row=8, column=0, columnspan=2, sticky='ew', pady=10)
        ttk.Label(control_frame, text="Crop Settings", font=('Arial', 10, 'bold')).grid(row=9, column=0, columnspan=2, pady=5)

        ttk.Label(control_frame, text="Padding (px):").grid(row=10, column=0, sticky=tk.W)
        self.crop_padding_var = tk.IntVar(value=0)
        ttk.Entry(control_frame, textvariable=self.crop_padding_var, width=10).grid(row=10, column=1, sticky=tk.W)

        ttk.Label(control_frame, text="Padding Ratio:").grid(row=11, column=0, sticky=tk.W)
        self.crop_padding_ratio_var = tk.DoubleVar(value=0.1)
        ttk.Entry(control_frame, textvariable=self.crop_padding_ratio_var, width=10).grid(row=11, column=1, sticky=tk.W)

        self.crop_square_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Square Crop", variable=self.crop_square_var).grid(row=12, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Resize settings
        ttk.Label(control_frame, text="Resize (optional):").grid(row=13, column=0, sticky=tk.W)
        resize_frame = ttk.Frame(control_frame)
        resize_frame.grid(row=13, column=1, sticky=tk.W)
        self.crop_resize_w_var = tk.StringVar(value="")
        self.crop_resize_h_var = tk.StringVar(value="")
        ttk.Entry(resize_frame, textvariable=self.crop_resize_w_var, width=6).pack(side=tk.LEFT)
        ttk.Label(resize_frame, text=" x ").pack(side=tk.LEFT)
        ttk.Entry(resize_frame, textvariable=self.crop_resize_h_var, width=6).pack(side=tk.LEFT)

        # Batch limit
        ttk.Separator(control_frame, orient='horizontal').grid(row=14, column=0, columnspan=2, sticky='ew', pady=10)
        ttk.Label(control_frame, text="Batch Options", font=('Arial', 10, 'bold')).grid(row=15, column=0, columnspan=2, pady=5)

        ttk.Label(control_frame, text="Max Images:").grid(row=16, column=0, sticky=tk.W)
        self.crop_max_images_var = tk.StringVar(value="")
        ttk.Entry(control_frame, textvariable=self.crop_max_images_var, width=10).grid(row=16, column=1, sticky=tk.W)
        ttk.Label(control_frame, text="(empty = all)").grid(row=17, column=1, sticky=tk.W)

        # Action buttons
        ttk.Separator(control_frame, orient='horizontal').grid(row=18, column=0, columnspan=2, sticky='ew', pady=10)

        ttk.Button(control_frame, text="Scan Directory", command=self.scan_crop_directory).grid(row=19, column=0, columnspan=2, pady=5)

        self.crop_preview_btn = ttk.Button(control_frame, text="Preview Crop", command=self.preview_crop_result, state='disabled')
        self.crop_preview_btn.grid(row=20, column=0, columnspan=2, pady=5)

        self.crop_run_btn = ttk.Button(control_frame, text="Run Batch Crop", command=self.run_batch_crop, state='disabled')
        self.crop_run_btn.grid(row=21, column=0, columnspan=2, pady=5)

        # Progress bar
        self.crop_progress = ttk.Progressbar(control_frame, mode='determinate')
        self.crop_progress.grid(row=22, column=0, columnspan=2, pady=10, sticky='ew')

        # ===== Center Panel: Preview =====
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        preview_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Preview canvas for original and cropped images
        preview_container = ttk.Frame(preview_frame)
        preview_container.pack(fill='both', expand=True)

        # Original image
        orig_frame = ttk.LabelFrame(preview_container, text="Original")
        orig_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5)
        self.crop_original_canvas = tk.Canvas(orig_frame, width=320, height=320, bg='gray20')
        self.crop_original_canvas.pack(pady=5)

        # Cropped image
        crop_frame = ttk.LabelFrame(preview_container, text="Cropped")
        crop_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5)
        self.crop_result_canvas = tk.Canvas(crop_frame, width=320, height=320, bg='gray20')
        self.crop_result_canvas.pack(pady=5)

        # Preview navigation
        nav_frame = ttk.Frame(preview_frame)
        nav_frame.pack(pady=5)

        self.crop_prev_btn = ttk.Button(nav_frame, text="<< Prev", command=self.prev_crop_preview, state='disabled')
        self.crop_prev_btn.grid(row=0, column=0, padx=5)

        self.crop_preview_label = ttk.Label(nav_frame, text="0 / 0")
        self.crop_preview_label.grid(row=0, column=1, padx=10)

        self.crop_next_btn = ttk.Button(nav_frame, text="Next >>", command=self.next_crop_preview, state='disabled')
        self.crop_next_btn.grid(row=0, column=2, padx=5)

        # ===== Right Panel: Results =====
        results_frame = ttk.LabelFrame(main_frame, text="Results & Log", padding="10")
        results_frame.grid(row=0, column=2, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # File list
        ttk.Label(results_frame, text="Found Image Pairs:").pack(anchor=tk.W)
        self.crop_files_listbox = tk.Listbox(results_frame, height=15, width=45)
        self.crop_files_listbox.pack(pady=5, fill='both', expand=True)
        self.crop_files_listbox.bind('<<ListboxSelect>>', self.on_crop_file_select)

        # Log
        ttk.Label(results_frame, text="Log:").pack(anchor=tk.W, pady=(10, 0))
        self.crop_log_text = scrolledtext.ScrolledText(results_frame, height=10, width=45)
        self.crop_log_text.pack(pady=5, fill='both', expand=True)

    def crop_log(self, message: str):
        """Add message to crop tab log"""
        self.crop_log_text.insert(tk.END, f"{message}\n")
        self.crop_log_text.see(tk.END)
        self.root.update()

    def browse_crop_input_dir(self):
        """Browse for crop input directory"""
        directory = filedialog.askdirectory(initialdir=self.crop_input_dir_var.get() or os.path.expanduser("~"))
        if directory:
            self.crop_input_dir_var.set(directory)
            # Auto-set output directory
            if not self.crop_output_dir_var.get():
                self.crop_output_dir_var.set(str(Path(directory).parent / f"{Path(directory).name}_cropped"))

    def browse_crop_output_dir(self):
        """Browse for crop output directory"""
        directory = filedialog.askdirectory(initialdir=self.crop_output_dir_var.get() or os.path.expanduser("~"))
        if directory:
            self.crop_output_dir_var.set(directory)

    def scan_crop_directory(self):
        """Scan input directory for RGB/Mask pairs"""
        input_dir = self.crop_input_dir_var.get()
        if not input_dir:
            messagebox.showwarning("No Directory", "Please select an input directory")
            return

        if not os.path.exists(input_dir):
            messagebox.showerror("Directory Not Found", f"Directory does not exist: {input_dir}")
            return

        self.crop_log(f"Scanning {input_dir}...")

        # Find image pairs
        self.crop_image_pairs = find_image_pairs(
            input_dir,
            rgb_pattern=self.crop_rgb_pattern_var.get(),
            mask_pattern=self.crop_mask_pattern_var.get()
        )

        # Update file list
        self.crop_files_listbox.delete(0, tk.END)
        for rgb_path, mask_path in self.crop_image_pairs:
            self.crop_files_listbox.insert(tk.END, f"{rgb_path.name}")

        self.crop_log(f"Found {len(self.crop_image_pairs)} image pairs")

        # Enable buttons if pairs found
        if self.crop_image_pairs:
            self.crop_preview_btn['state'] = 'normal'
            self.crop_run_btn['state'] = 'normal'
            self.crop_prev_btn['state'] = 'normal'
            self.crop_next_btn['state'] = 'normal'
            self.crop_preview_idx = 0
            self.update_crop_preview()
        else:
            self.crop_preview_btn['state'] = 'disabled'
            self.crop_run_btn['state'] = 'disabled'
            self.crop_prev_btn['state'] = 'disabled'
            self.crop_next_btn['state'] = 'disabled'
            messagebox.showinfo("No Pairs Found",
                f"No RGB/Mask pairs found with patterns:\n"
                f"RGB: '{self.crop_rgb_pattern_var.get()}'\n"
                f"Mask: '{self.crop_mask_pattern_var.get()}'\n\n"
                "Try adjusting the patterns or check directory structure.")

    def on_crop_file_select(self, event):
        """Handle file selection in crop list"""
        selection = self.crop_files_listbox.curselection()
        if selection:
            self.crop_preview_idx = selection[0]
            self.update_crop_preview()

    def update_crop_preview(self):
        """Update crop preview for current index"""
        if not self.crop_image_pairs:
            return

        idx = self.crop_preview_idx
        total = len(self.crop_image_pairs)
        self.crop_preview_label['text'] = f"{idx + 1} / {total}"

        rgb_path, mask_path = self.crop_image_pairs[idx]

        # Get resize settings
        resize = None
        try:
            w = self.crop_resize_w_var.get().strip()
            h = self.crop_resize_h_var.get().strip()
            if w and h:
                resize = (int(w), int(h))
        except ValueError:
            pass

        # Preview the crop
        result = preview_crop(
            str(rgb_path),
            str(mask_path),
            padding=self.crop_padding_var.get(),
            padding_ratio=self.crop_padding_ratio_var.get(),
            square=self.crop_square_var.get()
        )

        # Display original image with bbox
        self.display_crop_original(str(rgb_path), str(mask_path), result)

        # Display cropped result
        if result:
            rgb_cropped, mask_cropped, bbox = result
            self.display_crop_result(rgb_cropped, mask_cropped)
        else:
            self.crop_result_canvas.delete("all")
            self.crop_result_canvas.create_text(160, 160, text="No mask content", fill="white")

    def display_crop_original(self, rgb_path: str, mask_path: str, crop_result):
        """Display original image with bounding box overlay"""
        # Read and display original RGB
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            return

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Draw bounding box if available
        if crop_result:
            _, _, bbox = crop_result
            x, y, w, h = bbox
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Resize for display
        canvas_size = 320
        h_img, w_img = rgb.shape[:2]
        scale = min(canvas_size / w_img, canvas_size / h_img)
        new_w, new_h = int(w_img * scale), int(h_img * scale)
        rgb_resized = cv2.resize(rgb, (new_w, new_h))

        # Convert to PhotoImage
        image = Image.fromarray(rgb_resized)
        photo = ImageTk.PhotoImage(image)

        # Display
        self.crop_original_canvas.delete("all")
        x_offset = (canvas_size - new_w) // 2
        y_offset = (canvas_size - new_h) // 2
        self.crop_original_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
        self.crop_original_canvas.image = photo

    def display_crop_result(self, rgb_cropped: np.ndarray, mask_cropped: np.ndarray):
        """Display cropped result (RGB with mask overlay)"""
        # Create overlay
        rgb_display = rgb_cropped.copy()

        # Apply mask overlay (semi-transparent green where mask is active)
        mask_3ch = np.stack([mask_cropped] * 3, axis=-1)
        overlay = np.zeros_like(rgb_display)
        overlay[:, :, 1] = 100  # Green tint
        rgb_display = np.where(mask_3ch > 127, cv2.addWeighted(rgb_display, 0.7, overlay, 0.3, 0), rgb_display)

        # Resize for display
        canvas_size = 320
        h_img, w_img = rgb_display.shape[:2]
        scale = min(canvas_size / w_img, canvas_size / h_img)
        new_w, new_h = int(w_img * scale), int(h_img * scale)
        rgb_resized = cv2.resize(rgb_display, (new_w, new_h))

        # Convert to PhotoImage
        image = Image.fromarray(rgb_resized)
        photo = ImageTk.PhotoImage(image)

        # Display
        self.crop_result_canvas.delete("all")
        x_offset = (canvas_size - new_w) // 2
        y_offset = (canvas_size - new_h) // 2
        self.crop_result_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
        self.crop_result_canvas.image = photo

    def prev_crop_preview(self):
        """Show previous crop preview"""
        if self.crop_preview_idx > 0:
            self.crop_preview_idx -= 1
            self.crop_files_listbox.selection_clear(0, tk.END)
            self.crop_files_listbox.selection_set(self.crop_preview_idx)
            self.crop_files_listbox.see(self.crop_preview_idx)
            self.update_crop_preview()

    def next_crop_preview(self):
        """Show next crop preview"""
        if self.crop_preview_idx < len(self.crop_image_pairs) - 1:
            self.crop_preview_idx += 1
            self.crop_files_listbox.selection_clear(0, tk.END)
            self.crop_files_listbox.selection_set(self.crop_preview_idx)
            self.crop_files_listbox.see(self.crop_preview_idx)
            self.update_crop_preview()

    def preview_crop_result(self):
        """Preview crop for currently selected image"""
        if not self.crop_image_pairs:
            return
        self.update_crop_preview()

    def run_batch_crop(self):
        """Run batch crop operation"""
        if not self.crop_image_pairs:
            messagebox.showwarning("No Images", "Please scan directory first")
            return

        output_dir = self.crop_output_dir_var.get()
        if not output_dir:
            messagebox.showwarning("No Output", "Please specify output directory")
            return

        # Get max images limit
        max_images = None
        try:
            max_str = self.crop_max_images_var.get().strip()
            if max_str:
                max_images = int(max_str)
        except ValueError:
            messagebox.showerror("Invalid Input", "Max images must be a number")
            return

        # Get resize settings
        resize = None
        try:
            w = self.crop_resize_w_var.get().strip()
            h = self.crop_resize_h_var.get().strip()
            if w and h:
                resize = (int(w), int(h))
        except ValueError:
            messagebox.showerror("Invalid Input", "Resize dimensions must be numbers")
            return

        # Limit pairs if max_images specified
        pairs_to_process = self.crop_image_pairs
        if max_images is not None and max_images > 0:
            pairs_to_process = self.crop_image_pairs[:max_images]

        # Disable button during processing
        self.crop_run_btn['state'] = 'disabled'
        self.crop_progress['maximum'] = len(pairs_to_process)
        self.crop_progress['value'] = 0

        def crop_thread():
            try:
                self.crop_log(f"Starting batch crop of {len(pairs_to_process)} images...")

                def progress_callback(current, total, msg):
                    self.crop_progress['value'] = current
                    self.crop_log(f"[{current}/{total}] {msg}")
                    self.root.update()

                # Create output directories
                output_path = Path(output_dir)
                rgb_pattern = self.crop_rgb_pattern_var.get()
                mask_pattern = self.crop_mask_pattern_var.get()

                output_rgb_dir = output_path / rgb_pattern
                output_mask_dir = output_path / mask_pattern
                output_rgb_dir.mkdir(parents=True, exist_ok=True)
                output_mask_dir.mkdir(parents=True, exist_ok=True)

                successful = 0
                failed = 0

                for i, (rgb_file, mask_file) in enumerate(pairs_to_process):
                    progress_callback(i + 1, len(pairs_to_process), f"Processing {rgb_file.name}")

                    from image_cropper import crop_image_pair

                    output_rgb = output_rgb_dir / rgb_file.name
                    output_mask = output_mask_dir / mask_file.name

                    result = crop_image_pair(
                        str(rgb_file),
                        str(mask_file),
                        str(output_rgb),
                        str(output_mask),
                        padding=self.crop_padding_var.get(),
                        padding_ratio=self.crop_padding_ratio_var.get(),
                        square=self.crop_square_var.get(),
                        resize=resize
                    )

                    if result.success:
                        successful += 1
                    else:
                        failed += 1
                        self.crop_log(f"  Failed: {result.error_message}")

                self.crop_log(f"\nBatch crop complete!")
                self.crop_log(f"  Successful: {successful}")
                self.crop_log(f"  Failed: {failed}")
                self.crop_log(f"  Output: {output_dir}")

                messagebox.showinfo("Batch Crop Complete",
                    f"Processed {len(pairs_to_process)} images\n"
                    f"Successful: {successful}\n"
                    f"Failed: {failed}\n\n"
                    f"Output: {output_dir}")

            except Exception as e:
                self.crop_log(f"Error: {str(e)}")
                messagebox.showerror("Batch Crop Error", str(e))

            finally:
                self.crop_run_btn['state'] = 'normal'
                self.crop_progress['value'] = 0

        # Run in separate thread
        thread = threading.Thread(target=crop_thread, daemon=True)
        thread.start()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = SAM3DGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
