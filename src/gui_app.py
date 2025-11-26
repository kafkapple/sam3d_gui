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

        # Setup UI
        self.setup_ui()

        # Default data directory (relative to project root)
        project_root = Path(__file__).parent.parent
        self.data_dir = str(project_root / "data" / "markerless_mouse")
        self.output_dir = str(project_root / "outputs")

    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
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


def main():
    """Main entry point"""
    root = tk.Tk()
    app = SAM3DGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
