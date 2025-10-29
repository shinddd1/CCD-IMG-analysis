import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import subprocess
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Ellipse
from scipy.ndimage import map_coordinates # Imported directly for line profile
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import json
import tempfile
import time
import threading

# Import ROI analysis module
try:
    from roi_analysis import (
        detect_beam_shape,
        fit_ellipse_roi,
        fit_fan_shape_roi,
        calculate_roi_complex,
        calculate_integrated_intensity,
        compute_fitted_fan_shape
    )
except ImportError:
    print("Warning: roi_analysis module not found. Using default ellipse fitting only.")

# ── Global Variables & Setup ──
PIXEL_SIZE = 0.5        # Initial μm/px conversion factor
TICK_INTERVAL = 2.0    # Tick interval in μm
BEAM_SHAPE_MODE = 'ellipse'  # 'ellipse' or 'fan' - beam fitting mode

# === EUV Power Calculation Parameters (User Modify) ===
CCD_Gain_Para = 1.0
Laser_Rep_Rate = 150000  # Hz
Num_of_Pulses = 150000
E_photon_eV = 91.4  # eV
QE = 0.84  # Quantum Efficiency
Filter_Trans = 0.008   # Filter Transmittance
ML_Reflect = 0.65  # Mirror Reflectance
Optical_Coll_Eff = 1.0  # Optical Collection Efficiency
Ar_Trans = 0.96649 # Ar Transmittance
e_charge_C = 1.602e-19 # Elementary charge in Coulombs

# To store axis-specific data, including references to colorbars for dynamic updates
axis_data_map = {}
# Global list of active colorbar objects for the general double-click adjustment
active_colorbars = []

# === Program Restart Function ===
def restart_program():
    """설정 저장 후 프로그램 재시작"""
    print("\n[설정 저장 완료]")
    print("프로그램을 재시작합니다...")
    print("(잠시만 기다려주세요...)")
    
    script_path = os.path.abspath(sys.argv[0])
    python_exe = sys.executable
    
    # matplotlib 창 닫기
    plt.close('all')
    
    try:
        if sys.platform == 'win32':
            # Windows: 새 프로세스 시작
            proc = subprocess.Popen(
                [python_exe, script_path],
                shell=False,
                cwd=os.path.dirname(script_path),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 새 프로세스가 시작될 때까지 대기
            time.sleep(1.5)
            
            # 현재 프로세스 종료
            sys.exit(0)
        else:
            # Unix/Linux
            subprocess.Popen([python_exe, script_path])
            time.sleep(1.0)
            sys.exit(0)
        
    except Exception as e:
        print(f"재시작 오류: {e}")
        print("수동으로 재시작해주세요: python " + script_path)

# === File Selection Function ===
def ask_files_selection():
    """Select SPE files to analyze"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    files = filedialog.askopenfilenames(
        title="SPE 파일 선택",
        filetypes=[("SPE files", "*.spe"), ("All files", "*.*")],
        initialdir=r"C:\Users\user\Desktop\회사 관련 서류\신동엽-LEUS\CCD IMG"
    )
    
    root.destroy()
    
    if not files:
        print("No files selected. Exiting.")
        sys.exit(0)
    
    return list(files)

# === Beam Shape Selection Function ===
def ask_beam_shape():
    """Program start dialog to select beam shape (pinhole or fan)"""
    root = tk.Tk()
    root.withdraw()
    
    result = messagebox.askyesno(
        "Beam Shape Selection",
        "핀홀이 있나요?\n\n"
        "YES: 타원 피팅 (핀홀 사용)\n"
        "NO: 부채꼴 피팅 (핀홀 없음)",
        icon='question'
    )
    
    root.destroy()
    return 'ellipse' if result else 'fan'

# === Frame Persistence Functions ===
def load_frame_overrides():
    """Loads frame override settings from a JSON file."""
    try:
        with open("frame_override.json", "r") as f:
            data = json.load(f)
            # 기존 형식 {idx: frame} 지원
            if isinstance(list(data.values())[0], dict):
                return {int(k): v for k, v in data.items()}
            else:
                # 기존 형식을 새 형식으로 변환
                return {int(k): {"frame": v, "margin": 50} for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError, IndexError):
        return {}

def save_frame_override(plot_index, frame_index, margin=50):
    """Saves a single frame override setting to a JSON file."""
    overrides = load_frame_overrides()
    overrides[plot_index] = {"frame": frame_index, "margin": margin}
    with open("frame_override.json", "w") as f:
        json.dump(overrides, f, indent=4)

def update_frame_display(idx, new_frame):
    """프레임 변경 시 즉시 업데이트"""
    global file_list, file_list_full_paths, directory, axes_img_flat, axes_prof_flat
    
    # frame_override.json에 저장
    save_frame_override(idx, new_frame)
    
    print(f"[프레임 변경] Plot {idx+1} -> Frame {new_frame+1}")
    
    # 해당 플롯만 다시 처리
    if idx < len(file_list) and idx < len(axes_img_flat):
        # Use full path from selected files
        full_path = file_list_full_paths[idx] if idx < len(file_list_full_paths) else os.path.join(directory, file_list[idx])
        ax_img_current = axes_img_flat[idx]
        ax_prof_current = axes_prof_flat[idx]
        
        try:
            metadata = load_spe_metadata(full_path)
            num_frames_in_file = metadata[2]
            
            if num_frames_in_file > new_frame:
                process_and_display_frame(full_path, new_frame, num_frames_in_file, metadata, file_list[idx], ax_img_current, ax_prof_current, idx)
                ax_img_current.figure.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating frame: {e}")
    
# === Event Handlers ===
def on_click(event):
    print("on_click called")
    if not event.dblclick or event.inaxes is None:
        return
    
    # --- Part 1: Handle colorbar clicks for changing color limits ---
    # Find which colorbar (if any) was clicked and its corresponding image axis
    clicked_cb = None
    image_ax_for_cb = None
    im_for_cb = None

    for ax, ax_data in axis_data_map.items():
        cb = ax_data.get('colorbar_object')
        if cb and event.inaxes == cb.ax:
            clicked_cb = cb
            image_ax_for_cb = ax
            im_for_cb = ax_data.get('imshow_object')
            break

    if clicked_cb and image_ax_for_cb and im_for_cb:
        vmin, vmax = im_for_cb.get_clim()
        vmin_str = simpledialog.askstring("Set Colorbar Min", f"Current min: {vmin:.2f}\nEnter new min:", parent=_root)
        if vmin_str is None: return
        vmax_str = simpledialog.askstring("Set Colorbar Max", f"Current max: {vmax:.2f}\nEnter new max:", parent=_root)
        if vmax_str is None: return

        try:
            new_vmin = float(vmin_str)
            new_vmax = float(vmax_str)
            if new_vmin >= new_vmax:
                messagebox.showerror("Error", "Min must be less than Max.", parent=_root)
                return

            im_for_cb.set_clim(new_vmin, new_vmax)
            clicked_cb.update_normal(im_for_cb)
            # Correctly store user_clim against the IMAGE axis, not the colorbar axis
            axis_data_map[image_ax_for_cb]['user_clim'] = (new_vmin, new_vmax)
            event.canvas.draw_idle()
        except ValueError:
            messagebox.showerror("Error", "Invalid number entered.", parent=_root)
        return # We handled the colorbar click, so we're done.

    # --- Part 2: Handle image axis clicks for changing frames ---
    if event.inaxes in axis_data_map:
        ax_data = axis_data_map[event.inaxes]
        if ax_data['total_frames'] > 1:
            new_frame_num_str = simpledialog.askstring(
                "Change Frame",
                f"File: {os.path.basename(ax_data['filepath'])}\n"
                f"Current frame: {ax_data['current_frame'] + 1} of {ax_data['total_frames']}\n"
                f"Enter new frame number (1–{ax_data['total_frames']}):",
                parent=_root)
            if new_frame_num_str:
                try:
                    new_frame_num = int(new_frame_num_str)
                    if 1 <= new_frame_num <= ax_data['total_frames']:
                        new_frame_index = new_frame_num - 1
                        if new_frame_index != ax_data['current_frame']: # Only update if frame actually changes
                            update_frame_display(ax_data['idx'], new_frame_index)
                    else:
                        messagebox.showerror("Error", f"Frame number must be between 1 and {ax_data['total_frames']}.", parent=_root)
                except ValueError:
                    messagebox.showerror("Error", "Invalid frame number.", parent=_root)
        return

def on_key(event):
    global PIXEL_SIZE
    
    if event.key.lower() == 's':
        # Switch between ellipse and fan fitting
        global BEAM_SHAPE_MODE
        BEAM_SHAPE_MODE = 'fan' if BEAM_SHAPE_MODE == 'ellipse' else 'ellipse'
        print(f"[빔 모양 변경] {BEAM_SHAPE_MODE} 모드")
        # 재시작
        restart_program()
    
    if event.key.lower() == 'u':
        global PIXEL_SIZE, TICK_INTERVAL
        
        # Create dialog window
        dialog = tk.Toplevel()
        dialog.title("Axis Settings")
        dialog.geometry("350x150")
        dialog.transient(_root)
        dialog.grab_set()
        
        # Pixel size entry
        tk.Label(dialog, text="Pixel Size (μm/px):", font=("Arial", 9)).grid(row=0, column=0, padx=10, pady=10, sticky='w')
        entry_pixel = tk.Entry(dialog, width=15)
        entry_pixel.insert(0, str(PIXEL_SIZE))
        entry_pixel.grid(row=0, column=1, padx=10, pady=10)
        
        # Tick interval entry
        tk.Label(dialog, text="Tick Interval (μm):", font=("Arial", 9)).grid(row=1, column=0, padx=10, pady=10, sticky='w')
        entry_interval = tk.Entry(dialog, width=15)
        entry_interval.insert(0, str(TICK_INTERVAL))
        entry_interval.grid(row=1, column=1, padx=10, pady=10)
        
        def apply_settings():
            global PIXEL_SIZE, TICK_INTERVAL
            try:
                new_pixel_size = float(entry_pixel.get())
                new_tick_interval = float(entry_interval.get())
                
                if new_pixel_size <= 0:
                    messagebox.showerror("Error", "Pixel size must be positive.", parent=dialog)
                    return
                if new_tick_interval <= 0:
                    messagebox.showerror("Error", "Tick interval must be positive.", parent=dialog)
                    return
                
                PIXEL_SIZE = new_pixel_size
                TICK_INTERVAL = new_tick_interval
                
                # Update all image axis labels
                for ax_img_key in axis_data_map:
                    ax_data = axis_data_map[ax_img_key]
                    roi_shape = ax_data.get('current_roi_shape')
                    if roi_shape:
                        # Calculate tick positions based on interval
                        roi_width_um = roi_shape[1] * PIXEL_SIZE
                        roi_height_um = roi_shape[0] * PIXEL_SIZE
                        
                        # X ticks
                        num_xticks = int(roi_width_um / TICK_INTERVAL) + 1
                        if num_xticks < 2: num_xticks = 2
                        xt_um = np.linspace(0, roi_width_um, num_xticks)
                        xt = xt_um / PIXEL_SIZE
                        
                        # Y ticks
                        num_yticks = int(roi_height_um / TICK_INTERVAL) + 1
                        if num_yticks < 2: num_yticks = 2
                        yt_um = np.linspace(0, roi_height_um, num_yticks)
                        yt = yt_um / PIXEL_SIZE
                        
                        ax_img_key.set_xticks(xt)
                        ax_img_key.set_yticks(yt)
                        ax_img_key.set_xticklabels([f"{x:.1f}" for x in xt_um])
                        # Y축이 반전되어 있으므로 labels도 역순으로 표시
                        ax_img_key.set_yticklabels([f"{y:.1f}" for y in yt_um[::-1]])
                
                event.canvas.draw_idle()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid number entered.", parent=dialog)
        
        def cancel():
            dialog.destroy()
        
        # Buttons frame
        btn_frame = tk.Frame(dialog)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=15)
        
        btn_apply = tk.Button(btn_frame, text="Apply", command=apply_settings, width=12, height=2)
        btn_apply.pack(side='left', padx=5)
        
        btn_cancel = tk.Button(btn_frame, text="Cancel", command=cancel, width=12, height=2)
        btn_cancel.pack(side='left', padx=5)
        
        # Focus on first entry and bind Enter key
        entry_pixel.focus_set()
        entry_pixel.bind('<Return>', lambda e: entry_interval.focus_set())
        entry_interval.bind('<Return>', lambda e: apply_settings())
        
        # Make dialog modal
        dialog.wait_window()


# === SPE File Loading Functions ===
def load_spe_metadata(filepath):
    with open(filepath, "rb") as f:
        header_raw = f.read(4100) # SPE header is 4100 bytes
    
    width = int.from_bytes(header_raw[42:44], byteorder='little')
    height = int.from_bytes(header_raw[656:658], byteorder='little')
    num_frames = int.from_bytes(header_raw[1446:1450], byteorder='little')
    
    data_type_byte = header_raw[108]
    if data_type_byte == 0: # Float (4 bytes)
        dtype = np.float32
        bytes_per_pixel = 4
    elif data_type_byte == 1: # Long (4 bytes)
        dtype = np.int32
        bytes_per_pixel = 4
    elif data_type_byte == 2: # Integer (2 bytes)
        dtype = np.int16
        bytes_per_pixel = 2
    elif data_type_byte == 3: # Unsigned Integer (2 bytes)
        dtype = np.uint16
        bytes_per_pixel = 2
    else:
        raise ValueError(f"Unsupported SPE data type byte: {data_type_byte} in file {filepath}")
    
    header_size = 4100
    return width, height, num_frames, dtype, bytes_per_pixel, header_size

def load_spe_frame(filepath, frame_index, width, height, dtype, bytes_per_pixel, header_size):
    frame_size_in_bytes = width * height * bytes_per_pixel
    offset = header_size + frame_index * frame_size_in_bytes
    with open(filepath, "rb") as f:
        f.seek(offset)
        frame_data_raw = f.read(frame_size_in_bytes)
        if len(frame_data_raw) < frame_size_in_bytes:
            raise EOFError(f"Unexpected end of file when reading frame {frame_index} from {filepath}. Expected {frame_size_in_bytes} bytes, got {len(frame_data_raw)}.")
    
    image = np.frombuffer(frame_data_raw, dtype=dtype, count=width * height)
    image = image.reshape((height, width))
    return image.astype(np.float32) # Convert to float32 for consistent processing


# === Utility Functions ===
def get_line_profile(img, x0, y0, x1, y1, n_points=300):
    x = np.linspace(x0, x1, n_points)
    y = np.linspace(y0, y1, n_points)
    
    # Using scipy.ndimage.map_coordinates for accurate sub-pixel interpolation
    coords = np.vstack((y, x)) # map_coordinates expects (row, col)
    profile = map_coordinates(img, coords, order=1, mode='nearest') # order=1 for linear interpolation

    # Calculate distance along the line segment
    start_point = np.array([x0, y0])
    end_point = np.array([x1, y1])
    line_length_pixels = np.linalg.norm(end_point - start_point)
    
    dist_axis = np.linspace(0, line_length_pixels, n_points) * PIXEL_SIZE
    
    return dist_axis, profile


def get_auto_roi(image, margin=20):
    if image is None or image.size == 0: return None
    norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    # Filter out very small contours that might be noise
    min_contour_area = 10 # Adjust as needed
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    if not valid_contours: return None

    x, y, w, h = cv2.boundingRect(max(valid_contours, key=cv2.contourArea))
    xmin = max(0, x - margin)
    xmax = min(image.shape[1], x + w + margin)
    ymin = max(0, y - margin)
    ymax = min(image.shape[0], y + h + margin)
    return xmin, xmax, ymin, ymax

def compute_fwhm_from_profile(distance, profile):
    """Computes the Full-Width at Half-Maximum from a line profile."""
    if profile is None or len(profile) < 2 or PIXEL_SIZE <= 0:
        return 0, 0 # FWHM in pixels, FWHM in distance units
    
    peak_val = np.max(profile)
    baseline = np.min(profile)
    half_max = baseline + (peak_val - baseline) / 2.0
    
    indices_above_half = np.where(profile >= half_max)[0]
    
    if len(indices_above_half) < 2:
        return 0, 0

    fwhm_indices_span = indices_above_half[-1] - indices_above_half[0]
    
    if len(distance) > 1:
        dist_per_index = (distance[-1] - distance[0]) / (len(distance) - 1) if len(distance) > 1 else 0
        fwhm_dist = fwhm_indices_span * dist_per_index
    else:
        fwhm_dist = 0
        
    return fwhm_dist / PIXEL_SIZE, fwhm_dist

def calculate_ellipse_data(filepath, frame_idx, total_frames, metadata, fname_short):
    width, height, _, dtype_val, bytes_pp, header_s = metadata
    # ... 기존 코드 ...
    ellipse_data = {
        "Filename": fname_short, "Frame": frame_idx + 1,
        "Major Length (px)": None, "Minor Length (px)": None,
        "Major Length (μm)": None, "Minor Length (μm)": None,
        "FWHM Major (px)": None, "FWHM Minor (px)": None,
        "FWHM Major (μm)": None, "FWHM Minor (μm)": None,
        "Collected Power": None,
        "CCD Power": None,
        "CCD Counts": None,
        "Error": None
    }

    try:
        image_data = load_spe_frame(filepath, frame_idx, width, height, dtype_val, bytes_pp, header_s)
        roi_coords = get_auto_roi(image_data)
        if not roi_coords:
            ellipse_data["Error"] = "No ROI"
            return ellipse_data

        xmin, xmax, ymin, ymax = roi_coords
        
        # ROI를 정사각형으로 확장 (가장 긴 변에 맞춤)
        roi_width = xmax - xmin
        roi_height = ymax - ymin
        max_dim = max(roi_width, roi_height)
        
        # 폭 확장
        if roi_width < max_dim:
            padding_x = (max_dim - roi_width) // 2
            xmin = max(0, xmin - padding_x)
            xmax = min(image_data.shape[1], xmax + padding_x)
            # 한 쪽이라도 추가 확장 필요
            if xmax - xmin < max_dim:
                if xmin == 0:
                    xmax = max_dim
                else:
                    xmin = xmax - max_dim
        
        # 높이 확장
        if roi_height < max_dim:
            padding_y = (max_dim - roi_height) // 2
            ymin = max(0, ymin - padding_y)
            ymax = min(image_data.shape[0], ymax + padding_y)
            # 한 쪽이라도 추가 확장 필요
            if ymax - ymin < max_dim:
                if ymin == 0:
                    ymax = max_dim
                else:
                    ymin = ymax - max_dim
        
        roi = image_data[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            ellipse_data["Error"] = "Empty ROI"
            return ellipse_data

        peak_y_roi, peak_x_roi = np.unravel_index(np.argmax(roi), roi.shape)

        # ... 기존 intensity range, th_e2 등 계산 ...

        th_e2 = roi.max() / np.e**2
        mask_e2 = None
        mask_fwhm = None
        ellipse = None
        fwhm_major_px = fwhm_minor_px = angle = None

        if roi.max() > th_e2:
            mask_e2 = (roi > th_e2).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_e2, connectivity=8)
            if num_labels > 1:
                peak_label = labels[int(peak_y_roi), int(peak_x_roi)]
                if peak_label == 0 and num_labels > 1:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    if len(areas) > 0: peak_label = np.argmax(areas) + 1

                if peak_label != 0:
                    mask_peak_e2 = (labels == peak_label).astype(np.uint8)
                    contours_e2, _ = cv2.findContours(mask_peak_e2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours_e2:
                        cnt = max(contours_e2, key=cv2.contourArea)
                        if len(cnt) >= 5:
                            ellipse = cv2.fitEllipse(cnt)
                            (cx, cy), (maj, minr), angle = ellipse
                            
                            # OpenCV는 maj >= minr을 보장하지만, 실제 값 확인
                            # 실제 더 긴 축을 장축으로 사용
                            if maj < minr:
                                maj, minr = minr, maj  # swap
                                angle = (angle + 90) % 180  # 90도 회전
                            
                            a, b = maj / 2, minr / 2
                            ellipse_data["Eccentricity"] = np.sqrt(1 - (b**2)/(a**2)) if a > 0 else 0
                            ellipse_data["Major Length (px)"] = maj
                            ellipse_data["Minor Length (px)"] = minr
                            ellipse_data["Major Length (μm)"] = maj * PIXEL_SIZE
                            ellipse_data["Minor Length (μm)"] = minr * PIXEL_SIZE

                            # --- 1/e² 타원 평균 intensity & Power 계산 ---
                            yy, xx = np.indices(roi.shape)
                            angle_rad = np.deg2rad(angle)
                            x0 = cx
                            y0 = cy
                            x_rot = (xx - x0) * np.cos(angle_rad) + (yy - y0) * np.sin(angle_rad)
                            y_rot = -(xx - x0) * np.sin(angle_rad) + (yy - y0) * np.cos(angle_rad)
                            mask_ellipse = ((x_rot / (maj/2))**2 + (y_rot / (minr/2))**2) <= 1
                            if np.any(mask_ellipse):
                                area_e2 = np.sum(mask_ellipse)
                                mean_intensity_e2 = np.mean(roi[mask_ellipse])
                                ccd_counts = float(np.sum(roi[mask_ellipse]))
                                ellipse_data["CCD Counts"] = ccd_counts

                                # EUV Power Calculation
                                numerator = (
                                    ccd_counts
                                    * CCD_Gain_Para
                                    * (Laser_Rep_Rate / Num_of_Pulses)
                                    * E_photon_eV
                                    * e_charge_C
                                    * 1000
                                    / (E_photon_eV / 3.62)
                                )
                                denominator = (
                                    QE * Filter_Trans * ML_Reflect * Optical_Coll_Eff * Ar_Trans
                                )
                                if denominator > 0:
                                    euv_power_mW = numerator / denominator
                                else:
                                    euv_power_mW = 0
                                ellipse_data["Collected Power"] = euv_power_mW

                            # --- FWHM 타원 평균 intensity & Power 계산 ---
                            # FWHM 계산
                            dx_maj, dy_maj = np.cos(angle_rad), np.sin(angle_rad)
                            dx_min, dy_min = -np.sin(angle_rad), np.cos(angle_rad)
                            length = max(a, b) * 7.5

                            x0m, y0m = peak_x_roi - dx_maj * length, peak_y_roi - dy_maj * length
                            x1m, y1m = peak_x_roi + dx_maj * length, peak_y_roi + dy_maj * length
                            d_major, p_major = get_line_profile(roi, x0m, y0m, x1m, y1m)
                            _, fwhm_major_um = compute_fwhm_from_profile(d_major, p_major)
                            fwhm_major_px = fwhm_major_um / PIXEL_SIZE

                            x0n, y0n = peak_x_roi - dx_min * length, peak_y_roi - dy_min * length
                            x1n, y1n = peak_x_roi + dx_min * length, peak_y_roi + dy_min * length
                            d_minor, p_minor = get_line_profile(roi, x0n, y0n, x1n, y1n)
                            _, fwhm_minor_um = compute_fwhm_from_profile(d_minor, p_minor)
                            fwhm_minor_px = fwhm_minor_um / PIXEL_SIZE

                            ellipse_data["FWHM Major (px)"] = fwhm_major_px
                            ellipse_data["FWHM Minor (px)"] = fwhm_minor_px
                            ellipse_data["FWHM Major (μm)"] = fwhm_major_um
                            ellipse_data["FWHM Minor (μm)"] = fwhm_minor_um

                            # FWHM 타원 면적과 강도 계산
                            yy_fwhm, xx_fwhm = np.indices(roi.shape)
                            x0_fwhm = peak_x_roi
                            y0_fwhm = peak_y_roi
                            x_rot_fwhm = (xx_fwhm - x0_fwhm) * np.cos(angle_rad) + (yy_fwhm - y0_fwhm) * np.sin(angle_rad)
                            y_rot_fwhm = -(xx_fwhm - x0_fwhm) * np.sin(angle_rad) + (yy_fwhm - y0_fwhm) * np.cos(angle_rad)
                            mask_fwhm_ellipse = ((x_rot_fwhm / (fwhm_major_px/2))**2 + (y_rot_fwhm / (fwhm_minor_px/2))**2) <= 1
                            if np.any(mask_fwhm_ellipse):
                                area_fwhm = np.sum(mask_fwhm_ellipse)
                                mean_intensity_fwhm = np.mean(roi[mask_fwhm_ellipse])
                                ellipse_data["CCD Power"] = float(mean_intensity_fwhm * area_fwhm)
                        else:
                            ellipse_data["Error"] = "Not enough contour points"
                    else:
                        ellipse_data["Error"] = "No contours found"
                else:
                    ellipse_data["Error"] = "Peak not in a valid component"
            else:
                ellipse_data["Error"] = "No components found"
    except Exception as e:
        ellipse_data["Error"] = str(e)

    return ellipse_data

# === Core Processing and Plotting Function ===
def process_and_display_frame(ax_img, ax_prof, filepath, frame_idx,
                              total_frames, metadata, fname_short, 
                              original_idx):
    global global_vmin, global_vmax # Use pre-calculated global values if needed

    width, height, _, dtype_val, bytes_pp, header_s = metadata

    # --- Clear axes before redrawing (창 꺼졌다 켜지는 효과) ---
    ax_img.clear()
    ax_prof.clear()
    
    # --- Colorbar Management ---
    old_cb = axis_data_map[ax_img].get('colorbar_object')
    if old_cb:
        if old_cb.ax in ax_img.figure.axes: # Check if the colorbar's axis is still part of the figure
            ax_img.figure.delaxes(old_cb.ax) # Remove the colorbar's axis from the figure
        if old_cb in active_colorbars:
            active_colorbars.remove(old_cb)
        axis_data_map[ax_img]['colorbar_object'] = None # Clear reference in map

    ellipse_data = {
    "Filename": fname_short,
    "Frame": frame_idx + 1,
    "Major Length (px)": None,
    "Minor Length (px)": None,
    "Major Length (μm)": None,
    "Minor Length (μm)": None,
    "FWHM Major (px)": None,
    "FWHM Minor (px)": None,
    "FWHM Major (μm)": None,
    "FWHM Minor (μm)": None,
    "Collected Power": None,  # 1/e² 타원 내부 총 광자 수 또는 에너지
    "CCD Power": None,  # FWHM 타원 내부 총 광자 수 또는 에너지
    "CCD Counts": None,
    "Error": None
    }
    try:
        image_data = load_spe_frame(filepath, frame_idx, width, height, dtype_val, bytes_pp, header_s)
        
        # ROI는 항상 margin 50으로 계산 (좌표 일관성 유지)
        roi_coords_large = get_auto_roi(image_data, margin=50)  # 항상 margin 50으로 ROI 계산
        if not roi_coords_large:
            ax_img.set_title(f"{fname_short} (Fr. {frame_idx+1}/{total_frames})\n(No ROI)", fontsize=TITLE_FONTSIZE)
            ax_img.axis('off'); ax_prof.axis('off')
            return ellipse_data # Return default data
        
        xmin_large, xmax_large, ymin_large, ymax_large = roi_coords_large
        
        # ROI를 정사각형으로 확장 (가장 긴 변에 맞춤)
        roi_width = xmax_large - xmin_large
        roi_height = ymax_large - ymin_large
        max_dim = max(roi_width, roi_height)
        
        # 폭 확장
        if roi_width < max_dim:
            padding_x = (max_dim - roi_width) // 2
            xmin_large = max(0, xmin_large - padding_x)
            xmax_large = min(image_data.shape[1], xmax_large + padding_x)
            # 한 쪽이라도 추가 확장 필요
            if xmax_large - xmin_large < max_dim:
                if xmin_large == 0:
                    xmax_large = max_dim
                else:
                    xmin_large = xmax_large - max_dim
        
        # 높이 확장
        if roi_height < max_dim:
            padding_y = (max_dim - roi_height) // 2
            ymin_large = max(0, ymin_large - padding_y)
            ymax_large = min(image_data.shape[0], ymax_large + padding_y)
            # 한 쪽이라도 추가 확장 필요
            if ymax_large - ymin_large < max_dim:
                if ymin_large == 0:
                    ymax_large = max_dim
                else:
                    ymin_large = ymax_large - max_dim
        
        roi_large = image_data[ymin_large:ymax_large, xmin_large:xmax_large]
        
        # axis_data_map에서 margin 정보 가져오기 (없으면 50으로 초기화)
        display_margin = axis_data_map.get(ax_img, {}).get('current_margin', 50)
        
        # Figure 1 표시용: display_margin에 따라 roi_large의 중앙을 crop
        center_y = roi_large.shape[0] // 2
        center_x = roi_large.shape[1] // 2
        
        # display_margin이 50이면 전체, 20이면 중심 기준 작은 영역
        if display_margin == 20:
            # 중심 기준으로 작은 영역 crop (정사각형 유지)
            crop_size = min(roi_large.shape[0], roi_large.shape[1], int(min(roi_large.shape[0], roi_large.shape[1]) * 0.4))
            
            y0_crop = max(0, center_y - crop_size // 2)
            y1_crop = min(roi_large.shape[0], center_y + crop_size // 2)
            x0_crop = max(0, center_x - crop_size // 2)
            x1_crop = min(roi_large.shape[1], center_x + crop_size // 2)
            
            # 정사각형 확보
            if y1_crop - y0_crop != x1_crop - x0_crop:
                crop_size = min(y1_crop - y0_crop, x1_crop - x0_crop)
                y0_crop = max(0, center_y - crop_size // 2)
                y1_crop = y0_crop + crop_size
                x0_crop = max(0, center_x - crop_size // 2)
                x1_crop = x0_crop + crop_size
            
            roi = roi_large[y0_crop:y1_crop, x0_crop:x1_crop]
            
            # crop offset 저장 (roi_large 좌표 → roi 좌표 변환용)
            crop_offset_x = x0_crop
            crop_offset_y = y0_crop
        else:
            # display_margin이 50이면 전체 표시
            roi = roi_large
            crop_offset_x = 0
            crop_offset_y = 0
        print("ROI shape:", roi.shape)
        if roi.size == 0: # Handle empty ROI
            ax_img.set_title(f"{fname_short} (Fr. {frame_idx+1}/{total_frames})\n(Empty ROI)", fontsize=TITLE_FONTSIZE)
            ax_img.axis('off'); ax_prof.axis('off')
            return ellipse_data

        axis_data_map[ax_img]['current_roi_shape'] = roi.shape # Store for 'u' key updates

        peak_y_roi, peak_x_roi = np.unravel_index(np.argmax(roi), roi.shape)
        print("Peak (x, y):", peak_x_roi, peak_y_roi)
        
        # roi_large에서 peak 찾기 (Profile용)
        peak_y_roi_large, peak_x_roi_large = np.unravel_index(np.argmax(roi_large), roi_large.shape)


        current_vmin, current_vmax = 0, 1
        user_clim = axis_data_map[ax_img].get('user_clim')

        if INTENSITY_NORM == 'manual' and user_clim:
            current_vmin, current_vmax = user_clim
        elif INTENSITY_NORM == 'global':
            current_vmin, current_vmax = global_vmin, global_vmax
        else:  # Fallback to 'individual' for 'individual' mode or 'manual' without user input
            if roi.size > 0:
                # percentile 범위를 좁혀서 배경 노이즈 제거
                current_vmin = np.percentile(roi, 0.1)  # 0.1% (더 낮은 값으로 설정)
                current_vmax = np.percentile(roi, 99.9)  # 99.9% (더 높은 값으로 설정)
                if current_vmin >= current_vmax:
                    current_vmin = roi.min()
                    current_vmax = roi.max()
                    if current_vmin == current_vmax:
                        current_vmax += 1e-6  # Avoid zero range
            else:
                current_vmin, current_vmax = 0, 1

        # --- Plotting the image with original limits and aspect ---
        im = ax_img.imshow(roi, cmap=COLORMAP, vmin=current_vmin, vmax=current_vmax, 
                   interpolation='nearest', aspect='equal')
        cb = plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        axis_data_map[ax_img]['colorbar_object'] = cb
        active_colorbars.append(cb)

        axis_data_map[ax_img]['imshow_object'] = im
        axis_data_map[ax_img]['colorbar_object'] = cb
        active_colorbars.append(cb)

        ax_img.set_xlim(0, roi.shape[1])
        ax_img.set_ylim(roi.shape[0], 0)
        # Zoom 시에도 정사각형 비율 유지 (adjustable='datalim' 사용)
        ax_img.set_aspect('equal', adjustable='datalim')
        
        # --- 1/e^2 calculations ---
        # Define 1/e^2 threshold based on the roi_large's actual min/max intensity for contouring
        # The threshold is set to 1/e^2 of the absolute maximum intensity in the roi_large.
        # This is a common definition, particularly when background noise is minimal.
        th_e2 = roi_large.max() / np.e**2
        if roi_large.max() > th_e2 : # Ensure there are pixels above threshold
            mask_e2 = (roi_large > th_e2).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_e2, connectivity=8)
            if num_labels > 1: # More than background
                # Find component containing the peak
                peak_label = labels[int(peak_y_roi_large), int(peak_x_roi_large)]
                # If peak is in background (label 0), or if it's an isolated noise component
                # Try to find the largest component that isn't the background (label 0)
                if peak_label == 0 and num_labels > 1:
                    areas = stats[1:, cv2.CC_STAT_AREA] # Exclude background
                    if len(areas) > 0:
                        peak_label = np.argmax(areas) + 1 # +1 because we excluded label 0

                if peak_label != 0: # Proceed only if a valid peak component is found
                    mask_peak_e2 = (labels == peak_label).astype(np.uint8)
                    contours_e2, _ = cv2.findContours(mask_peak_e2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours_e2:
                        cnt = max(contours_e2, key=cv2.contourArea)
                        
                        # BEAM_SHAPE_MODE에 따라 타원 또는 부채꼴 피팅
                        if BEAM_SHAPE_MODE == 'fan':
                            # 부채꼴 피팅
                            try:
                                fan_result = fit_fan_shape_roi(roi_large, th_e2, peak_y_roi_large, peak_x_roi_large)
                                if fan_result:
                                    cnt_fan, mask_fan, (cx_large, cy_large), angle = fan_result
                                    # roi_large 좌표를 roi 좌표계로 변환
                                    cx = cx_large - crop_offset_x
                                    cy = cy_large - crop_offset_y
                                    
                                    # Contour를 roi 좌표계로 변환
                                    cnt_roi = cnt_fan.reshape(-1, 2).copy()
                                    cnt_roi[:, 0] -= crop_offset_x
                                    cnt_roi[:, 1] -= crop_offset_y
                                    ax_img.plot(cnt_roi[:, 0], cnt_roi[:, 1], 
                                              color='red', linewidth=1.5, linestyle='--', 
                                              label='1/e² RAW (Fan)', alpha=0.7)
                                    
                                    # 마스크 생성 및 CCD Counts 계산
                                    yy, xx = np.indices(roi.shape)
                                    mask_fan_roi = np.zeros(roi.shape, dtype=bool)
                                    cnt_points_roi = cnt_roi.astype(int)
                                    cv2.fillPoly(mask_fan_roi, [cnt_points_roi], True)
                                    
                                    if np.any(mask_fan_roi):
                                        ccd_counts = float(np.sum(roi[mask_fan_roi]))
                                        ellipse_data["CCD Counts"] = ccd_counts
                                        
                                        # EUV Power Calculation
                                        numerator = (
                                            ccd_counts
                                            * CCD_Gain_Para
                                            * (Laser_Rep_Rate / Num_of_Pulses)
                                            * E_photon_eV
                                            * e_charge_C
                                            * 1000
                                            / (E_photon_eV / 3.62)
                                        )
                                        denominator = (
                                            QE * Filter_Trans * ML_Reflect * Optical_Coll_Eff * Ar_Trans
                                        )
                                        if denominator > 0:
                                            euv_power_mW = numerator / denominator
                                        else:
                                            euv_power_mW = 0
                                        ellipse_data["Collected Power"] = euv_power_mW
                                    
                                    # Fitted Fan Shape 계산 및 표시
                                    try:
                                        fitted_shape = compute_fitted_fan_shape(cnt_fan, cx_large, cy_large, angle)
                                        if fitted_shape is not None and len(fitted_shape) > 0:
                                            fitted_shape_roi = fitted_shape.copy()
                                            fitted_shape_roi[:, 0] -= crop_offset_x
                                            fitted_shape_roi[:, 1] -= crop_offset_y
                                            ax_img.plot(fitted_shape_roi[:, 0], fitted_shape_roi[:, 1], 
                                                      color='cyan', linewidth=2, 
                                                      label='1/e² Fitting', alpha=0.8)
                                            print(f"[Fitted Fan Shape] Points: {len(fitted_shape)}")
                                        else:
                                            print(f"[Warning] fitted_shape is None or empty")
                                    except Exception as e:
                                        print(f"[Error computing fitted fan shape]: {e}")
                        
                        if len(cnt) >= 5 and BEAM_SHAPE_MODE == 'ellipse':
                            # Continue with ellipse fitting only if in ellipse mode
                            # 타원 피팅
                            ellipse = cv2.fitEllipse(cnt)
                            (cx_large, cy_large), (maj, minr), angle = ellipse
                            
                            # roi_large 좌표를 roi 좌표계로 변환 (crop offset 사용)
                            cx = cx_large - crop_offset_x
                            cy = cy_large - crop_offset_y
                            
                            # RAW 데이터 (contour)를 점선으로 그리기 (roi 좌표계로 변환)
                            cnt_reshaped = cnt.reshape(-1, 2)
                            cnt_roi = cnt_reshaped.copy()
                            cnt_roi[:, 0] -= crop_offset_x
                            cnt_roi[:, 1] -= crop_offset_y
                            ax_img.plot(cnt_roi[:, 0], cnt_roi[:, 1], 
                                      color='red', linewidth=1.5, linestyle='--', 
                                      label='1/e² RAW', alpha=0.7)
                            
                            # OpenCV는 maj >= minr을 보장하지만, 실제 값 확인
                            # 실제 더 긴 축을 장축으로 사용
                            if maj < minr:
                                maj, minr = minr, maj  # swap
                                angle = (angle + 90) % 180  # 90도 회전
                            
                            a, b = maj / 2, minr / 2
                            ecc = np.sqrt(1 - (b**2)/(a**2)) if a > 0 else 0
                            ellipse_data["Eccentricity"] = ecc
                            ellipse_data["Major Length (px)"] = maj
                            ellipse_data["Minor Length (px)"] = minr
                            ellipse_data["Major Length (μm)"] = maj * PIXEL_SIZE
                            ellipse_data["Minor Length (μm)"] = minr * PIXEL_SIZE

                            # 1/e² 타원 면적과 강도 계산
                            yy, xx = np.indices(roi.shape)
                            angle_rad_e2 = np.deg2rad(angle)
                            x0_e2 = cx
                            y0_e2 = cy
                            x_rot_e2 = (xx - x0_e2) * np.cos(angle_rad_e2) + (yy - y0_e2) * np.sin(angle_rad_e2)
                            y_rot_e2 = -(xx - x0_e2) * np.sin(angle_rad_e2) + (yy - y0_e2) * np.cos(angle_rad_e2)
                            mask_ellipse = ((x_rot_e2 / (maj/2))**2 + (y_rot_e2 / (minr/2))**2) <= 1
                            if np.any(mask_ellipse):
                                ccd_counts = float(np.sum(roi[mask_ellipse]))
                                ellipse_data["CCD Counts"] = ccd_counts

                                # EUV Power Calculation
                                numerator = (
                                    ccd_counts
                                    * CCD_Gain_Para
                                    * (Laser_Rep_Rate / Num_of_Pulses)
                                    * E_photon_eV
                                    * e_charge_C
                                    * 1000
                                    / (E_photon_eV / 3.62)
                                )
                                denominator = (
                                    QE * Filter_Trans * ML_Reflect * Optical_Coll_Eff * Ar_Trans
                                )
                                if denominator > 0:
                                    euv_power_mW = numerator / denominator
                                else:
                                    euv_power_mW = 0
                                ellipse_data["Collected Power"] = euv_power_mW

                            ax_img.add_patch(Ellipse((cx, cy), maj, minr, angle=angle,
                                                    edgecolor='red', lw=2, fill=False, linestyle='-',
                                                    label='1/e² Ellipse'))

                            # --- FWHM/프로파일 기반 타원 ---
                            angle_rad = np.deg2rad(angle)
                            dx = np.cos(angle_rad)
                            dy = np.sin(angle_rad)
                            length = max(a, b) * 7.5

                            # roi_large 기준 좌표 계산 (Profile용 & Figure 1 표시용)
                            x0m_large = peak_x_roi_large - dx * length
                            y0m_large = peak_y_roi_large - dy * length
                            x1m_large = peak_x_roi_large + dx * length
                            y1m_large = peak_y_roi_large + dy * length
                            
                            # roi_large 좌표를 roi 좌표계로 변환 (Figure 1 표시용 - crop offset 사용)
                            x0m = x0m_large - crop_offset_x
                            y0m = y0m_large - crop_offset_y
                            x1m = x1m_large - crop_offset_x
                            y1m = y1m_large - crop_offset_y
                            ax_img.plot([x0m, x1m], [y0m, y1m], color='magenta', lw=1.5, label='Major Axis')
                            d_major, p_major = get_line_profile(roi_large, x0m_large, y0m_large, x1m_large, y1m_large)
                            ax_prof.plot(d_major, p_major, color='magenta', label='Major')

                            # Minor axis (cyan)
                            dx_n, dy_n = -np.sin(angle_rad), np.cos(angle_rad)
                            x0n_large = peak_x_roi_large - dx_n * length
                            y0n_large = peak_y_roi_large - dy_n * length
                            x1n_large = peak_x_roi_large + dx_n * length
                            y1n_large = peak_y_roi_large + dy_n * length
                            
                            # roi_large 좌표를 roi 좌표계로 변환 (Figure 1 표시용 - crop offset 사용)
                            x0n = x0n_large - crop_offset_x
                            y0n = y0n_large - crop_offset_y
                            x1n = x1n_large - crop_offset_x
                            y1n = y1n_large - crop_offset_y
                            ax_img.plot([x0n, x1n], [y0n, y1n], color='cyan', lw=1.5, label='Minor Axis')
                            d_minor, p_minor = get_line_profile(roi_large, x0n_large, y0n_large, x1n_large, y1n_large)
                            ax_prof.plot(d_minor, p_minor, color='cyan', label='Minor')

                            # FWHM 계산 (profile 기준)
                            fwhm_major_px, fwhm_major_um = compute_fwhm_from_profile(d_major, p_major)
                            fwhm_minor_px, fwhm_minor_um = compute_fwhm_from_profile(d_minor, p_minor)

                            # FWHM RAW 데이터: 실제 강도 기반 (1/e² RAW와 동일한 방식)
                            peak_val = roi[int(peak_y_roi), int(peak_x_roi)]
                            peak_val_large = roi_large[int(peak_y_roi_large), int(peak_x_roi_large)]
                            baseline = np.percentile(roi_large, 10)  # 하위 10%를 베이스라인으로
                            half_max = baseline + (peak_val_large - baseline) / 2.0
                            
                            # FWHM 경계 마스크 생성
                            mask_fwhm = (roi_large >= half_max).astype(np.uint8)
                            contours_fwhm, _ = cv2.findContours(mask_fwhm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours_fwhm:
                                # 가장 큰 contour 찾기
                                cnt_fwhm = max(contours_fwhm, key=cv2.contourArea)
                                if len(cnt_fwhm) > 0:
                                    cnt_reshaped = cnt_fwhm.reshape(-1, 2)
                                    cnt_roi_fwhm = cnt_reshaped.copy()
                                    cnt_roi_fwhm[:, 0] -= crop_offset_x
                                    cnt_roi_fwhm[:, 1] -= crop_offset_y
                                    ax_img.plot(cnt_roi_fwhm[:, 0], cnt_roi_fwhm[:, 1], 
                                              color='lime', linewidth=1.5, linestyle='--', 
                                              label='FWHM RAW', alpha=0.7)

                            # FWHM Ellipse 추가 (중심은 peak 기준, roi 좌표계로 변환 - crop offset 사용)
                            peak_x_roi_from_large = peak_x_roi_large - crop_offset_x
                            peak_y_roi_from_large = peak_y_roi_large - crop_offset_y
                            ax_img.add_patch(Ellipse(
                                (peak_x_roi_from_large, peak_y_roi_from_large),
                                width=fwhm_major_px,
                                height=fwhm_minor_px,
                                angle=angle,
                                edgecolor='lime',
                                lw=1.5,
                                linestyle='-',
                                fill=False,
                                label='FWHM Ellipse'
                            ))

                            # FWHM 값 저장
                            ellipse_data["FWHM Major (px)"] = fwhm_major_px
                            ellipse_data["FWHM Minor (px)"] = fwhm_minor_px
                            ellipse_data["FWHM Major (μm)"] = fwhm_major_um
                            ellipse_data["FWHM Minor (μm)"] = fwhm_minor_um

                            # FWHM 타원 면적과 강도 계산
                            yy_fwhm, xx_fwhm = np.indices(roi.shape)
                            angle_rad_fwhm = np.deg2rad(angle)
                            x0_fwhm = peak_x_roi
                            y0_fwhm = peak_y_roi
                            x_rot_fwhm = (xx_fwhm - x0_fwhm) * np.cos(angle_rad_fwhm) + (yy_fwhm - y0_fwhm) * np.sin(angle_rad_fwhm)
                            y_rot_fwhm = -(xx_fwhm - x0_fwhm) * np.sin(angle_rad_fwhm) + (yy_fwhm - y0_fwhm) * np.cos(angle_rad_fwhm)
                            mask_fwhm_ellipse = ((x_rot_fwhm / (fwhm_major_px/2))**2 + (y_rot_fwhm / (fwhm_minor_px/2))**2) <= 1
                            if np.any(mask_fwhm_ellipse):
                                # 타원 면적 (픽셀 수)
                                area_fwhm = np.sum(mask_fwhm_ellipse)
                                # 타원 내부 평균 강도
                                mean_intensity_fwhm = np.mean(roi[mask_fwhm_ellipse])
                                # 총 Power (광자 수 또는 에너지)
                                ellipse_data["CCD Power"] = float(mean_intensity_fwhm * area_fwhm)

        ax_img.plot(peak_x_roi, peak_y_roi, 'yx', markersize=8, label='Peak Intensity')
        
        # Titles and labels
        frame_info_str = f"(Fr. {frame_idx + 1}/{total_frames})" if total_frames > 1 else ""
        intensity_info = f"I: {current_vmin:.1f}–{current_vmax:.1f}"
        ax_img.set_title(f"{fname_short} {frame_info_str}\n{intensity_info}", fontsize=TITLE_FONTSIZE)

        # Update tick labels based on current ROI shape and PIXEL_SIZE
        # Note: The ticks will be for the *full image coordinates*, not ROI coordinates
        # unless you adjust roi.shape[1] and roi.shape[0] to be full image dimensions.
        # For simplicity, if ROI moves, these ticks might not perfectly align with the new ROI origin.
        # If AUTO_SCALE is True, the plot will naturally fit the full image or the ROI if not manually setting limits.
        # If MANUAL_XLIM/YLIM are used, they override this.
        # axis label은 roi_large 기준 좌표를 표시 (crop된 경우 crop_offset 고려)
        if AUTO_SCALE: # Only set auto ticks if not manually scaling
            # roi는 crop된 영역, axis label은 roi_large 기준 좌표 표시
            # Calculate tick positions based on interval
            roi_width_um = roi.shape[1] * PIXEL_SIZE
            roi_height_um = roi.shape[0] * PIXEL_SIZE
            
            # X ticks
            num_xticks = int(roi_width_um / TICK_INTERVAL) + 1
            if num_xticks < 2: num_xticks = 2
            xt_um = np.linspace(0, roi_width_um, num_xticks)
            x_tick_positions = xt_um / PIXEL_SIZE
            
            # roi_large 기준 실제 좌표로 변환
            xt = x_tick_positions + crop_offset_x
            
            # Y ticks
            num_yticks = int(roi_height_um / TICK_INTERVAL) + 1
            if num_yticks < 2: num_yticks = 2
            yt_um = np.linspace(0, roi_height_um, num_yticks)
            y_tick_positions = yt_um / PIXEL_SIZE
            
            # roi_large 기준 실제 좌표로 변환
            yt = y_tick_positions + crop_offset_y
            
            ax_img.set_xticks(x_tick_positions)
            ax_img.set_yticks(y_tick_positions)
            ax_img.set_xticklabels([f"{x:.1f}" for x in xt_um])
            # Y축이 반전되어 있으므로 labels도 역순으로 표시
            ax_img.set_yticklabels([f"{y:.1f}" for y in yt_um[::-1]])
        else: # If manual limits are set, the ticks should reflect those limits in microns
            # Calculate ticks based on interval
            x_range = MANUAL_XLIM[1] - MANUAL_XLIM[0]
            y_range = MANUAL_YLIM[1] - MANUAL_YLIM[0]
            
            num_xticks = int(x_range / TICK_INTERVAL) + 1
            if num_xticks < 2: num_xticks = 2
            num_yticks = int(y_range / TICK_INTERVAL) + 1
            if num_yticks < 2: num_yticks = 2
            
            xt_um = np.linspace(MANUAL_XLIM[0], MANUAL_XLIM[1], num_xticks)
            yt_um = np.linspace(MANUAL_YLIM[0], MANUAL_YLIM[1], num_yticks)
            xt = xt_um / PIXEL_SIZE
            yt = yt_um / PIXEL_SIZE
            
            ax_img.set_xticks(xt); ax_img.set_yticks(yt)
            ax_img.set_xticklabels([f"{x:.1f}" for x in xt_um])
            # Y축이 반전되어 있으므로 labels도 역순으로 표시
            ax_img.set_yticklabels([f"{y:.1f}" for y in yt_um[::-1]])
        ax_img.set_xlabel("X (μm)", fontsize=AXIS_FONTSIZE)
        ax_img.set_ylabel("Y (μm)", fontsize=AXIS_FONTSIZE)
        
        # 범례 순서 지정: Major/Minor Axis 먼저, 그 다음 1/e², FWHM
        handles, labels = ax_img.get_legend_handles_labels()
        desired_order = ['Major Axis', 'Minor Axis', '1/e² RAW', '1/e² Ellipse', 'FWHM RAW', 'FWHM Ellipse', 'Peak Intensity']
        # 원하는 순서대로 정렬
        ordered_handles = []
        ordered_labels = []
        for desired_label in desired_order:
            if desired_label in labels:
                idx = labels.index(desired_label)
                ordered_handles.append(handles[idx])
                ordered_labels.append(labels[idx])
        # 나머지 항목 추가 (중복 방지)
        for i, label in enumerate(labels):
            if label not in ordered_labels:
                ordered_handles.append(handles[i])
                ordered_labels.append(label)
        
        ax_img.legend(ordered_handles, ordered_labels, fontsize=LEGEND_FONTSIZE, loc='lower right')

        ax_prof.set_title(f"Profiles: {fname_short} {frame_info_str}", fontsize=TITLE_FONTSIZE)
        ax_prof.set_xlabel(f"Distance from Peak (μm)", fontsize=AXIS_FONTSIZE)
        ax_prof.set_ylabel('Intensity', fontsize=AXIS_FONTSIZE)
        ax_prof.legend(fontsize=LEGEND_FONTSIZE)
        ax_prof.grid(True, alpha=0.3)

    except Exception as e:
        frame_info_str = f"(Fr. {frame_idx + 1}/{total_frames})" if total_frames > 1 else ""
        error_msg = str(e).replace("\n", " ") # Keep error message on one line for title
        ax_img.set_title(f"{fname_short} {frame_info_str}\n(Error: {error_msg[:30]}...)", fontsize=TITLE_FONTSIZE)
        ax_img.axis('off'); ax_prof.axis('off')
        print(f"[ERROR] Processing {filepath} Frame {frame_idx+1}: {e}")
        pass
    
    return ellipse_data

# === Main Script Parameters & Execution ===
COLORMAP = 'viridis'
TITLE_FONTSIZE = 9
AXIS_FONTSIZE = 7
LEGEND_FONTSIZE = 6
AUTO_SCALE = True
MANUAL_XLIM = [0, 100]
MANUAL_YLIM = [0, 100]
INTENSITY_NORM = 'individual' # 'individual', 'global', or 'manual'

# === Ask user to select files ===
print("="*50)
print("파일 선택 다이얼로그를 여는 중...")
print("="*50)

file_list_full_paths = ask_files_selection()
print(f"선택된 파일: {len(file_list_full_paths)}개")
for i, fpath in enumerate(file_list_full_paths):
    print(f"  [{i+1}] {os.path.basename(fpath)}")

# Extract filenames and directory for compatibility
file_list_orig = [os.path.basename(f) for f in file_list_full_paths]
if file_list_full_paths:
    directory = os.path.dirname(file_list_full_paths[0])
else:
    print("No files selected")
    sys.exit()

# Sort files numerically if they follow a pattern like 'file_1.spe', 'file_10.spe'
try:
    file_list = sorted(file_list_orig, key=lambda x: int("".join(filter(str.isdigit, os.path.splitext(x)[0])) or 0))
except ValueError:
    file_list = sorted(file_list_orig) # Fallback to alphabetical sort

# === Ask user for beam shape ===
BEAM_SHAPE_MODE = ask_beam_shape()
print(f"\n[빔 모양 선택] Mode: {'타원 피팅 (핀홀)' if BEAM_SHAPE_MODE == 'ellipse' else '부채꼴 피팅 (핀홀 없음)'}")

# --- STAGE 1: Pre-analyze all frames to find the best one ---
print("="*50)
print("STAGE 1: 사전 분석 - 모든 프레임을 분석하여 최적 프레임을 찾습니다.")
print("="*50)
all_frames_data = []
for idx, fname in enumerate(file_list):
    # Use full path from selected files
    full_path = file_list_full_paths[idx] if idx < len(file_list_full_paths) else os.path.join(directory, fname)
    print(f"  Analyzing [{idx+1}/{len(file_list)}] {fname}...")
    try:
        metadata = load_spe_metadata(full_path)
        num_frames = metadata[2]
        if num_frames == 0: continue
        for i in range(num_frames):
            frame_data = calculate_ellipse_data(full_path, i, num_frames, metadata, fname)
            all_frames_data.append(frame_data)
    except Exception as e:
        print(f"    [ERROR] Failed to analyze {fname}: {e}")

all_frames_df = pd.DataFrame(all_frames_data)
excel_all_frames_path = "ellipse_fitting_summary_ALL_FRAMES.xlsx"
all_frames_df.to_excel(excel_all_frames_path, index=False)
print(f"\n전체 프레임 분석 결과가 '{excel_all_frames_path}'에 저장되었습니다.")

# --- STAGE 2: Find best frames and update JSON ---
new_overrides = {}
if not all_frames_df.empty and "Major Length (px)" in all_frames_df.columns:
    valid_df = all_frames_df.dropna(subset=["Major Length (px)", "Minor Length (px)"]).copy()
    valid_df = valid_df[valid_df["Frame"] > 1].copy()  # ✅ Skip frame 1
    valid_df["Ratio"] = valid_df["Minor Length (px)"] / valid_df["Major Length (px)"]
    valid_df["RatioDiff"] = (valid_df["Ratio"] - 1).abs()
    
    for idx, fname in enumerate(file_list):
        file_df = valid_df[valid_df["Filename"] == fname]
        if not file_df.empty:
            best_frame_row = file_df.loc[file_df["RatioDiff"].idxmin()]
            print(f"  Best frame for {fname}: Frame {best_frame_row['Frame']}, Ratio={best_frame_row['Ratio']:.3f}, Major={best_frame_row['Major Length (px)']:.1f}, Minor={best_frame_row['Minor Length (px)']:.1f}")
            new_overrides[idx] = int(best_frame_row["Frame"]) - 1  # Frame is 1-based

if new_overrides:
    with open("frame_override.json", "w") as f:
        json.dump(new_overrides, f, indent=4)
    print("최적 프레임 정보를 'frame_override.json'에 업데이트했습니다.")
    frame_overrides = new_overrides # Use the newly found overrides for this run
else:
    print("경고: 최적 프레임을 찾지 못했습니다. 기존 설정을 사용합니다.")
    frame_overrides = load_frame_overrides() # Fallback to old settings

print("\n" + "="*50)
print("STAGE 3: 시각화 - 최적 프레임을 플롯에 표시합니다.")
print("="*50)
    
num_files = len(file_list)

# 파일 수에 따라 최적의 cols 계산
if num_files == 1:
    cols = 1
    rows = 1
elif num_files == 2:
    cols = 2
    rows = 1
elif num_files <= 4:
    cols = 2
    rows = 2
elif num_files <= 8:
    cols = 4
    rows = 2
elif num_files <= 12:
    cols = 4
    rows = 3
elif num_files <= 16:
    cols = 4
    rows = 4
else:
    # 16개 초과 시 cols=5 또는 cols=6로 조정
    cols = 5 if num_files <= 25 else 6
    rows = (num_files + cols - 1) // cols

# Global intensity normalization (calculated once based on the first frame of each file's ROI)
global_vmin, global_vmax = 0, 1
if INTENSITY_NORM == 'global':
    all_intensities_for_global_norm = []
    print("Calculating global intensity normalization...")
    for idx, fname_norm in enumerate(file_list):
        try:
            # Use full path from selected files
            path_norm = file_list_full_paths[idx] if idx < len(file_list_full_paths) else os.path.join(directory, fname_norm)
            meta_norm = load_spe_metadata(path_norm)
            if meta_norm[2] > 0: # num_frames > 0
                img_data_norm = load_spe_frame(path_norm, 0, meta_norm[0], meta_norm[1], meta_norm[3], meta_norm[4], meta_norm[5])
                roi_coords_norm = get_auto_roi(img_data_norm)
                if roi_coords_norm:
                    xmin_n, xmax_n, ymin_n, ymax_n = roi_coords_norm
                    roi_n = img_data_norm[ymin_n:ymax_n, xmin_n:xmax_n]
                    if roi_n.size > 0:
                         all_intensities_for_global_norm.extend(roi_n.flatten())
        except Exception as e_norm:
            print(f"  Skipping {fname_norm} for global norm due to error: {e_norm}")
            continue
    if all_intensities_for_global_norm:
        global_vmin = np.percentile(all_intensities_for_global_norm, 1)
        global_vmax = np.percentile(all_intensities_for_global_norm, 99)
        if global_vmin >= global_vmax : # Handle flat data or all same values
            global_vmin = min(all_intensities_for_global_norm)
            global_vmax = max(all_intensities_for_global_norm)
            if global_vmin == global_vmax:
                global_vmax += 1e-6 # Ensure range is not zero
        print(f"Global vmin={global_vmin:.2f}, vmax={global_vmax:.2f} calculated.")
    else:
        print("Warning: Could not gather data for global normalization. Defaulting to individual normalization or manual adjustment.")
        INTENSITY_NORM = 'individual' # Fallback

# 파일 수에 따라 figsize 조정
if num_files == 1:
    fig_size_img = (8, 8)
    fig_size_prof = (8, 6)
elif num_files == 2:
    fig_size_img = (14, 7)
    fig_size_prof = (14, 5)
elif num_files <= 4:
    fig_size_img = (12, 12)
    fig_size_prof = (12, 8)
elif num_files <= 8:
    fig_size_img = (22, 11)
    fig_size_prof = (22, 8)
else:
    fig_size_img = (7 * cols, 6 * rows)
    fig_size_prof = (7 * cols, 4 * rows)

fig_img, axes_img_list = plt.subplots(rows, cols, figsize=fig_size_img)
fig_prof, axes_prof_list = plt.subplots(rows, cols, figsize=fig_size_prof)

# Flatten axes arrays for easier iteration, handle single plot case
axes_img_flat = np.array(axes_img_list).flatten()
axes_prof_flat = np.array(axes_prof_list).flatten()

# 파일 수에 따라 여백 조정
if num_files <= 4:
    # 적은 수의 파일: 여백 더 줄임
    fig_img.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.1, wspace=0.2, hspace=0.35)
    fig_prof.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.1, wspace=0.2, hspace=0.35)
else:
    # 많은 파일: 기존 여백 유지
    fig_img.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08, wspace=0.15, hspace=0.3)
    fig_prof.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08, wspace=0.15, hspace=0.3)

fig_img.canvas.mpl_connect('button_press_event', on_click)
fig_img.canvas.mpl_connect('key_press_event', on_key)
# fig_prof.canvas.mpl_connect('key_press_event', on_key) # Optionally connect to profile window too

results = [None] * num_files # Pre-allocate results list
if __name__ == "__main__":
    # frame_overrides 로드
    frame_overrides = load_frame_overrides()
    print("frame_overrides loaded:", frame_overrides)

    for idx, fname in enumerate(file_list):
        # Use full path from selected files
        full_path = file_list_full_paths[idx] if idx < len(file_list_full_paths) else os.path.join(directory, fname)
        ax_img_current = axes_img_flat[idx]
        ax_prof_current = axes_prof_flat[idx]

        try:
            metadata = load_spe_metadata(full_path)
            num_frames_in_file = metadata[2]

            if num_frames_in_file == 0:
                ax_img_current.set_title(f"{idx+1}: {fname}\n(No frames found)", fontsize=TITLE_FONTSIZE)
                ax_img_current.axis('off'); ax_prof_current.axis('off')
                results[idx] = {"Filename": fname, "Frame": 0, "Error": "No frames"}
                continue

            # Use the saved frame override if it exists, otherwise default to frame 0
            frame_data = frame_overrides.get(idx, {"frame": 0, "margin": 50})
            if isinstance(frame_data, dict):
                initial_frame_index = frame_data.get("frame", 0)
                initial_margin = frame_data.get("margin", 50)
            else:
                # 기존 형식 지원 (backward compatibility)
                initial_frame_index = frame_data
                initial_margin = 50
            
            print(f"Plot {idx}: initial_frame_index = {initial_frame_index}, margin = {initial_margin}")
            # Validate that the saved frame is within the bounds of the file
            if not (0 <= initial_frame_index < num_frames_in_file):
                print(f"Warning: Saved frame {initial_frame_index + 1} for {fname} is out of bounds. Resetting to 1.")
                initial_frame_index = 0
                save_frame_override(idx, 0) # Correct the invalid entry in the JSON file

            temp_img_for_dims = load_spe_frame(full_path, 0, metadata[0], metadata[1], metadata[3], metadata[4], metadata[5])
            initial_img_height, initial_img_width = temp_img_for_dims.shape
            ax_img_current.set_xlim(0, initial_img_width)
            ax_img_current.set_ylim(initial_img_height, 0)

            axis_data_map[ax_img_current] = {
                'filepath': full_path,
                'ax_prof': ax_prof_current,
                'total_frames': num_frames_in_file,
                'current_frame': initial_frame_index, # Use the loaded/validated frame
                'metadata': metadata,
                'idx': idx,
                'imshow_object': None,
                'colorbar_object': None,
                'current_roi_shape': None,
                'initial_xlim': (0, initial_img_width),
                'initial_ylim': (initial_img_height, 0),
                'current_margin': initial_margin  # margin: 50 또는 20
            }

            ellipse_data_for_file = process_and_display_frame(
                ax_img=ax_img_current,
                ax_prof=ax_prof_current,
                filepath=full_path,
                frame_idx=initial_frame_index, # Pass the loaded/validated frame
                total_frames=num_frames_in_file,
                metadata=metadata,
                fname_short=fname,
                original_idx=idx
            )
            results[idx] = ellipse_data_for_file

        except Exception as e_main:
            error_msg_main = str(e_main).replace("\n", " ")
            ax_img_current.set_title(f"{idx+1}: {fname}\n(Load Error: {error_msg_main[:30]}...)", fontsize=TITLE_FONTSIZE)
            ax_img_current.axis('off'); ax_prof_current.axis('off')
            print(f"[CRITICAL ERROR] Loading or initial processing of {fname}: {e_main}")
            results[idx] = {"Filename": fname, "Frame": 0, "Error": str(e_main)}

    # Clean up unused subplots
    for i in range(num_files, len(axes_img_flat)):
        axes_img_flat[i].axis('off')
        axes_prof_flat[i].axis('off')

    fig_img.suptitle(f'Beam Profile Analysis - Colormap: {COLORMAP} (Press "u" to change μm/px, Dbl-Click image to change frame, Dbl-Click cbar for scale)', fontsize=14)
    fig_prof.suptitle(f'Line Profiles - Intensity Norm: {INTENSITY_NORM} (Best Frames Auto-Selected)', fontsize=14)

    # 아래 tight_layout 부분 등에서 VSCode/Pyright 등 linter가 'rect' 옵션(튜플이어야 함)에 리스트를 줬으니 회색으로 보일 수 있음.
    # 원래: fig_img.tight_layout(rect=[0, 0, 1, 0.96], pad=2.0)
    # 수정 권장: rect=(0, 0, 1, 0.96)  ← 리스트→튜플 변경 필요!
    fig_img.tight_layout(rect=(0, 0, 1, 0.96), pad=2.0)
    fig_prof.tight_layout(rect=(0, 0, 1, 0.96), pad=2.0)

    # Excel saving
    df = pd.DataFrame([res for res in results if res is not None]) # Filter out None entries if any failed critically
    excel_filename = "ellipse_fitting_summary_multi_frame.xlsx"
    try:
        df.to_excel(excel_filename, index=False)
        print(f"\nPlotting complete. Displayed frames summary saved to {excel_filename}")
    except Exception as e_excel:
        print(f"\nError saving to Excel: {e_excel}. Results may not be saved.")

    # === 여기서 바로 frame_override.json 자동 생성 ===
    frame_override = {}
    # 아래 groupby 및 'notnull', 'loc', 'idxmin' 사용에서도 'DataFrame'이 아니라 ndarray 등에 써서 linter가 회색/빨간줄 낼 수 있음!
    # df가 진짜 DataFrame임을 확실히 하세요.
    if not df.empty and "Major Length (px)" in df and "Minor Length (px)" in df:
        for fname, group in df.groupby("Filename"):
            # 첫 번째 프레임(보통 Frame==1)은 제외
            group = group[group["Frame"] > 1].copy()
            # Major/Minor 값이 유효한 경우만
            group = group[(group["Major Length (px)"].notnull()) & (group["Minor Length (px)"].notnull())]
            # 1/e² 타원 기준 비율 계산
            group["Ratio"] = group["Minor Length (px)"] / group["Major Length (px)"]
            group["RatioDiff"] = (group["Ratio"] - 1).abs()
            if not group.empty:
                best_row = group.loc[group["RatioDiff"].idxmin()]
                plot_idx = df["Filename"].drop_duplicates().tolist().index(fname)
                frame_override[plot_idx] = int(best_row["Frame"]) - 1  # Frame이 1부터 시작이면 -1

        with open("frame_override.json", "w", encoding="utf-8") as f:
            json.dump(frame_override, f, indent=4)
        print("frame_override.json updated!")
    else:
        print("Warning: Could not update frame_override.json (missing columns or empty data).")

    print(f"Using colormap: {COLORMAP}, Intensity normalization: {INTENSITY_NORM}")
    print(f"μm Conversion Factor (PIXEL_SIZE): {PIXEL_SIZE} μm/px")

    _root = tk.Tk()
    _root.withdraw()
    try:
        plt.show()
        _root.destroy()
    except Exception as e:
        print(e)