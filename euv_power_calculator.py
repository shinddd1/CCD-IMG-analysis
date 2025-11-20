import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EUVPowerCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("EUV Power Calculator")
        self.root.geometry("650x750")
        
        # 고정 파라미터 (절대 변경 안됨)
        self.E_PHOTON = 91.4  # eV
        self.QE = 0.84
        self.FILTER_TRANS = 0.008
        self.OPTICAL_COLL_EFF = 0.0017977
        self.AR_TRANS = 0.0330910
        self.E_CHARGE = 1.602e-19  # Coulombs
        
        # 사용자 입력 변수 (Tkinter Variables)
        self.ccd_counts = tk.DoubleVar(value=0.0)
        self.ccd_gain = tk.DoubleVar(value=1.0)
        self.laser_rep_rate = tk.DoubleVar(value=100000.0)
        self.num_pulses = tk.DoubleVar(value=20000.0)
        self.ml_reflect = tk.DoubleVar(value=0.65)
        
        # 결과 변수
        self.result_power = tk.StringVar(value="Press Calculate or Enter to compute")
        
        self.create_widgets()
        print("[Calculator] Initialized - Manual calculation mode")
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        ttk.Label(main_frame, text="EUV Power Calculator", 
                 font=("Arial", 18, "bold")).grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        row = 1
        
        # === 입력 파라미터들 ===
        ttk.Label(main_frame, text="━━━ Input Parameters ━━━", 
                 font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, pady=(10, 5))
        row += 1
        
        # CCD Counts (읽기 전용)
        ttk.Label(main_frame, text="CCD Counts:", font=("Arial", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.ccd_counts, width=20, state='readonly',
                 font=("Arial", 10)).grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
        ttk.Label(main_frame, text="(from analysis)", font=("Arial", 8), 
                 foreground="gray").grid(row=row, column=2, sticky=tk.W)
        row += 1
        
        # CCD Gain
        ttk.Label(main_frame, text="CCD Gain:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.gain_entry = ttk.Entry(main_frame, textvariable=self.ccd_gain, width=20, font=("Arial", 10))
        self.gain_entry.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
        self.gain_entry.bind('<Return>', lambda e: self.on_enter_pressed())
        self.gain_entry.bind('<FocusOut>', lambda e: self.sync_entry_to_var(self.gain_entry, self.ccd_gain))
        row += 1
        
        # Laser Rep Rate
        ttk.Label(main_frame, text="Laser Rep Rate (Hz):", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.rep_entry = ttk.Entry(main_frame, textvariable=self.laser_rep_rate, width=20, font=("Arial", 10))
        self.rep_entry.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
        self.rep_entry.bind('<Return>', lambda e: self.on_enter_pressed())
        self.rep_entry.bind('<FocusOut>', lambda e: self.sync_entry_to_var(self.rep_entry, self.laser_rep_rate))
        row += 1
        
        # Number of Pulses
        ttk.Label(main_frame, text="Number of Pulses:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.pulses_entry = ttk.Entry(main_frame, textvariable=self.num_pulses, width=20, font=("Arial", 10))
        self.pulses_entry.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
        self.pulses_entry.bind('<Return>', lambda e: self.on_enter_pressed())
        self.pulses_entry.bind('<FocusOut>', lambda e: self.sync_entry_to_var(self.pulses_entry, self.num_pulses))
        row += 1
        
        # ML Reflectance
        ttk.Label(main_frame, text="ML Reflectance:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.ml_entry = ttk.Entry(main_frame, textvariable=self.ml_reflect, width=20, font=("Arial", 10))
        self.ml_entry.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
        self.ml_entry.bind('<Return>', lambda e: self.on_enter_pressed())
        self.ml_entry.bind('<FocusOut>', lambda e: self.sync_entry_to_var(self.ml_entry, self.ml_reflect))
        row += 1
        
        # === 고정 파라미터들 ===
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1
        
        ttk.Label(main_frame, text="━━━ Fixed Parameters ━━━", 
                 font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, pady=(5, 5))
        row += 1
        
        fixed_params = [
            ("Photon Energy (eV):", self.E_PHOTON),
            ("Quantum Efficiency:", self.QE),
            ("Filter Transmission:", self.FILTER_TRANS),
            ("Optical Coll. Eff:", self.OPTICAL_COLL_EFF),
            ("Ar Transmission:", self.AR_TRANS),
        ]
        
        for label, value in fixed_params:
            ttk.Label(main_frame, text=label, font=("Arial", 9)).grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Label(main_frame, text=f"{value}", font=("Arial", 9), 
                     foreground="gray").grid(row=row, column=1, sticky=tk.W, padx=10, pady=2)
            row += 1
        
        # === 수식 표시 ===
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        row += 1
        
        formula_frame = ttk.LabelFrame(main_frame, text="Formula", padding="10")
        formula_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        fig = Figure(figsize=(6, 0.9), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis('off')
        latex_text = (
            r"$P_{\mathrm{EUV}}\,[\mathrm{mW}] = "
            r"\frac{N_{\mathrm{CCD}} \cdot G \cdot f_{\mathrm{rep}} \cdot e \cdot 1000}"
            r"{N_{\mathrm{pulses}} \cdot QE \cdot T_{\mathrm{filter}} \cdot R_{\mathrm{ML}} \cdot \eta_{\mathrm{coll}} \cdot T_{\mathrm{Ar}}}"
            r"\times 3.62$"
        )
        ax.text(0.02, 0.5, latex_text, fontsize=15, va='center', ha='left')
        canvas = FigureCanvasTkAgg(fig, master=formula_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
        row += 1
        
        # === 결과 표시 ===
        result_frame = ttk.LabelFrame(main_frame, text="Result", padding="15")
        result_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        
        ttk.Label(result_frame, text="EUV Power:", font=("Arial", 12)).grid(row=0, column=0, sticky=tk.W, padx=5)
        self.result_label = ttk.Label(result_frame, textvariable=self.result_power, 
                                      font=("Arial", 16, "bold"), foreground="blue")
        self.result_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Label(result_frame, text="mW", font=("Arial", 12)).grid(row=0, column=2, sticky=tk.W)
        
        row += 1
        
        # === 버튼들 ===
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Calculate", command=self.calculate_power,
                  width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save to Excel", command=self.save_to_excel,
                  width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset_defaults,
                  width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Copy", command=self.copy_result,
                  width=10).pack(side=tk.LEFT, padx=5)
    
    def sync_entry_to_var(self, entry_widget, tk_var):
        """Entry 위젯의 텍스트를 Tkinter 변수로 동기화"""
        try:
            value = float(entry_widget.get())
            tk_var.set(value)
            print(f"[Sync] Entry value synced: {value}")
        except ValueError:
            print(f"[Sync] Invalid value in entry: {entry_widget.get()}")
    
    def calculate_power(self):
        """파워 계산 메인 함수"""
        try:
            # Entry 위젯들에서 직접 값 읽어서 변수에 저장
            self.sync_entry_to_var(self.gain_entry, self.ccd_gain)
            self.sync_entry_to_var(self.rep_entry, self.laser_rep_rate)
            self.sync_entry_to_var(self.pulses_entry, self.num_pulses)
            self.sync_entry_to_var(self.ml_entry, self.ml_reflect)
            
            # 이제 변수에서 값 읽어오기
            ccd_counts = self.ccd_counts.get()
            gain = self.ccd_gain.get()
            rep_rate = self.laser_rep_rate.get()
            num_pulses = self.num_pulses.get()
            ml_reflect = self.ml_reflect.get()
            
            print("\n" + "="*60)
            print("CALCULATING EUV POWER")
            print("="*60)
            print(f"CCD Counts      : {ccd_counts}")
            print(f"Gain            : {gain}")
            print(f"Rep Rate (Hz)   : {rep_rate}")
            print(f"Num Pulses      : {num_pulses}")
            print(f"ML Reflectance  : {ml_reflect}")
            print("-"*60)
            
            if ccd_counts == 0:
                self.result_power.set("No CCD Counts")
                messagebox.showwarning("Warning", "CCD Counts is 0!")
                return
            
            # 분자: CCD Counts × Gain × Rep Rate × e × 1000
            numerator = ccd_counts * gain * rep_rate * self.E_CHARGE * 1000
            print(f"Numerator       : {numerator:.6e}")
            
            # 분모: N_Pulses × QE × Filter Trans × ML Reflect × Coll Eff × Ar Trans
            denominator = (num_pulses * self.QE * self.FILTER_TRANS * 
                          ml_reflect * self.OPTICAL_COLL_EFF * self.AR_TRANS)
            print(f"Denominator     : {denominator:.6e}")
            
            if denominator == 0:
                self.result_power.set("Error: Div by 0")
                messagebox.showerror("Error", "Denominator is zero!")
                return
            
            # 최종 계산
            power_mw = (numerator / denominator) * 3.62
            
            print(f"Power (mW)      : {power_mw:.10f}")
            print("="*60)
            
            # 결과 업데이트 - 여러 방법으로 확실하게
            result_text = f"{power_mw:.10f}"
            
            # 1. StringVar 업데이트
            self.result_power.set(result_text)
            
            # 2. Label 직접 텍스트 설정
            self.result_label.configure(text=result_text)
            
            # 3. 강제 UI 갱신
            self.result_label.update()
            self.root.update_idletasks()
            self.root.update()
            
            print(f"[UI] Result updated to: {result_text}")
            print(f"[UI] Label text is now: {self.result_label.cget('text')}")
            print("="*60 + "\n")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.result_power.set(error_msg)
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
    
    def on_enter_pressed(self):
        """Enter 키 누를 때: 계산만 수행"""
        print("[Calculator] Enter pressed - Calculate only")
        self.calculate_power()
    
    def get_power_value(self):
        """현재 계산된 파워 값을 반환 (float or None)"""
        try:
            text = self.result_power.get()
            if "Error" in text or "Press" in text or "No" in text:
                return None
            return float(text)
        except:
            return None
    
    def calculate_power_for_ccd_counts(self, ccd_counts_value):
        """특정 CCD Counts에 대한 파워를 계산"""
        try:
            gain = self.ccd_gain.get()
            rep_rate = self.laser_rep_rate.get()
            num_pulses = self.num_pulses.get()
            ml_reflect = self.ml_reflect.get()
            
            if ccd_counts_value == 0:
                return None
            
            numerator = ccd_counts_value * gain * rep_rate * self.E_CHARGE * 1000
            denominator = (num_pulses * self.QE * self.FILTER_TRANS * 
                          ml_reflect * self.OPTICAL_COLL_EFF * self.AR_TRANS)
            
            if denominator == 0:
                return None
            
            power_mw = (numerator / denominator) * 3.62
            return power_mw
        except:
            return None
    
    def save_to_excel(self):
        """엑셀 파일에 저장"""
        try:
            current_ccd_counts = self.ccd_counts.get()
            
            # 저장할 파일 목록
            files = [
                "nonellipse_fitting_summary_ALL_FRAMES.xlsx",
                "nonellipse_fitting_summary_multi_frame.xlsx",
                "ellipse_fitting_summary_ALL_FRAMES.xlsx",
                "ellipse_fitting_summary_multi_frame.xlsx"
            ]
            
            updated_count = 0
            saved_files = []
            
            for filename in files:
                if not os.path.exists(filename):
                    continue
                
                df = pd.read_excel(filename)
                
                if "CCD Counts" not in df.columns or "Collected Power" not in df.columns:
                    continue
                
                is_all_frames = "ALL_FRAMES" in filename
                file_updated = 0
                
                if is_all_frames:
                    # ALL_FRAMES 파일: 모든 프레임에 대해 각각 계산
                    print(f"[Save] Processing ALL_FRAMES file: {filename}")
                    for idx, row in df.iterrows():
                        row_ccd_counts = row["CCD Counts"]
                        if pd.notna(row_ccd_counts) and row_ccd_counts > 0:
                            # 각 프레임의 CCD Counts로 파워 계산
                            calculated_power = self.calculate_power_for_ccd_counts(row_ccd_counts)
                            if calculated_power is not None:
                                df.at[idx, "Collected Power"] = calculated_power
                                file_updated += 1
                                print(f"  Frame {row.get('Frame', '?')}: CCD={row_ccd_counts:.0f} -> Power={calculated_power:.6f} mW")
                else:
                    # multi_frame 파일: 현재 CCD Counts와 일치하는 행만 업데이트
                    matching = df[np.isclose(df["CCD Counts"], current_ccd_counts, rtol=0.01)]
                    
                    if len(matching) > 0:
                        power = self.get_power_value()
                        if power is None:
                            messagebox.showwarning("Warning", "No valid power to save. Calculate first!")
                            return
                        
                        for idx in matching.index:
                            df.at[idx, "Collected Power"] = power
                            file_updated += 1
                            print(f"[Save] Updated row {idx} in {filename}: CCD={current_ccd_counts:.0f} -> Power={power:.6f} mW")
                
                if file_updated > 0:
                    df.to_excel(filename, index=False)
                    saved_files.append(f"{filename} ({file_updated} rows)")
                    updated_count += file_updated
            
            if updated_count > 0:
                msg = f"Updated {updated_count} rows:\n" + "\n".join(saved_files)
                messagebox.showinfo("Success", msg)
                print(f"[Save] Total {updated_count} rows updated across {len(saved_files)} files")
            else:
                messagebox.showwarning("Warning", "No rows were updated")
                
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def reset_defaults(self):
        """기본값으로 리셋"""
        self.ccd_gain.set(1.0)
        self.laser_rep_rate.set(100000.0)
        self.num_pulses.set(20000.0)
        self.ml_reflect.set(0.65)
        self.result_power.set("Press Calculate or Enter to compute")
        print("[Calculator] Reset to defaults")
    
    def copy_result(self):
        """결과 복사"""
        try:
            power = self.get_power_value()
            if power is not None:
                self.root.clipboard_clear()
                self.root.clipboard_append(f"{power:.10f}")
                messagebox.showinfo("Copied", f"Copied: {power:.10f} mW")
            else:
                messagebox.showwarning("Warning", "No valid result to copy")
        except Exception as e:
            messagebox.showerror("Error", f"Copy failed: {str(e)}")
    
    def update_ccd_counts(self, counts):
        """외부에서 CCD Counts 업데이트"""
        print(f"[Calculator] CCD Counts updated: {counts}")
        self.ccd_counts.set(counts)


# === 전역 함수 ===
calculator_instance = None

def create_calculator_window(parent=None):
    """계산기 창 생성"""
    global calculator_instance
    
    if parent is None:
        window = tk.Tk()
    else:
        window = tk.Toplevel(parent)
    
    calculator_instance = EUVPowerCalculator(window)
    return window, calculator_instance

def update_calculator_ccd_counts(counts):
    """외부에서 CCD Counts 업데이트"""
    global calculator_instance
    if calculator_instance:
        calculator_instance.update_ccd_counts(counts)
    else:
        print("[Calculator] Warning: No calculator instance!")

def get_default_parameters():
    """기본 파라미터 반환"""
    return {
        'CCD_Gain_Para': 1.0,
        'Laser_Rep_Rate': 100000.0,
        'Num_of_Pulses': 20000.0,
        'ML_Reflect': 0.65
    }

def get_fixed_parameters():
    """고정 파라미터 반환"""
    return {
        'E_photon_eV': 91.4,
        'QE': 0.84,
        'Filter_Trans': 0.008,
        'Optical_Coll_Eff': 0.0017977,
        'Ar_Trans': 0.0330910,
        'e_charge_C': 1.602e-19
    }


if __name__ == "__main__":
    root = tk.Tk()
    calculator = EUVPowerCalculator(root)
    root.mainloop()
