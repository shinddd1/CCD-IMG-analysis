import pandas as pd

# 엑셀 파일 읽기
df = pd.read_excel("ellipse_fitting_summary_multi_frame.xlsx")

# 필요한 파라미터 입력 (혹은 파일에서 읽기)
GAIN = 1.0
QE = 0.85
T = 0.1  # 초
E_PHOTON = 3.2e-19  # J

# Detected Power 계산 (예: Mean Intensity (1/e2) 기준)
df["Detected Power (1/e2)"] = df["Mean Intensity (1/e2)"] * GAIN * E_PHOTON / (QE * T)
df["Detected Power (FWHM)"] = df["Mean Intensity (FWHM)"] * GAIN * E_PHOTON / (QE * T)

# 결과 저장
df.to_excel("ellipse_fitting_summary_with_power.xlsx", index=False)
print("Detected Power 계산 결과가 ellipse_fitting_summary_with_power.xlsx에 저장되었습니다.")