# CCD Image Analysis and Intensity

## 개요

CCD 이미지 분석 및 강도 측정을 위한 Python 프로그램입니다. SPE 파일 형식의 이미지 데이터를 처리하여 빔 프로파일을 분석하고 타원 피팅을 수행합니다.

## 주요 기능

- **SPE 파일 지원**: Princeton Instruments의 SPE 파일 형식 지원
- **자동 ROI 감지**: 이미지에서 관심 영역(ROI) 자동 감지
- **타원 피팅**: 1/e² 기준 타원 피팅 및 FWHM 계산
- **프레임 분석**: 다중 프레임 이미지에서 최적 프레임 자동 선택
- **Interactive 조정**: 
  - `u` 키: 픽셀 크기 및 눈금 간격 조정
  - 더블클릭: 프레임 변경
  - Colorbar 더블클릭: 강도 범위 조정

## 시스템 요구사항

- Python 3.x
- NumPy
- Matplotlib
- OpenCV (cv2)
- Pandas
- SciPy
- tkinter

## 설치

```bash
pip install numpy matplotlib opencv-python pandas scipy
```

## 사용 방법

1. `directory` 변수를 SPE 파일이 있는 경로로 수정합니다:
   ```python
   directory = r"YOUR_PATH_HERE"
   ```

2. 스크립트를 실행합니다:
   ```bash
   python "CCD img analysis_and_intensity.py"
   ```

3. 인터랙티브 조정:
   - 키보드 단축키로 설정을 조정합니다
   - 더블클릭으로 프레임을 변경합니다
   - Colorbar 더블클릭으로 강도 범위를 조정합니다

## 기능 설명

### 타원 피팅

- **1/e² 타원**: 최대 강도의 1/e² 기준으로 타원 피팅
- **Major/Minor 축**: 장축과 단축 길이 계산
- **FWHM 계산**: Full-Width at Half-Maximum 계산
- **Power 계산**: 타원 내부 총 강도 계산

### 출력 파일

- `ellipse_fitting_summary_ALL_FRAMES.xlsx`: 모든 프레임 분석 결과
- `ellipse_fitting_summary_multi_frame.xlsx`: 최적 프레임 결과
- `frame_override.json`: 프레임 설정 저장

## 키보드 단축키

- `u`: 픽셀 크기 및 눈금 간격 조정

## 라이선스

MIT License

