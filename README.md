# CCD 이미지 분석 및 강도 측정 프로그램

SPE 파일 형식의 CCD 이미지를 분석하고 빔 프로파일을 측정하는 Python 프로그램입니다.

## 주요 기능

- **SPE 파일 분석**: WinView/Princeton Instruments SPE 파일 형식 지원
- **자동 ROI 감지**: 이미지에서 빔 영역을 자동으로 감지
- **빔 모양 피팅**:
  - 타원 피팅 (핀홀 사용 시)
  - 부채꼴 피팅 (핀홀 없을 때)
- **Line Profile 분석**: Major/Minor 축을 따라 강도 프로파일 추출
- **FWHM 계산**: Full-Width at Half-Maximum 자동 계산
- **Excel 출력**: 분석 결과 및 line profile 데이터를 엑셀 파일로 저장
- **인터랙티브 GUI**: 프레임 변경, 축 설정, 컬러바 조정 등

## 설치 방법

### 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install numpy matplotlib opencv-python pandas scipy openpyxl
```

### 필수 라이브러리

- `numpy` >= 1.20.0
- `matplotlib` >= 3.3.0
- `opencv-python` >= 4.5.0
- `pandas` >= 1.2.0
- `scipy` >= 1.6.0
- `openpyxl` >= 3.0.0

### 선택적 모듈

- `roi_analysis`: 고급 ROI 분석 기능 (없어도 기본 기능 사용 가능)
- `euv_power_calculator`: EUV 파워 계산기 (없어도 기본 기능 사용 가능)

## 사용 방법

### 1. 프로그램 실행

```bash
python "CCD img analysis_and_intensity.py"
```

### 2. 파일 선택

프로그램 실행 시 파일 선택 다이얼로그가 나타납니다. 분석할 SPE 파일을 선택하세요.

### 3. 빔 모양 선택

프로그램 시작 시 빔 모양을 선택합니다:
- **YES (타원 피팅)**: 핀홀이 있는 경우
- **NO (부채꼴 피팅)**: 핀홀이 없는 경우

### 4. 분석 결과 확인

프로그램은 자동으로:
1. 모든 프레임을 사전 분석하여 최적 프레임 선택
2. 선택된 프레임을 시각화
3. 분석 결과를 엑셀 파일로 저장

## 키보드 단축키

- **`u`**: 축 설정 변경 (픽셀 크기, tick 간격)
- **`s`**: 빔 모양 모드 전환 (타원 ↔ 부채꼴)
- **이미지 더블 클릭**: 프레임 변경
- **컬러바 더블 클릭**: 강도 범위 수동 조정

## 출력 파일

### 1. 요약 데이터 (Summary)

- **타원 모드**: `ellipse_fitting_summary_multi_frame.xlsx`
- **부채꼴 모드**: `nonellipse_fitting_summary_multi_frame.xlsx`

**포함 데이터:**
- 파일명, 프레임 번호
- Major/Minor 길이 (픽셀 및 마이크로미터)
- FWHM 값 (Major/Minor)
- 면적 (μm²)
- CCD Counts
- 기타 분석 파라미터

### 2. 전체 프레임 분석 결과

- **타원 모드**: `ellipse_fitting_summary_ALL_FRAMES.xlsx`
- **부채꼴 모드**: `nonellipse_fitting_summary_ALL_FRAMES.xlsx`

모든 프레임에 대한 분석 결과를 포함합니다.

### 3. Line Profile 데이터

- **타원 모드**: `ellipse_line_profiles.xlsx`
- **부채꼴 모드**: `nonellipse_line_profiles.xlsx`

**파일별로 시트가 분리되어 저장됩니다:**
- 각 시트: 파일명 (확장자 제외)
- 컬럼: Frame, Axis (Major/Minor), Distance (μm), Intensity

### 4. 설정 파일

- `frame_override.json`: 각 파일의 최적 프레임 정보 저장

## 주요 파라미터

### 전역 변수 (코드 내 수정 가능)

```python
PIXEL_SIZE = 0.5        # 픽셀 크기 (μm/px)
TICK_INTERVAL = 5.0     # 축 tick 간격 (μm)
BEAM_SHAPE_MODE = 'ellipse'  # 'ellipse' 또는 'fan'
```

### 시각화 설정

```python
COLORMAP = 'viridis'           # 컬러맵
INTENSITY_NORM = 'individual'  # 'individual', 'global', 'manual'
AUTO_SCALE = True              # 자동 스케일링
```

## 분석 기능

### 1. 자동 ROI 감지

이미지에서 빔 영역을 자동으로 감지하고 ROI를 설정합니다.

### 2. 1/e² 타원/부채꼴 피팅

최대 강도의 1/e² 지점에서 타원 또는 부채꼴을 피팅합니다.

### 3. FWHM 계산

Line profile을 기반으로 Full-Width at Half-Maximum을 계산합니다.

### 4. 최적 프레임 자동 선택

모든 프레임을 분석하여 Major/Minor 비율이 가장 1에 가까운 프레임을 자동으로 선택합니다.

## 문제 해결

### ImportError 발생 시

필수 라이브러리가 설치되지 않았을 수 있습니다:

```bash
pip install -r requirements.txt
```

### roi_analysis 모듈 없음

기본 타원 피팅 기능만 사용됩니다. 경고 메시지는 무시해도 됩니다.

### euv_power_calculator 모듈 없음

EUV 파워 계산기 기능만 사용할 수 없습니다. 기본 분석 기능은 정상 작동합니다.

## 라이선스

이 프로그램은 연구 및 교육 목적으로 제공됩니다.

## 문의

문제가 발생하거나 개선 사항이 있으면 이슈를 등록해주세요.
