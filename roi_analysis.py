"""
ROI Analysis Module
다양한 빔 모양에 대한 ROI 계산 모듈
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List


def fit_ellipse_roi(image, th_e2, peak_y, peak_x):
    """
    타원 형태 빔에 대한 ROI 계산
    
    Args:
        image: 입력 이미지
        th_e2: 1/e² 임계값
        peak_y, peak_x: 피크 위치
    
    Returns:
        중심, 장축/단축, 각도, 마스크
    """
    mask_e2 = (image > th_e2).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_e2, connectivity=8)
    
    if num_labels <= 1:
        return None
    
    peak_label = labels[int(peak_y), int(peak_x)]
    if peak_label == 0 and num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) > 0:
            peak_label = np.argmax(areas) + 1
    
    if peak_label != 0:
        mask_peak_e2 = (labels == peak_label).astype(np.uint8)
        contours_e2, _ = cv2.findContours(mask_peak_e2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_e2:
            cnt = max(contours_e2, key=cv2.contourArea)
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (maj, minr), angle = ellipse
                
                # Swap if needed
                if maj < minr:
                    maj, minr = minr, maj
                    angle = (angle + 90) % 180
                
                return (cx, cy), (maj, minr), angle, mask_peak_e2
    
    return None


def fit_fan_shape_roi(image, th_e2, peak_y, peak_x):
    """
    부채꼴 형태 빔에 대한 ROI 계산
    
    Args:
        image: 입력 이미지
        th_e2: 임계값
        peak_y, peak_x: 피크 위치
    
    Returns:
        contour, mask, (cx, cy), angle
    """
    mask_e2 = (image > th_e2).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_e2, connectivity=8)
    
    if num_labels <= 1:
        return None
    
    peak_label = labels[int(peak_y), int(peak_x)]
    if peak_label == 0 and num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) > 0:
            peak_label = np.argmax(areas) + 1
    
    if peak_label != 0:
        mask_peak = (labels == peak_label).astype(np.uint8)
        contours, _ = cv2.findContours(mask_peak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            
            # 부채꼴의 중심 계산
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(peak_x), int(peak_y)
            
            # 각도 계산 (피크와 중심을 기준)
            angle = np.arctan2(peak_y - cy, peak_x - cx) * 180 / np.pi
            
            return cnt, mask_peak, (cx, cy), angle
    
    return None


def get_fan_ellipse_mask(roi, cx, cy, contour):
    """
    부채꼴 contour를 마스크로 변환
    
    Args:
        roi: ROI 이미지
        cx, cy: 중심점
        contour: contour 데이터
    
    Returns:
        마스크 (boolean array)
    """
    if contour is None:
        return None
    
    # Contour를 ROI의 픽셀 좌표로 변환
    mask = np.zeros(roi.shape, dtype=bool)
    cnt_points = contour.reshape(-1, 2).astype(int)
    
    # Contour 내부를 True로 설정
    cv2.fillPoly(mask, [cnt_points], True)
    
    return mask


def compute_fitted_fan_shape(contour, cx, cy, angle):
    """
    부채꼴 contour를 smoothing하여 fitted shape 계산
    원본 contour를 부드럽게 만듦
    
    Args:
        contour: contour 데이터
        cx, cy: 중심점
        angle: 중심 각도 (도)
    
    Returns:
        fitted_points: fitted fan shape의 좌표 배열 (Nx2)
    """
    if contour is None or len(contour) < 3:
        return None
    
    # Contour를 numpy 배열로 변환
    cnt_points = contour.reshape(-1, 2).astype(float)
    
    # 원본 contour를 그대로 사용
    fitted_points = cnt_points
    
    # Gaussian smoothing 적용 (부드러운 곡선을 위해)
    if len(fitted_points) > 4:
        
        # 이웃 점들의 평균 사용 (moving average)
        n = len(fitted_points)
        smoothed = np.zeros_like(fitted_points)
        
        # Gaussian-like smoothing (양쪽 2개씩 확인)
        for i in range(n):
            weights = [0.15, 0.35, 0.50, 0.35, 0.15]
            weighted_sum = np.zeros(2)
            
            for j, weight in enumerate(weights):
                idx = (i + j - 2) % n  # -2 to +2
                weighted_sum += fitted_points[idx] * weight
            
            smoothed[i] = weighted_sum
        
        fitted_points = smoothed
    
    # 점 수를 늘려서 더 부드러운 곡선 생성
    if len(fitted_points) < 80:
        new_points = []
        for i in range(len(fitted_points)):
            next_i = (i + 1) % len(fitted_points)
            # 각 구간을 4개 점으로 분할
            for j in range(4):
                t = j / 4.0
                x = fitted_points[i][0] * (1 - t) + fitted_points[next_i][0] * t
                y = fitted_points[i][1] * (1 - t) + fitted_points[next_i][1] * t
                new_points.append([x, y])
        
        fitted_points = np.array(new_points)
    
    return fitted_points


def detect_beam_shape(mask_contour) -> str:
    """
    빔의 모양을 자동 감지
    
    Args:
        mask_contour: contour 영역
    
    Returns:
        'ellipse' 또는 'fan' 또는 'irregular'
    """
    if mask_contour is None or len(mask_contour) < 5:
        return 'irregular'
    
    # 타원 피팅 시도
    try:
        ellipse = cv2.fitEllipse(mask_contour)
        (cx, cy), (maj, minr), angle = ellipse
        area = cv2.contourArea(mask_contour)
        
        # 타원 면적과 실제 면적 비교
        ellipse_area = np.pi * (maj/2) * (minr/2)
        area_ratio = area / ellipse_area if ellipse_area > 0 else 1.0
        
        # 비율이 0.7 이상이면 타원으로 판단
        if area_ratio > 0.7:
            return 'ellipse'
        else:
            return 'fan'
    except:
        return 'irregular'


def calculate_roi_complex(roi, th_e2, peak_y_roi, peak_x_roi):
    """
    다양한 형태의 빔에 대한 ROI 계산
    
    Args:
        roi: ROI 이미지
        th_e2: 임계값
        peak_y_roi, peak_x_roi: 피크 위치
    
    Returns:
        ROI 정보 (타입에 따라 다른 형태)
    """
    # 먼저 타원 시도
    ellipse_result = fit_ellipse_roi(roi, th_e2, peak_y_roi, peak_x_roi)
    
    if ellipse_result is None:
        # 타원 실패 시 부채꼴 시도
        fan_result = fit_fan_shape_roi(roi, th_e2, peak_y_roi, peak_x_roi)
        return fan_result
    else:
        return ellipse_result


def calculate_integrated_intensity(roi, mask):
    """
    마스크 내부의 intensity 합계 계산 (CCD Counts)
    
    Args:
        roi: ROI 이미지
        mask: 마스크
    
    Returns:
        intensity 합계
    """
    if np.any(mask):
        return float(np.sum(roi[mask]))
    return 0.0

