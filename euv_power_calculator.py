"""
Line Profile 저장 모듈
CCD 이미지 분석 결과의 line profile 데이터를 Excel 파일로 저장하는 기능을 제공합니다.
"""

import pandas as pd
import os


def find_best_frame_for_file(file_data_list):
    """
    파일별로 best frame을 찾습니다.
    Best frame은 Major/Minor Length 비율이 1에 가장 가까운 프레임입니다.
    
    Parameters:
    -----------
    file_data_list : list
        특정 파일의 모든 프레임 데이터 리스트
    
    Returns:
    --------
    dict or None
        Best frame 데이터, 없으면 None
    """
    if not file_data_list:
        return None
    
    valid_frames = []
    for frame_data in file_data_list:
        major = frame_data.get('Major Length (px)')
        minor = frame_data.get('Minor Length (px)')
        frame_num = frame_data.get('Frame', 0)
        
        # Major와 Minor가 모두 있고, Frame > 1인 경우만 고려
        if major is not None and minor is not None and frame_num > 1:
            ratio = minor / major if major > 0 else 0
            ratio_diff = abs(ratio - 1)
            valid_frames.append({
                'frame_data': frame_data,
                'ratio': ratio,
                'ratio_diff': ratio_diff
            })
    
    if not valid_frames:
        return None
    
    # RatioDiff가 가장 작은 프레임이 best frame
    best = min(valid_frames, key=lambda x: x['ratio_diff'])
    return best['frame_data']


def save_line_profiles(all_frames_data, profile_excel_filename=None):
    """
    Line profile 데이터를 Excel 파일로 저장합니다.
    각 CCD 이미지 파일별로 best frame을 찾아 별도의 Excel 파일로 저장합니다.
    
    Parameters:
    -----------
    all_frames_data : list
        모든 프레임의 분석 결과 리스트. 각 요소는 딕셔너리로 다음 키를 포함할 수 있습니다:
        - 'Filename': 파일명
        - 'Frame': 프레임 번호
        - 'Major Length (px)': Major 축 길이 (best frame 찾기용)
        - 'Minor Length (px)': Minor 축 길이 (best frame 찾기용)
        - '_profile_major_distance': Major 축 거리 데이터 (list)
        - '_profile_major_intensity': Major 축 강도 데이터 (list)
        - '_profile_minor_distance': Minor 축 거리 데이터 (list)
        - '_profile_minor_intensity': Minor 축 강도 데이터 (list)
    
    profile_excel_filename : str, optional
        사용되지 않음 (하위 호환성을 위해 유지)
    
    Returns:
    --------
    bool
        저장 성공 여부
    """
    print(f"\n[Line Profile Saver] 저장 시작: 각 파일별로 Best Frame 찾아 별도 엑셀 파일로 저장")
    print(f"  처리할 프레임 데이터 개수: {len(all_frames_data) if all_frames_data else 0}")
    
    try:
        # 파일별로 데이터 그룹화
        data_by_file = {}  # {filename: [all frame data]}
        
        for idx, frame_data in enumerate(all_frames_data):
            if frame_data is None:
                print(f"  [{idx+1}] 프레임 데이터가 None입니다. 건너뜁니다.")
                continue
            
            filename = frame_data.get('Filename', '')
            if not filename:
                print(f"  [{idx+1}] 파일명이 없습니다. 건너뜁니다.")
                continue
            
            if filename not in data_by_file:
                data_by_file[filename] = []
            
            data_by_file[filename].append(frame_data)
        
        print(f"\n  수집된 파일 수: {len(data_by_file)}")
        
        # 통합 파일명 생성
        if profile_excel_filename:
            excel_filename = profile_excel_filename
        else:
            excel_filename = "line_profiles_BestFrame.xlsx"
        
        print(f"\n  통합 엑셀 파일 생성: {excel_filename}")
        
        # 모든 파일의 데이터를 먼저 수집
        sheets_data = {}  # {sheet_name: DataFrame}
        
        # 각 파일별로 best frame 찾기 및 데이터 준비
        for filename, file_data_list in data_by_file.items():
            print(f"\n  [{filename}] 처리 중...")
            
            # Best frame 찾기
            best_frame_data = find_best_frame_for_file(file_data_list)
            
            if best_frame_data is None:
                print(f"    ✗ Best frame을 찾을 수 없습니다.")
                continue
            
            best_frame_num = best_frame_data.get('Frame', 0)
            print(f"    ✓ Best Frame: {best_frame_num}")
            
            # Best frame의 profile 데이터 추출
            major_dist = None
            major_int = None
            minor_dist = None
            minor_int = None
            
            # Major profile
            has_major = '_profile_major_distance' in best_frame_data and '_profile_major_intensity' in best_frame_data
            if has_major:
                major_dist = best_frame_data['_profile_major_distance']
                major_int = best_frame_data['_profile_major_intensity']
                if major_dist and major_int and len(major_dist) == len(major_int):
                    print(f"      ✓ Major profile: {len(major_dist)} points")
                else:
                    print(f"      ✗ Major profile: 데이터 형식 오류")
                    major_dist = None
                    major_int = None
            else:
                print(f"      ✗ Major profile: 데이터 없음")
            
            # Minor profile
            has_minor = '_profile_minor_distance' in best_frame_data and '_profile_minor_intensity' in best_frame_data
            if has_minor:
                minor_dist = best_frame_data['_profile_minor_distance']
                minor_int = best_frame_data['_profile_minor_intensity']
                if minor_dist and minor_int and len(minor_dist) == len(minor_int):
                    print(f"      ✓ Minor profile: {len(minor_dist)} points")
                else:
                    print(f"      ✗ Minor profile: 데이터 형식 오류")
                    minor_dist = None
                    minor_int = None
            else:
                print(f"      ✗ Minor profile: 데이터 없음")
            
            # Profile 데이터가 하나라도 있으면 시트 데이터로 추가
            if (major_dist is not None and major_int is not None) or (minor_dist is not None and minor_int is not None):
                # 하나의 시트에 4개 열로 병합
                # 최대 길이 찾기
                max_len = 0
                if major_dist is not None:
                    max_len = max(max_len, len(major_dist))
                if minor_dist is not None:
                    max_len = max(max_len, len(minor_dist))
                
                # 데이터 준비 (길이 맞추기)
                data_dict = {}
                
                if major_dist is not None and major_int is not None:
                    # Major 데이터를 최대 길이에 맞춤
                    major_dist_padded = list(major_dist) + [None] * (max_len - len(major_dist))
                    major_int_padded = list(major_int) + [None] * (max_len - len(major_int))
                    data_dict['Major Distance (μm)'] = major_dist_padded
                    data_dict['Major Intensity'] = major_int_padded
                else:
                    # Major 데이터가 없으면 빈 열
                    data_dict['Major Distance (μm)'] = [None] * max_len
                    data_dict['Major Intensity'] = [None] * max_len
                
                if minor_dist is not None and minor_int is not None:
                    # Minor 데이터를 최대 길이에 맞춤
                    minor_dist_padded = list(minor_dist) + [None] * (max_len - len(minor_dist))
                    minor_int_padded = list(minor_int) + [None] * (max_len - len(minor_int))
                    data_dict['Minor Distance (μm)'] = minor_dist_padded
                    data_dict['Minor Intensity'] = minor_int_padded
                else:
                    # Minor 데이터가 없으면 빈 열
                    data_dict['Minor Distance (μm)'] = [None] * max_len
                    data_dict['Minor Intensity'] = [None] * max_len
                
                # DataFrame 생성
                df_combined = pd.DataFrame(data_dict)
                
                # 시트 이름은 파일명 (확장자 제거, 최대 31자 제한)
                # Excel 시트 이름에 사용할 수 없는 문자 제거
                base_sheet_name = os.path.splitext(filename)[0]
                # Excel 시트 이름에 사용할 수 없는 문자 제거: [ ] : * ? / \
                invalid_chars = ['[', ']', ':', '*', '?', '/', '\\']
                for char in invalid_chars:
                    base_sheet_name = base_sheet_name.replace(char, '_')
                
                sheet_name = base_sheet_name[:31]
                
                # 시트 이름 중복 방지
                original_sheet_name = sheet_name
                counter = 1
                while sheet_name in sheets_data:
                    # 중복되면 번호 추가 (최대 31자 유지)
                    suffix = f"_{counter}"
                    max_base_len = 31 - len(suffix)
                    sheet_name = original_sheet_name[:max_base_len] + suffix
                    counter += 1
                    if counter > 999:  # 무한 루프 방지
                        break
                
                sheets_data[sheet_name] = df_combined
                print(f"      ✓ 시트 데이터 준비 완료: {sheet_name} ({max_len} rows, 4 columns)")
            else:
                print(f"      ✗ 저장할 profile 데이터가 없습니다.")
        
        # 통합 엑셀 파일 생성
        if sheets_data:
            print(f"\n  총 {len(sheets_data)}개의 시트를 생성합니다:")
            for sheet_name in sheets_data.keys():
                print(f"    - {sheet_name}")
            
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                sheet_count = 0
                for sheet_name, df in sheets_data.items():
                    try:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        sheet_count += 1
                        print(f"    ✓ 시트 생성 완료 [{sheet_count}/{len(sheets_data)}]: {sheet_name} ({len(df)} rows)")
                    except Exception as e_sheet:
                        print(f"    ✗ 시트 생성 실패: {sheet_name} - {e_sheet}")
                        import traceback
                        traceback.print_exc()
            
            print(f"\n[Line Profile Saver] 저장 완료: {excel_filename}")
            print(f"  생성된 시트 수: {sheet_count}/{len(sheets_data)}")
            return True
        else:
            print(f"\n[Line Profile Saver] 저장할 line profile 데이터가 없습니다.")
            return False
            
    except Exception as e:
        print(f"\n[Line Profile Saver] 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

