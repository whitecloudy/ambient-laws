import os
import re
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from glob import glob

def parse_range(range_str):
    """
    Parses a string or integer/list representing single integers or ranges into a sorted list of ints.
    Examples:
      "-1" -> [-1] (All items)
      "0" -> [0]
      "0-3" -> [0, 1, 2, 3]
      "0,1,2" -> [0, 1, 2]
    """
    if isinstance(range_str, int):
        return [range_str]
    if isinstance(range_str, (list, tuple, set)):
        return sorted(list(set(int(x) for x in range_str)))
        
    result = set()
    parts = str(range_str).split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            result.add(int(part))
            continue
        except ValueError:
            pass
            
        if '-' in part:
            idx = part.find('-', 1)
            if idx != -1:
                start_str, end_str = part[:idx].strip(), part[idx+1:].strip()
                try:
                    start, end = int(start_str), int(end_str)
                    if start > end:
                        start, end = end, start
                    result.update(range(start, end + 1))
                    continue
                except ValueError:
                    pass
        raise argparse.ArgumentTypeError(f"Invalid range specification: '{part}'")
    if not result:
        raise argparse.ArgumentTypeError(f"No valid integers found in input: '{range_str}'")
    return sorted(list(result))

def parse_int_list(val_str):
    """
    Parses comma-separated integers or integer list.
    Examples:
      "5,19,32,46" -> [5, 19, 32, 46]
    """
    if not val_str or str(val_str).strip().lower() == 'none':
        return []
    if isinstance(val_str, (list, tuple, set)):
        return [int(x) for x in val_str]
    result = []
    for part in str(val_str).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            result.append(int(part))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid integer in list: '{part}'")
    return result

def process_file(args):
    """
    하나의 파일 쌍(.csi.npy, .noise.npy)을 처리하여 .npz 청크로 분할합니다.
    """
    prefix, output_dir, frame_size, cell_size, subcarrier_size, target_users, exclude_subcarriers = args
    
    csi_path = f"{prefix}.csi.npy"
    noise_path = f"{prefix}.noise.npy"

    if not (os.path.exists(csi_path) and os.path.exists(noise_path)):
        print(f"Warning: Skipping {prefix} because a file is missing.")
        return

    try:
        csi_data = np.load(csi_path, mmap_mode='r')
        noise_data = np.load(noise_path, mmap_mode='r')
    except Exception as e:
        print(f"Error loading data for {prefix}: {e}")
        return

    if csi_data.ndim == 4 and noise_data.ndim == 3:
        # 데이터 shape: [Frame, User, Cell, Subcarrier] 및 [Frame, User, Cell]
        n_frames, n_users, n_cells, n_subcarriers = csi_data.shape
        pilot_rep = 1
    elif csi_data.ndim == 5 and noise_data.ndim == 3:
        # 데이터 shape: [Frame, User, Pilot_Rep, Cell, Subcarrier] 및 [Frame, User, Cell]
        n_frames, n_users, pilot_rep, n_cells, n_subcarriers = csi_data.shape
    else:
        print(f"Error: Unexpected data shape for {prefix}")
        return

    # 제외할 subcarrier를 제외한 유효 subcarrier 인덱스 필터링
    valid_subcarrier_indices = [i for i in range(n_subcarriers) if i not in exclude_subcarriers]
    n_valid_subcarriers = len(valid_subcarrier_indices)

    # 출력 디렉토리 생성
    file_prefix = os.path.basename(prefix)
    target_dir = os.path.join(output_dir, file_prefix)
    os.makedirs(target_dir, exist_ok=True)

    match_all_users = -1 in target_users

    # 데이터 분할 및 저장
    for u_start in range(n_users): # User 단위 처리
        if not match_all_users and u_start not in target_users:
            continue
        for p_start in range(pilot_rep):
            for f_start in range(0, n_frames, frame_size):
                f_end = min(f_start + frame_size, n_frames)
                if f_end - f_start < frame_size: continue # 꽉 찬 프레임만 사용

                for c_start in range(0, n_cells, cell_size):
                    c_end = min(c_start + cell_size, n_cells)
                    if c_end - c_start < cell_size: continue # 꽉 찬 셀만 사용
                    for s_start in range(0, n_valid_subcarriers, subcarrier_size):
                        s_end = min(s_start + subcarrier_size, n_valid_subcarriers)
                        if s_end - s_start < subcarrier_size: continue # 꽉 찬 subcarrier만 사용
                        
                        sub_indices = valid_subcarrier_indices[s_start:s_end]

                        # 데이터 청크 추출 (기본 슬라이싱 후 유효 subcarrier 인덱싱)
                        if csi_data.ndim == 5:
                            csi_chunk = csi_data[f_start:f_end, u_start, p_start, c_start:c_end][..., sub_indices]
                        elif csi_data.ndim == 4:
                            csi_chunk = csi_data[f_start:f_end, u_start, c_start:c_end][..., sub_indices]

                        # noise 데이터는 subcarrier, pilot repetition 차원이 없음
                        noise_chunk = noise_data[f_start:f_end, u_start, c_start:c_end]
                        
                        # 파일명 생성 및 저장
                        output_filename = f"frame{f_start:05d}_user{u_start}_pilot{p_start}_cell{c_start:02d}_subcarrier{s_start}.npz"
                        output_path = os.path.join(target_dir, output_filename)
                        
                        np.savez_compressed(output_path, csi=csi_chunk, noise=noise_chunk)

def main():
    parser = argparse.ArgumentParser(description="RENEW 데이터셋 분할 스크립트")
    parser.add_argument("--input-dir", required=True, help="입력 .csi.npy와 .noise.npy 파일이 있는 디렉토리")
    parser.add_argument("--output-dir", required=True, help="출력 .npz 파일들을 저장할 최상위 디렉토리")
    parser.add_argument("--user", type=parse_range, default="-1", help="필터링할 user 번호 또는 범위 (예: -1 (전체), 0, 1, 0-3) (default: '-1')")
    # [5, 19, 32, 46]은 실제 WiFi에서 파일럿 subcarrier로 쓰이는 -21, -7, 7, 21번에 해당함
    parser.add_argument("--exclude-subcarriers", type=parse_int_list, default="5,19,32,46", help="제외할 subcarrier 인덱스 목록 (예: 5,19,32,46) (default: '5,19,32,46')")
    parser.add_argument("--frame-size", type=int, default=14, help="분할할 Frame 차원의 크기")
    parser.add_argument("--cell-size", type=int, default=8, help="분할할 Cell 차원의 크기")
    parser.add_argument("--subcarrier-size", type=int, default=52, help="분할할 Subcarrier 차원의 크기")
    parser.add_argument("--workers", type=int, default=1, help="사용할 병렬 작업자 수")
    
    args = parser.parse_args()

    # 입력 디렉토리에서 파일 목록 찾기
    csi_files = glob(os.path.join(args.input_dir, '*.csi.npy'))
    if not csi_files:
        print(f"Error: No .csi.npy files found in {args.input_dir}")
        return
        
    prefixes = sorted([f.replace('.csi.npy', '') for f in csi_files])
    
    print(f"총 {len(prefixes)}개의 파일 쌍을 처리합니다.")
    print(f"선택된 user: {args.user}")
    print(f"제외할 subcarrier: {args.exclude_subcarriers}")
    print(f"사용할 작업자 수: {args.workers}")
    
    # 멀티프로세싱 풀 생성
    pool_args = [
        (prefix, args.output_dir, args.frame_size, args.cell_size, args.subcarrier_size, args.user, args.exclude_subcarriers)
        for prefix in prefixes
    ]
    
    with Pool(args.workers) as p:
        list(tqdm(p.imap_unordered(process_file, pool_args), total=len(prefixes), desc="파일 처리 중"))

    print("모든 파일 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()
