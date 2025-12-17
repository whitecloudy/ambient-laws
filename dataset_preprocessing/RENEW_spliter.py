
import os
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from glob import glob

def process_file(args):
    """
    하나의 파일 쌍(.csi.npy, .noise.npy)을 처리하여 .npz 청크로 분할합니다.
    """
    prefix, output_dir, frame_size, cell_size, subcarrier_size = args
    
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

    # 데이터 shape: [Frame, User, Cell, Subcarrier] 및 [Frame, User, Cell]
    n_frames, n_users, n_cells, n_subcarriers = csi_data.shape
    
    # 출력 디렉토리 생성
    file_prefix = os.path.basename(prefix)
    target_dir = os.path.join(output_dir, file_prefix)
    os.makedirs(target_dir, exist_ok=True)

    # 데이터 분할 및 저장
    for u_start in range(n_users): # User는 1개 단위로 처리
        for f_start in range(0, n_frames, frame_size):
            f_end = min(f_start + frame_size, n_frames)
            if f_end - f_start < frame_size: continue # 꽉 찬 프레임만 사용

            for c_start in range(0, n_cells, cell_size):
                c_end = min(c_start + cell_size, n_cells)
                if c_end - c_start < cell_size: continue # 꽉 찬 셀만 사용

                for s_start in range(0, n_subcarriers, subcarrier_size):
                    s_end = min(s_start + subcarrier_size, n_subcarriers)
                    if s_end - s_start < subcarrier_size: continue # 꽉 찬 subcarrier만 사용
                    
                    # 데이터 청크 추출
                    csi_chunk = csi_data[f_start:f_end, u_start, c_start:c_end, s_start:s_end]
                    # noise 데이터는 subcarrier 차원이 없음
                    noise_chunk = noise_data[f_start:f_end, u_start, c_start:c_end]
                    
                    # 파일명 생성 및 저장
                    output_filename = f"{f_start}_{u_start}_{c_start}_{s_start}.npz"
                    output_path = os.path.join(target_dir, output_filename)
                    
                    np.savez_compressed(output_path, csi=csi_chunk, noise=noise_chunk)

def main():
    parser = argparse.ArgumentParser(description="RENEW 데이터셋 분할 스크립트")
    parser.add_argument("--input-dir", required=True, help="입력 .csi.npy와 .noise.npy 파일이 있는 디렉토리")
    parser.add_argument("--output-dir", required=True, help="출력 .npz 파일들을 저장할 최상위 디렉토리")
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
    print(f"사용할 작업자 수: {args.workers}")
    
    # 멀티프로세싱 풀 생성
    pool_args = [(prefix, args.output_dir, args.frame_size, args.cell_size, args.subcarrier_size) for prefix in prefixes]
    
    with Pool(args.workers) as p:
        list(tqdm(p.imap_unordered(process_file, pool_args), total=len(prefixes), desc="파일 처리 중"))

    print("모든 파일 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()
