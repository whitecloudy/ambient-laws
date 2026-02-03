import os
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

def compare_files(args):
    """
    두 개의 .npz 파일을 로드하여 csi 및 noise 데이터가 일치하는지 확인합니다.
    """
    file_path_a, file_path_b = args
    
    if not os.path.exists(file_path_b):
        return f"MISSING: {os.path.basename(file_path_a)} (Not found in old directory)"

    try:
        with np.load(file_path_a) as data_a, np.load(file_path_b) as data_b:
            # 키 확인
            if 'csi' not in data_a or 'csi' not in data_b:
                 return f"ERROR: 'csi' key missing in {os.path.basename(file_path_a)}"
            
            csi_a = data_a['csi']
            csi_b = data_b['csi']
            
            if not np.array_equal(csi_a, csi_b):
                return f"FAIL: CSI mismatch in {os.path.basename(file_path_a)}"
            
            # # Noise 데이터 비교 (존재하는 경우)
            # if 'noise' in data_a and 'noise' in data_b:
            #     noise_a = data_a['noise']
            #     noise_b = data_b['noise']
            #     if not np.array_equal(noise_a, noise_b):
            #         return f"FAIL: Noise mismatch in {os.path.basename(file_path_a)}"

    except Exception as e:
        return f"ERROR: {os.path.basename(file_path_a)} - {str(e)}"

    return "PASS"

def main():
    parser = argparse.ArgumentParser(description="두 RENEW .npz 데이터셋 디렉토리 비교 검증")
    parser.add_argument("--new-dir", required=True, help="새로 생성된 데이터셋 디렉토리 경로")
    parser.add_argument("--old-dir", required=True, help="기존(비교 대상) 데이터셋 디렉토리 경로")
    parser.add_argument("--workers", type=int, default=4, help="병렬 처리 프로세스 수")
    
    args = parser.parse_args()

    # new_dir 내의 모든 .npz 파일 찾기 (하위 디렉토리 포함)
    print(f"Searching for .npz files in {args.new_dir}...")
    new_files = glob(os.path.join(args.new_dir, "**", "*.npz"), recursive=True)
    
    if not new_files:
        print(f"Error: No .npz files found in {args.new_dir}")
        return

    print(f"Found {len(new_files)} files. Starting comparison...")

    # 비교할 파일 쌍 리스트 생성
    pool_args = []
    for f_path in new_files:
        rel_path = os.path.relpath(f_path, args.new_dir)
        old_f_path = os.path.join(args.old_dir, rel_path)
        pool_args.append((f_path, old_f_path))

    pass_count = 0
    fail_count = 0
    missing_count = 0
    error_count = 0

    with Pool(args.workers) as p:
        for result in tqdm(p.imap_unordered(compare_files, pool_args), total=len(pool_args)):
            if result == "PASS":
                pass_count += 1
            elif result.startswith("FAIL"):
                fail_count += 1
                print(f"\n{result}")
            elif result.startswith("MISSING"):
                missing_count += 1
            elif result.startswith("ERROR"):
                error_count += 1
                print(f"\n{result}")

    print("\n" + "="*30)
    print(f"검증 완료")
    print(f"총 파일 수: {len(new_files)}")
    print(f"일치 (PASS): {pass_count}")
    print(f"불일치 (FAIL): {fail_count}")
    print(f"누락됨 (MISSING in old dir): {missing_count}")
    print(f"에러 (ERROR): {error_count}")
    print("="*30)

    if fail_count > 0 or error_count > 0:
        exit(1)

if __name__ == "__main__":
    main()
