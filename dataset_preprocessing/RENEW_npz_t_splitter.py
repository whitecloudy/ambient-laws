import os
import re
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def parse_range(range_str):
    """
    Parses a string or integer/list representing single integers or ranges into a sorted list of ints.
    Examples:
      "-1" -> [-1] (All cells)
      "0" -> [0]
      "0-3" -> [0, 1, 2, 3]
      "0,1,2" -> [0, 1, 2]
      "0-2,4,6-8" -> [0, 1, 2, 4, 6, 7, 8]
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
        raise argparse.ArgumentTypeError(f"Invalid cell range specification: '{part}'")
    if not result:
        raise argparse.ArgumentTypeError(f"No valid integers found in input: '{range_str}'")
    return sorted(list(result))

def process_file(args):
    """
    Processes a single npz file:
    1. Parses starting frame index 'num' from the filename.
    2. Loads 'csi' and 'noise' data.
    3. Decomposes the T dimension (time/frame index t).
    4. Saves each frame separately as frame<num+t>_user<user>_cell<cell>_subcarrier0.npz in output_dir.
    """
    file_path, output_dir, user, target_cells = args
    filename = os.path.basename(file_path)
    
    # Match dynamically for specified user and any cell, and extract start frame & cell
    match = re.match(rf'^frame(\d+)_user{user}_mean_cell(\d+)_subcarrier0\.npz$', filename)
    if not match:
        return {"status": "skipped", "file": filename, "reason": f"Not matching user{user}_mean_cell"}
        
    start_frame = int(match.group(1))
    cell_num = int(match.group(2))
    
    if -1 not in target_cells and cell_num not in target_cells:
        return {"status": "skipped", "file": filename, "reason": f"Cell {cell_num} not in target cell range {target_cells}"}
    
    try:
        data = np.load(file_path)
        if 'csi' not in data or 'noise' not in data:
            return {"status": "error", "file": filename, "reason": "Missing csi or noise key"}
            
        csi = data['csi']      # Expected shape [T, 8, 52]
        noise = data['noise']  # Expected shape [T, 8]
        
        # Verify shapes
        if csi.ndim != 3 or noise.ndim != 2:
            return {"status": "error", "file": filename, "reason": f"Unexpected shapes: csi={csi.shape}, noise={noise.shape}"}
            
        T = csi.shape[0]
        if noise.shape[0] != T:
            return {"status": "error", "file": filename, "reason": f"Mismatch in T dimension: csi.T={T}, noise.T={noise.shape[0]}"}
            
        # Process and save each t
        for t in range(T):
            csi_t = csi[t:t+1]      # shape [1, 8, 52]
            noise_t = noise[t:t+1]  # shape [1, 8]
            
            out_filename = f"frame{start_frame + t}_user{user}_cell{cell_num}_subcarrier0.npz"
            out_path = os.path.join(output_dir, out_filename)
            
            # Save using np.savez_compressed (consistent with project standard)
            np.savez_compressed(out_path, csi=csi_t, noise=noise_t)
            
        return {"status": "success", "file": filename, "frames": T}
        
    except Exception as e:
        return {"status": "error", "file": filename, "reason": str(e)}

def main():
    parser = argparse.ArgumentParser(description="RENEW npz T-dimension splitter script")
    parser.add_argument("--input-dir", required=True, help="입력 npz 파일들이 있는 디렉토리")
    parser.add_argument("--output-dir", required=True, help="분할된 npz 파일들을 저장할 디렉토리")
    parser.add_argument("--user", type=int, default=1, help="필터링할 user 번호 (default: 1)")
    parser.add_argument("--cell", type=parse_range, default="-1", help="필터링할 cell 번호 또는 범위 (예: -1 (전체), 0, 0-3, 0,1,2) (default: '-1')")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="사용할 병렬 작업자 수 (default: CPU 코어 수)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all npz files (supporting recursive or direct children)
    all_files = glob(os.path.join(args.input_dir, "**", "*.npz"), recursive=True)
    if not all_files:
        # Try non-recursive glob if recursive didn't return anything or directory structure is flat
        all_files = glob(os.path.join(args.input_dir, "*.npz"))
        
    # Eliminate duplicate paths if any
    all_files = sorted(list(set(all_files)))
    
    if not all_files:
        print(f"Error: No npz files found in {args.input_dir}")
        return
        
    # Prepare files for filtering and processing
    process_list = []
    skipped_count = 0
    target_cells = args.cell
    match_all_cells = -1 in target_cells
    cell_pattern = rf'^frame\d+_user{args.user}_mean_cell(\d+)_subcarrier0\.npz$'
    for f in all_files:
        filename = os.path.basename(f)
        match = re.match(cell_pattern, filename)
        if match:
            cell_num = int(match.group(1))
            if match_all_cells or cell_num in target_cells:
                process_list.append(f)
            else:
                skipped_count += 1
        else:
            skipped_count += 1
            
    cell_filter_msg = "all cells" if match_all_cells else f"cell not in {target_cells}"
    print(f"Total npz files found: {len(all_files)}")
    print(f"Skipped (non-user{args.user} or {cell_filter_msg}): {skipped_count}")
    print(f"Files to process: {len(process_list)}")
    
    if not process_list:
        print("No files to process.")
        return
        
    pool_args = [(f, args.output_dir, args.user, target_cells) for f in process_list]
    
    success_count = 0
    error_count = 0
    total_frames_saved = 0
    
    # Process using multiprocessing pool
    with Pool(args.workers) as pool:
        for res in tqdm(pool.imap_unordered(process_file, pool_args), total=len(pool_args), desc="Splitting files"):
            if res["status"] == "success":
                success_count += 1
                total_frames_saved += res["frames"]
            elif res["status"] == "error":
                error_count += 1
                print(f"\nError processing {res['file']}: {res['reason']}")
                
    print("\n" + "="*40)
    print("Processing Completed")
    print(f"Successfully processed files: {success_count}/{len(process_list)}")
    if error_count > 0:
        print(f"Failed files (with errors): {error_count}")
    print(f"Total split frames saved: {total_frames_saved}")
    print("="*40)

if __name__ == "__main__":
    main()
