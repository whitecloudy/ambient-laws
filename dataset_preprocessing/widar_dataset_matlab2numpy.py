import numpy as np
import scipy.io as sio
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import click
import csv
import multiprocessing

def extract_date_and_userN_from_dir(dir):
    dir_split = os.path.normpath(dir).split(os.sep)
    date = int(dir_split[-3])
    userN = int(dir_split[-2].replace("user", ""))
    return date, userN

def extract_label_from_filename(filename):
    splited_filename = filename.split('.')[0].split('-')
    user_id = int(splited_filename[0].replace("user", ""))
    raw_gesture_id = int(splited_filename[1])
    torso_location = int(splited_filename[2])
    face_orientation = int(splited_filename[3])
    repitition = int(splited_filename[4])
    rx_id = int(splited_filename[5].replace("r", ""))
    return {"user_id": user_id, 
            "raw_gesture_id": raw_gesture_id, 
            "torso_location": torso_location, 
            "face_orientation": face_orientation, 
            "repitition": repitition, 
            "rx_id": rx_id}

def reassembling_filename(date, label_info):
    filename = f"{date}-user{label_info['user_id']}-gesture{label_info['gesture_id']}-torso{label_info['torso_location']}-face{label_info['face_orientation']}-rep{label_info['repitition']}-rx{label_info['rx_id']}.npz"
    return filename


def process_file(file_path, save_dir, meta_df):
    # Key : ["csi_data", "time", "noise_array"]
    data = sio.loadmat(file_path)

    date, userN = extract_date_and_userN_from_dir(file_path)
    filename = os.path.basename(file_path)
    label_info = extract_label_from_filename(filename)

    csi_data = data['csi_data'] 
    time = data['time']
    noise_array = data['noise_array']

    gesture_meta_data = meta_df[(meta_df['date'] == date) & (meta_df['user'] == userN)]
    assert not gesture_meta_data.empty, f"No metadata found for date {date} and user {userN}"
    
    filtered_columns = [col for col in gesture_meta_data.columns if col not in ['date', 'user']]
    target_id = label_info["raw_gesture_id"]
    
    col_index = -1
    row_data = gesture_meta_data.iloc[0]
    for i, col in enumerate(filtered_columns):
        if row_data[col] == target_id:
            col_index = i
            break
            
    assert col_index != -1, f"Could not find column with value {target_id} for date {date} and user {userN}"

    label_info['gesture_id'] = col_index

    reassembled_filename = reassembling_filename(date, label_info)
    save_path = os.path.join(save_dir, reassembled_filename)
    np.savez_compressed(save_path, csi_data=csi_data, time=time, noise_array=noise_array)

def process_chunk(chunk_file_list, save_dir, meta_df):
    print(f"Processing chunk with {len(chunk_file_list)} files...")
    for file_path in chunk_file_list:
        process_file(file_path, save_dir, meta_df)


@click.command()
@click.option('--dir', default="../../nas_archive/Dataset/Widar dataset/matlab processed csi", help='Directory of the raw data')
@click.option('--save_dir', default="../data/widar_preprocess", help='Directory to save the preprocessed data')
@click.option('--n_proc', default=4, help='Number of processes to use for preprocessing')
@click.option('--process_chunk_size', default=128, help='Number of files to process in each chunk')
@click.option('--meta_csv', default=None, help='Path to save the metadata csv file')
def __main__(dir, save_dir, n_proc, process_chunk_size, meta_csv):
    scan_file_list = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('.mat'):
                scan_file_list.append(os.path.join(root, file))
    scan_file_list = sorted(scan_file_list)
    total_len = len(scan_file_list)

    if meta_csv is None:
        meta_csv = save_dir+"/widar_data_label.csv"

    os.makedirs(save_dir, exist_ok=True)
    
    meta_df = pd.read_csv(meta_csv)

    with tqdm(total=total_len) as pbar:
        for i in range(0, total_len, process_chunk_size*n_proc):
            chunk_file_list_split = [scan_file_list[min(i+j*process_chunk_size, total_len):i+min((j+1)*process_chunk_size, total_len)] for j in range(n_proc)]
            total_len_chunk = sum(len(chunk) for chunk in chunk_file_list_split)
            with multiprocessing.Pool(n_proc) as pool:
                pool.starmap(process_chunk, [(chunk_file_list, save_dir, meta_df) for chunk_file_list in chunk_file_list_split])
            pbar.update(total_len_chunk)



if __name__ == "__main__":
    __main__()