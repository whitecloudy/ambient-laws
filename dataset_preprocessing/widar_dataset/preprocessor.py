from csi_file_opener import csi_file_process
from multiprocessing import Pool, cpu_count
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np


class BaseData(object):
    def set_num_processes(self, n_proc):
        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


def npy_filename_maker(data_label : dict):
    result = '_'.join(str(value) for value in data_label.values())
    return result


class CsiData_preprocessor(BaseData):
    def __init__(self, dir="../data/widar_dataset/CSI", save_dir="../data/widar_preprocess", n_proc=1, process_chunk_size=128):
        super().__init__()
        self.set_num_processes(n_proc=n_proc)

        self.gesture_label_numbering = pd.read_csv(dir+'/widar_data_label.csv', index_col=['date', 'user'])
        self.gestures_true_id = {str(name) : idx for idx, name in enumerate(self.gesture_label_numbering.columns)}
        self.save_dir = save_dir
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        date_dir_list = glob(dir+"/*")
        # print(glob(dir+"/**/*.dat",recursive=True))
        total_len = sum([len(glob(dir+"/**/*.dat", recursive=True)) for dir in date_dir_list])

        data_label_df_list = []

        with tqdm(total=total_len) as pbar:
            for date_dir in date_dir_list:
                date = date_dir.split("/")[-1]
                dir_list = glob(date_dir+"/**/*.dat", recursive=True)
                step_size = process_chunk_size
                for i in range(0, len(dir_list), process_chunk_size):
                    sub_dir_list = dir_list[i:min(step_size+i, len(dir_list))]

                    data_label_df = self.process_all(sub_dir_list)

                    if len(data_label_df) == 0:
                        continue
                    else:
                        data_label_df_list.append(data_label_df)

                    # data_df.to_pickle(save_dir+"/"+date+"_"+str(i))

                    pbar.update(len(sub_dir_list))

        data_label_df_list = pd.concat(data_label_df_list, ignore_index=True)

        data_label_df_list.to_pickle(save_dir+"/labels.pkl")

    def process_all(self, dir_list):
        data_list = []

        _n_proc = min(self.n_proc, len(dir_list))  # no more than file_names needed here

        input_list = [(dir, self.save_dir, self.gesture_label_numbering, self.gestures_true_id) for dir in dir_list]

        if _n_proc > 1:
            with Pool(processes=_n_proc) as pool:
                data_series = list(pool.map(CsiData_preprocessor.process_single_file_star, input_list))
                data_list += data_series
        else:
            for input in input_list:
                data_series = CsiData_preprocessor.process_single_file_star(input)
                data_list.append(data_series)

        return pd.DataFrame(data_list)

    
    @staticmethod
    def process_single_file_star(args):
        return CsiData_preprocessor.process_single_file(*args)

    @staticmethod
    def process_single_file(dir, save_dir, gesture_label_numbering, gestures_true_id):
        data_backbone, csi_data = csi_file_process(dir)

        def gesture_name_parser(gesture_label_numbering : pd.DataFrame, gestures_true_id, date, user, g_id):
            gesture_n = str(gesture_label_numbering.columns[gesture_label_numbering.loc[date, user]== g_id].values[0])
            num = gestures_true_id[gesture_n]
            return num

        try:
            data_backbone["gesture"] = gesture_name_parser(gesture_label_numbering,
                                                       gestures_true_id,
                                                       data_backbone["date"],
                                                       data_backbone["user"],
                                                       data_backbone["gesture"])
        except IndexError:
            print("Index Error")
            print(dir)
            exit(1)

        csi, time, noise = csi_data

        npy_filename = npy_filename_maker(data_backbone)

        out_dir = "/".join((save_dir, npy_filename))

        np.savez(out_dir, csi=csi, time=time, noise=noise)
        # np.save(out_dir+".csi", csi, allow_pickle=False)
        # np.save(out_dir+".time", time, allow_pickle=False)

        data_backbone["file name"] = npy_filename

        data_backbone = pd.Series(data_backbone)
        
        return data_backbone
    
def fix_labels_file(dir):
    data_label_df_list = pd.DataFrame(pd.read_pickle(dir+"/labels.pkl"))
    # data_label_df_list = data_label_df_list.drop(columns=["file dir",])
    filename_list = []
    for idx, row in data_label_df_list.iterrows():
        filename_list.append(npy_filename_maker(row.to_dict()))

    data_label_df_list['file name'] = filename_list
    # print(data_label_df_list)
    
    data_label_df_list.to_pickle(dir+"/labels.pkl")
    # print(data_label_df_list)


def load_label_datafile(save_dir) -> pd.DataFrame:
        data_label_df_list = pd.read_pickle(save_dir+"/labels.pkl")
        return data_label_df_list

if __name__=="__main__":
    # my_data = CsiData_preprocessor(dir="/data2/alex9395/widar_dataset/CSI", save_dir="/data2/alex9395/widar_preprocessed", n_proc=64, process_chunk_size=4096)

    data_df = load_label_datafile("../data2/widar_preprocessed")
    shape_list = []
    for idx, row in data_df.iterrows():
        data = np.load("".join(["../data2/widar_preprocessed/", row["file name"], ".npz"]))
        shape_list.append(data["csi"].shape[0])
        print(data["csi"].shape[0])
        assert data["csi"].shape[0] == data["time"].shape[0] == data["noise"].shape[0]
    data_df["frame_len"] = shape_list
    print(data_df)
    data_df.to_pickle("../data2/widar_preprocessed/labels2.pkl")

    # live_data = {'date' : [20181109, 20181115, 20181117, 20181211]}
    # live_data = {'date' : [20181208,]}


    # mydata = CsiData(save_dir="../ssddata/widar_preprocess", live_cond=live_data)

    # print(mydata.cond_label)
    # print(len(mydata))

    # for data in mydata:
    #     print(data)

    # fix_labels_file("../data/widar_preprocess")

    # my_data.data_df.to_pickle("../data/result.zip", compression="zip")
    # dir="../data/widar_dataset/CSI"
    # print(glob(dir+"/*"))

