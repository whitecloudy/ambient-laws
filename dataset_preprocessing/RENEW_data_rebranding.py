import numpy as np
import torch

def pilot_name_exchange(fname):
    # get other data from pilot pair
    if fname.find('pilot0') != -1:
        change_fname = fname.replace('pilot0', 'pilot1')
    elif fname.find('pilot1') != -1:
        change_fname = fname.replace('pilot1', 'pilot0')
    else:
        import warnings
        warnings.warn(f"Filename {fname} does not contain 'pilot0' or 'pilot1'. Returning original item.")
        return fname
    return change_fname


def rebrand_renew_data(D0, D1, origianl_sigma, target_sigma_gain=1.0, target_sigma_additive=0.0):
    """
    RENEW 데이터셋의 각 pilot 0, 1을 받아서 이를 이용해 원하는 Noise 수준의 데이터셋으로 rebranding
    
    D0 = D + N0
    D1 = D + N1
    N0, N1 ~ N(0, origianl_sigma^2)
    D0 - D1 = N0 - N1 
    D_new = D + N_new = \alpha * (D0 - D1) + 0.5 * (D0 + D1)
    Var(N_new) = (target_sigma_gain * origianl_sigma + target_sigma_additive)^2 = (2 * \alpha^2 + 0.5) * origianl_sigma^2
    => \alpha = sqrt(((target_sigma_gain * origianl_sigma + target_sigma_additive)^2 / origianl_sigma^2 - 0.5) / 2)

    :param D0: pilot 0의 데이터 (numpy array)
    :param D1: pilot 1의 데이터 (numpy array)
    :param origianl_sigma: 원래 데이터셋의 Noise std dev 수준
    :param target_sigma_gain: 원하는 Noise std dev 수준 (target_sigma = origianl_sigma * target_sigma_gain)
    :param target_sigma_additive: 원하는 Noise std dev 수준 (target_sigma = origianl_sigma + target_sigma_additive)
    :return: rebrand된 데이터셋
    """
    alpha = np.sqrt(((target_sigma_gain * origianl_sigma + target_sigma_additive)**2 / origianl_sigma**2 - 0.5) / 2)
    alpha = np.expand_dims(alpha, axis=-1)
    D_new = alpha * (D0 - D1) + 0.5 * (D0 + D1)
    return D_new, target_sigma_gain * origianl_sigma + target_sigma_additive, alpha

# def rebrand_renew_data_sigma_additive(D0, D1, origianl_sigma, target_sigma_additive):
#     """
#     RENEW 데이터셋의 각 pilot 0, 1을 받아서 이를 이용해 원하는 Noise 수준의 데이터셋으로 rebranding
    
#     D0 = D + N0
#     D1 = D + N1
#     N0, N1 ~ N(0, origianl_sigma^2)
#     D0 - D1 = N0 - N1 
#     D_new = D + N_new = \alpha * (D0 - D1) + 0.5 * (D0 + D1)
#     Var(N_new) = (target_sigma_additive + origianl_sigma)^2 = (2 * \alpha^2 + 0.5) * origianl_sigma^2
#     => \alpha = sqrt(((target_sigma_additive + origianl_sigma)^2 / origianl_sigma^2 - 0.5) / 2)
    
#     :param D0: pilot 0의 데이터 (numpy array)
#     :param D1: pilot 1의 데이터 (numpy array)
#     :param origianl_sigma: 원래 데이터셋의 Noise std dev 수준
#     :param target_sigma_additive: 원하는 Noise std dev 수준 (target_sigma = origianl_sigma + target_sigma_additive)

#     :return: rebrand된 데이터셋
#     """
#     alpha = np.sqrt(((target_sigma_additive + origianl_sigma)**2 / origianl_sigma**2 - 0.5) / 2)
#     D_new = alpha * (D0 - D1) + 0.5 * (D0 + D1)
#     return D_new, target_sigma_additive + origianl_sigma


def load_pilot_datas(file_path):
    """
    주어진 파일 경로에서 pilot 데이터를 로드하는 함수 (예시 구현)
    
    :param file_path: 데이터 파일 경로
    :return: pilot 0과 pilot 1 데이터 (numpy array)
    """
    import os
    from glob import glob
    import fnmatch

    _fname = []

    # 파일 경로가 리스트인지, 디렉토리인지 확인 후 파일 경로들 로드
    if isinstance(file_path, list):
        for path in file_path:
            _fname += glob(os.path.join(path, '*.npz'))
    elif os.path.isdir(file_path):
        _fname = glob(os.path.join(file_path, '*.npz'), recursive=True)
    else:
        raise IOError('Path must point to a directory or list of file paths')
    
    # pilot 0 파일들만 필터링
    _fname = [fname for fname in _fname if fnmatch.fnmatch(fname, '*pilot0*.npz')]

    return _fname

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RENEW 데이터 rebranding 예시')
    parser.add_argument('--data_path', type=str, required=True, help='RENEW 데이터셋 파일 경로 또는 디렉토리')
    parser.add_argument('--additive_sigma', type=float, default=0.0, required=False, help='추가할 Noise std dev 수준, Gain 이후 적용됨')
    parser.add_argument('--sigma_gain', type=float, default=1.0, required=False, help='곱할 Noise std dev 수준, Additive 이전 적용됨' )
    parser.add_argument('--output_path', type=str, required=True, help='Rebrand된 데이터 저장 경로')
    args = parser.parse_args()

    output_path = args.output_path
    import os
    import copy

    pilot0_file_dir_list = load_pilot_datas(args.data_path)
    import tqdm

    os.path.exists(output_path) or os.makedirs(output_path)

    for pilot0_file_dir in tqdm.tqdm(pilot0_file_dir_list):
        pilot1_file_dir = pilot_name_exchange(pilot0_file_dir)

        data0 = np.load(pilot0_file_dir)
        data1 = np.load(pilot1_file_dir)

        D0 = data0['csi']
        D1 = data1['csi']
        origianl_sigma = data0['noise']  # Assuming sigma is stored in the file

        # print(D0.shape, D1.shape, origianl_sigma.shape)
        D_new, new_sigma, used_alpha = rebrand_renew_data(D0, D1, origianl_sigma, target_sigma_gain=args.sigma_gain, target_sigma_additive=args.additive_sigma)
    
        new_data = dict(data0)
        new_data['csi'] = D_new
        new_data['noise'] = new_sigma

        output_file = pilot0_file_dir.replace('pilot0', f'sigma-gain{args.sigma_gain}-add{args.additive_sigma}')
        output_file_dir = os.path.join(output_path, os.path.basename(output_file))

        # print(f"Saving rebranded data to {output_file_dir}")

        # DEBUG
        # sampling_std = np.expand_dims(np.std(new_data['csi']-data0['csi'], axis=-1), axis=-1)
        # expected_std = 2**0.5 * (used_alpha - 0.5) * np.expand_dims(data0['noise'], axis=-1)

        # print(np.mean(sampling_std/expected_std))

        # print()
        # print(new_data['noise'])
        # print(data0['noise'])
        # break

        np.savez(output_file_dir, **new_data)
    print("Rebranding completed.")