import numpy as np
import struct
import pandas as pd

def dbinv(x):
    """dB를 mW로 변환하는 함수."""
    return 10 ** (x / 10)

def db(magnitude, scale_type='pow'):
    """magnitude 값을 mW에서 dBm으로 변환하는 함수."""
    if scale_type == 'pow':
        return 10 * np.log10(magnitude)
    else:
        raise ValueError("scale_type은 'pow'이어야 합니다.")


def get_total_rss(csi_st):
    # RSS의 초기값 설정
    rssi_mag = 0.0
    
    # 각 RSS 값이 0이 아닐 경우 mW 단위로 변환하여 누적
    if csi_st.rssi_a != 0:
        rssi_mag += dbinv(csi_st.rssi_a)
    if csi_st.rssi_b != 0:
        rssi_mag += dbinv(csi_st.rssi_b)
    if csi_st.rssi_c != 0:
        rssi_mag += dbinv(csi_st.rssi_c)

    # 총 RSS 값을 dBm으로 변환하고 AGC 보정 적용
    ret = db(rssi_mag, 'pow') - 44 - csi_st.agc
    return ret


def get_csi_and_noise(csi_st):
    # CSI 가져오기
    csi = csi_st.csi

    # CSI와 RSSI(mW) 사이의 스케일 팩터 계산
    csi_sq = csi * np.conj(csi)
    csi_pwr = np.sum(csi_sq)
    rssi_pwr = dbinv(get_total_rss(csi_st))  # get_total_rss 함수가 필요합니다.
    scale = rssi_pwr / (csi_pwr / 30)

    # thermal noise 처리
    if csi_st.noise == -127:
        noise_db = -92
    else:
        noise_db = csi_st.noise
    thermal_noise_pwr = dbinv(noise_db)

    # 양자화 오차 전력
    quant_error_pwr = scale * (csi_st.Nrx * csi_st.Ntx)

    # 전체 잡음 및 오차 전력
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    # 스케일 적용된 CSI 반환
    csi = csi * np.sqrt(scale)
    noise_std = np.sqrt(total_noise_pwr)
    if csi_st.Ntx == 2:
        csi *= np.sqrt(2)
        noise_std *= np.sqrt(2)
    elif csi_st.Ntx == 3:
        csi *= np.sqrt(dbinv(4.5))  # Intel에서 4.5 dB를 적용함
        noise_std *= np.sqrt(dbinv(4.5))

    return csi, noise_std



def get_scaled_csi(csi_st):
    # CSI 가져오기
    csi = csi_st.csi

    # CSI와 RSSI(mW) 사이의 스케일 팩터 계산
    csi_sq = csi * np.conj(csi)
    csi_pwr = np.sum(csi_sq)
    rssi_pwr = dbinv(get_total_rss(csi_st))  # get_total_rss 함수가 필요합니다.
    scale = rssi_pwr / (csi_pwr / 30)

    # thermal noise 처리
    if csi_st.noise == -127:
        noise_db = -92
    else:
        noise_db = csi_st.noise
    thermal_noise_pwr = dbinv(noise_db)

    # 양자화 오차 전력
    quant_error_pwr = scale * (csi_st.Nrx * csi_st.Ntx)

    # 전체 잡음 및 오차 전력
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    # 스케일 적용된 CSI 반환
    ret = csi * np.sqrt(scale / total_noise_pwr)
    if csi_st.Ntx == 2:
        ret *= np.sqrt(2)
    elif csi_st.Ntx == 3:
        ret *= np.sqrt(dbinv(4.5))  # Intel에서 4.5 dB를 적용함

    return ret

class CSIEntry:
    def __init__(self):
        self.timestamp_low = None
        self.bfee_count = None
        self.Nrx = None
        self.Ntx = None
        self.rssi_a = None
        self.rssi_b = None
        self.rssi_c = None
        self.noise = None
        self.agc = None
        self.perm = None
        self.rate = None
        self.csi = None

def read_bfee(bytes_data):
    """
    C에서 구현된 read_bfee 함수를 Python으로 변환한 것.
    bytes_data: uint8 바이트 배열 (20바이트 이상의 길이 필요)
    """
    csi_entry = CSIEntry()
    
    # 1. timestamp_low (4 bytes)
    csi_entry.timestamp_low = struct.unpack('I', bytes_data[0:4])[0]

    # 2. bfee_count (2 bytes)
    csi_entry.bfee_count = struct.unpack('H', bytes_data[4:6])[0]

    # 3. Nrx, Ntx, RSSI A, B, C (1 byte each)
    csi_entry.Nrx = bytes_data[8]
    csi_entry.Ntx = bytes_data[9]
    csi_entry.rssi_a = bytes_data[10]
    csi_entry.rssi_b = bytes_data[11]
    csi_entry.rssi_c = bytes_data[12]

    # 4. Noise (signed byte)
    csi_entry.noise = struct.unpack('b', bytes_data[13:14])[0]

    # 5. AGC (1 byte)
    csi_entry.agc = bytes_data[14]

    # 6. Antenna selection (1 byte)
    antenna_sel = bytes_data[15]

    # 7. len (2 bytes) & fake_rate_n_flags (2 bytes)
    length = struct.unpack('H', bytes_data[16:18])[0]
    csi_entry.rate = struct.unpack('H', bytes_data[18:20])[0]

    # 8. CSI 배열 크기 계산
    calc_len = (30 * (csi_entry.Nrx * csi_entry.Ntx * 8 * 2 + 3) + 7) // 8

    # 길이 확인
    if length != calc_len:
        raise ValueError("Wrong beamforming matrix size")

    # 9. CSI 추출
    payload = bytes_data[20:]
    csi = np.zeros((csi_entry.Ntx, csi_entry.Nrx, 30), dtype=complex)

    index = 0
    for i in range(30):
        index += 3  # 스킵할 부분
        remainder = index % 8
        for j in range(csi_entry.Nrx * csi_entry.Ntx):
            # 실수부 추출
            tmp_real = (payload[index // 8] >> remainder) | (payload[index // 8 + 1] << (8 - remainder))
            # real_part = float(np.int8(tmp_real))
            real_part = float(np.array(tmp_real).astype(np.int8))

            # 허수부 추출
            tmp_imag = (payload[index // 8 + 1] >> remainder) | (payload[index // 8 + 2] << (8 - remainder))
            # imag_part = float(np.int8(tmp_imag))
            imag_part = float(np.array(tmp_imag).astype(np.int8))

            csi[j // csi_entry.Nrx, j % csi_entry.Nrx, i] = real_part + 1j * imag_part
            index += 16

    csi_entry.csi = csi

    # 10. Permutation 배열 생성
    csi_entry.perm = [
        (antenna_sel & 0x3),
        ((antenna_sel >> 2) & 0x3),
        ((antenna_sel >> 4) & 0x3)
    ]

    return csi_entry



def read_bf_file(filename):
    try:
        # 파일 열기
        with open(filename, 'rb') as f:
            # 파일 끝으로 이동해 파일 크기 확인
            f.seek(0, 2)
            len_file = f.tell()
            f.seek(0, 0)

            # 초기 변수 설정
            ret = []
            cur = 0
            count = 0
            broken_perm = False
            triangle = [0, 1, 3]  # 각 안테나에 따른 합

            # 파일 전체 데이터 읽기
            while cur < (len_file - 3):
                # 사이즈 필드와 코드 읽기
                field_len, code = struct.unpack('>HB', f.read(3))
                cur += 3

                # 코드가 처리할 수 없는 경우 필드 스킵
                if code == 187:  # 빔포밍 데이터 코드
                    bytes_data = f.read(field_len - 1)
                    cur += field_len - 1

                    if len(bytes_data) != field_len - 1:
                        return ret  # 데이터가 부족할 경우 함수 종료
                else:
                    f.seek(field_len - 1, 1)
                    cur += field_len - 1
                    continue

                # 코드가 187일 경우 데이터 처리
                if code == 187:
                    count += 1
                    # read_bfee 함수 필요
                    csi_entry = read_bfee(bytes_data)
                    ret.append(csi_entry)

                    perm = csi_entry.perm
                    Nrx = csi_entry.Nrx
                    if Nrx == 1:  # 안테나가 하나일 경우 퍼미팅 불필요
                        continue
                    if sum(perm) != triangle[Nrx - 1]:  # 기본 값이 아닌 경우
                        if not broken_perm:
                            broken_perm = True
                            print(f"WARN ONCE: Found CSI ({filename}) with Nrx={Nrx} and invalid perm={perm}")
                    else:
                        csi_entry.csi[:, perm[:Nrx], :] = csi_entry.csi[:, :Nrx, :]

    except IOError:
        print(f"Couldn't open file {filename}")
        return None

    return ret[:count]  # 저장된 항목들만 반환


def csi_get_all(filename):
    csi_trace = read_bf_file(filename)  # Python 버전의 read_bf_file 함수 필요
    timestamp = np.zeros(len(csi_trace))
    cfr_array = np.zeros((len(csi_trace), 3, 30), dtype=np.complex64)
    noise_array = np.zeros(len(csi_trace), dtype=np.float32)

    for k in range(len(csi_trace)):
        csi_entry = csi_trace[k]  # k번째 패킷

        try:
            csi_all, noise_all = get_csi_and_noise(csi_entry)
            csi_all = np.squeeze(csi_all)   # (3, 30)
            noise_all = np.squeeze(noise_all) # 단일 scalar 값
            # csi_all = np.squeeze(get_scaled_csi(csi_entry)).T  # Python 버전의 get_scaled_csi 함수 필요
        except Exception as err:
            print(err)
            continue

        # csi = np.hstack((csi_all[:, 0], csi_all[:, 1], csi_all[:, 2]))  # 각 안테나 쌍의 CSI 선택
        timestamp[k] = csi_entry.timestamp_low
        cfr_array[k] = csi_all
        noise_array[k] = noise_all.real

    return cfr_array, noise_array, timestamp


def csi_file_process(dat_file_dir):
    data_backbone = {}

    date = int(dat_file_dir.split("/")[-3])

    filename = dat_file_dir.split("/")[-1]
    filename = filename.replace("user", "").replace(".dat", "").replace("r", "")
    split_filename = filename.split('-')

    user = int(split_filename[0])
    gesture_id = int(split_filename[1])
    torso_loc_id = int(split_filename[2])
    face_loc_id = int(split_filename[3])
    repetition_num = int(split_filename[4])
    rx_id = int(split_filename[5])
    csi_data = csi_get_all(dat_file_dir)

    data_backbone["date"] = date
    data_backbone["user"] = user
    # data_backbone["gesture"] = gesture_name_parser(date, user, gesture_id)
    data_backbone["gesture"] = gesture_id
    data_backbone["torso"] = torso_loc_id
    data_backbone["face"] = face_loc_id
    data_backbone["repetition"] = repetition_num
    data_backbone["rx"] = rx_id

    return data_backbone, csi_data

