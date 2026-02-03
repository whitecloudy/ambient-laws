import numpy as np
from training.dataset import renewRfProcessedDataset
import dnnlib
from train import parse_int_list
import click, os
from glob import glob
from tqdm import tqdm


@click.command()

# Main options.
# @click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str,  default='data/RENEW/splited/ArgosCSI-96x8-2016-11-04-05-37-37_2.4GHz_track_left_to_right_NLOS')
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str,  default='data/RENEW/splited_split_pliot_seq/ArgosCSI-96x8-2016-11-04-05-37-37_2.4GHz_track_left_to_right_NLOS')
# @click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str,  default='data/RENEW/splited/ArgosCSI-96x8-2016-11-04-04-07-44_2.4GHz_static_LOS')
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)

# RF dataset related
@click.option('--view_as_complex', help='Whether to view the data as complex numbers.', type=bool, default=False, show_default=True)
@click.option('--complex_merge_axis', help='Axis to merge real and imaginary parts when view_as_complex is False. Set to None to not merge.', type=int, default=0, show_default=True)
@click.option('--transpose', help='Transpose the data axes according to the given order. Provide a list of two integers representing the new order of the first two axes (frame_resolution and ant_resolution). Set to None to not transpose.', type=str, default="1,0", show_default=True)
@click.option('--frame_res', help='Frame resolution of the RF data.', type=int, default=14, show_default=True)
@click.option('--ant_res', help='Antenna resolution of the RF data.', type=int, default=8, show_default=True)
@click.option('--data_norm', help='Data normalization value for the RF data.', type=float, default=1.0, show_default=True)

@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)

# Scaling laws related
@click.option("--corruption_probability", help="Controls what percentage of images should be corrupted.", type=float, default=0.0)
@click.option("--sigma", help="How much noise to add to the corrupted images.", type=float, default=0.0)
def main(**opts):
    opts = dnnlib.EasyDict(opts)
    # Initialize config dict.
    c = dnnlib.EasyDict()
    # dataset_kwargs for RENEW dataset
    c.dataset_kwargs = dnnlib.EasyDict(path=opts.data, use_labels=opts.cond, cache=opts.cache, sigma=opts.sigma, 
                                       corruption_probability_per_image=opts.corruption_probability, corruption_probability_per_pixel=1.0, 
                                       only_positive=False, view_as_complex=True,
                                       resolution=(opts.frame_res, opts.ant_res), transpose=parse_int_list(opts.transpose) if opts.transpose is not None else None,
                                       normalize_value=opts.data_norm, noise_mean_flag=True)

    dataset = renewRfProcessedDataset(**c.dataset_kwargs)

    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    image_sum = np.complex128(0)
    image_sq_sum = np.float64(0)
    image_count = 0

    sigma_sum = np.float64(0)
    sigma_sq_sum = np.float64(0)
    sigma_count = 0

    print("Calculating mean and std...")
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        img = data['image']
        sigma = data['sigma']

        image_sum += np.sum(np.abs(img))
        image_sq_sum += np.sum(np.abs(img)**2)
        image_count += img.size

        sigma_sum += np.sum(sigma)
        sigma_sq_sum += np.sum(sigma**2)
        sigma_count += sigma.size

    image_mean = image_sum / image_count
    image_std = np.sqrt(image_sq_sum / image_count)

    sigma_mean = sigma_sum / sigma_count
    sigma_std = np.sqrt(sigma_sq_sum / sigma_count - sigma_mean**2)

    print(f"Image Mean: {image_mean}")
    print(f"Image Std: {image_std}")
    print(f"Sigma Mean: {sigma_mean}")
    print(f"Sigma Std: {sigma_std}")

if __name__ == "__main__": 
    main()