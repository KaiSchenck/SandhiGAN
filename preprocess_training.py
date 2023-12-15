#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-07-23 15:25

import os
import time
import argparse
import gc
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Custom Classes
import preprocess


def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)
        
def get_validation_files(wavs, log_f0s_mean_A, log_f0s_std_A, log_f0s_mean_B, log_f0s_std_B, coded_sps_A_std):
    num_mcep = 36
    sampling_rate = 44000
    frame_period = 5.0
    n_frames = 128
    
    coded_sp_norms = []
    f0s = []
    
    for wav in wavs:
        wav = preprocess.wav_padding(wav=wav,
                                     sr=sampling_rate,
                                     frame_period=frame_period,
                                     multiple=4)
        f0, timeaxis, sp, ap = preprocess.world_decompose(
            wav=wav, fs=sampling_rate, frame_period=frame_period)
        f0_converted = preprocess.pitch_conversion(f0=f0,
                                                   mean_log_src=log_f0s_mean_A,
                                                   std_log_src=log_f0s_std_A,
                                                   mean_log_target=log_f0s_mean_B,
                                                   std_log_target=log_f0s_std_B)
        f0s.append(f0_converted)
        coded_sp = preprocess.world_encode_spectral_envelop(
            sp=sp, fs=sampling_rate, dim=num_mcep)
        coded_sp_transposed = coded_sp.T
        coded_sp_norm = (coded_sp_transposed -
                         self.coded_sps_A_mean) / self.coded_sps_A_std
        coded_sp_norm = np.array([coded_sp_norm])

        if torch.cuda.is_available():
            coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
        else:
            coded_sp_norm = torch.from_numpy(coded_sp_norm).float()
    
        np.savez(os.path.join(cache_folder, 'mcep_normalization.npz'),
                 coded_sp_norm=coded_sp_norm,
                 std_A=coded_sps_A_std,
                 mean_B=coded_sps_B_mean,
                 std_B=coded_sps_B_std)
                 

def preprocess_for_training(train_A_dir, train_B_dir, cache_folder):
    num_mcep = 36
    sampling_rate = 44000
    frame_period = 5.0
    n_frames = 128

    print("Starting to preprocess data.......")
    start_time = time.time()

    wavs_A = preprocess.load_wavs(train_A_dir)

    print("A dataset loaded")
    
    #f0s_B = [preprocess.get_f0(wav.astype(np.float64), sampling_rate, frame_period)[0] for wav in tqdm(wavs_B)]

    #save_pickle(variable=f0s_B,
                #fileName=os.path.join(cache_folder, "f0s_B.pickle"))
                
    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = preprocess.world_encode_data(
        wave=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
        
    save_pickle(variable=f0s_A,
            fileName=os.path.join(cache_folder, "f0s_A.pickle"))
        
    log_f0s_mean_A, log_f0s_std_A = preprocess.logf0_statistics(f0s=f0s_A)
    
    coded_sps_A_transposed = preprocess.transpose_in_list(lst=coded_sps_A)
    
    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = preprocess.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_A_transposed)
        
    save_pickle(variable=coded_sps_A_norm,
                fileName=os.path.join(cache_folder, "coded_sps_A_norm.pickle"))
             
    del coded_sps_A_norm
    
    gc.collect()
    
    wavs_B = preprocess.load_wavs(train_B_dir)
    print("B dataset loaded")
    
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = preprocess.world_encode_data(
        wave=wavs_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
        
    save_pickle(variable=f0s_B,
                fileName=os.path.join(cache_folder, "f0s_B.pickle"))


    log_f0s_mean_B, log_f0s_std_B = preprocess.logf0_statistics(f0s=f0s_B)
    
    coded_sps_B_transposed = preprocess.transpose_in_list(lst=coded_sps_B)

    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = preprocess.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_B_transposed)

    np.savez(os.path.join(cache_folder, 'logf0s_normalization.npz'),
             mean_A=log_f0s_mean_A,
             std_A=log_f0s_std_A,
             mean_B=log_f0s_mean_B,
             std_B=log_f0s_std_B)
    
    np.savez(os.path.join(cache_folder, 'mcep_normalization.npz'),
             mean_A=coded_sps_A_mean,
             std_A=coded_sps_A_std,
             mean_B=coded_sps_B_mean,
             std_B=coded_sps_B_std)
    
    save_pickle(variable=coded_sps_B_norm,
                fileName=os.path.join(cache_folder, "coded_sps_B_norm.pickle"))
                
    print("Log Pitch A")
    print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_A, log_f0s_std_A))
    print("Log Pitch B")
    print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_B, log_f0s_std_B))


    end_time = time.time()
    print("Preprocessing finsihed!! see your directory ../cache for cached preprocessed data")

    print("Time taken for preprocessing {:.4f} seconds".format(
        end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare data for training Cycle GAN using PyTorch')
    train_A_dir_default = "./a_tiny.npy"
    train_B_dir_default = "./b_tiny.npy"
    cache_folder_default = './cache/'
    

    parser.add_argument('--train_A_dir', type=str,
                        help="Directory for source voice sample", default=train_A_dir_default)
    parser.add_argument('--train_B_dir', type=str,
                        help="Directory for target voice sample", default=train_B_dir_default)
    parser.add_argument('--cache_folder', type=str,
                        help="Store preprocessed data in cache folders", default=cache_folder_default)
    argv = parser.parse_args()

    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    cache_folder = argv.cache_folder

    preprocess_for_training(train_A_dir, train_B_dir, cache_folder)
