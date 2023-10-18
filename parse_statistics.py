"""
This script extract useful statistics from the .txt created by the experiments
"""
import os

import pandas as pd

file_path = "/Users/gianlucadestefano/Desktop/mnt/risultati/Statistics/Transferability/"


def extract_statistics(file_path, relevant_stats):
    retults = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if 'INFO key:' not in line:
                continue
            stat_key, stat_value = line[line.index(' INFO key: ') + 11:].replace('\n', '').replace('mccs', 'mcc').split(
                ':')
            if stat_key.strip() in relevant_stats:
                retults[relevant_stats[stat_key]] = stat_value

    return retults


def extract_thresholdless_stats(file_path):
    pre_attack_thresholdless_stats_dict = {
        'median-bg': 'median-bg',
        'median-gt': 'median-gt',
        'median-decoy': 'median-decoy',
        'visibility-gt': 'visibility-gt',
        'visibility-decoy': 'visibility-dc',
        'auc_dec_pre': 'auc_dec',
        'auc_gt_pre': 'auc_gt',
    }

    post_attack_thresholdless_stats_dict = {
        'median-bg-attacked': 'median-bg',
        'median-gt-attacked': 'median-gt',
        'median-decoy-attacked': 'median-decoy',
        'visibility-gt-attacked': 'visibility-gt',
        'visibility-decoy-attacked': 'visibility-dc',
        'auc_dec_post': 'auc_dec',
        'auc_gt_post': 'auc_gt',
    }

    # Extract statistics about the native performance of the detection method
    pre_thresholdless_stats = extract_statistics(file_path, relevant_stats=pre_attack_thresholdless_stats_dict)
    assert len(pre_thresholdless_stats) == len(pre_attack_thresholdless_stats_dict)

    # Extract statistics about the performance of the detection method after our attack
    post_thresholdless_stats = extract_statistics(file_path, relevant_stats=post_attack_thresholdless_stats_dict)
    assert len(post_thresholdless_stats) == len(post_attack_thresholdless_stats_dict)

    # Create a dataframes with the statistics
    pre_thresholdless_stats['name'] = 'pre_attack'
    post_thresholdless_stats['name'] = 'post_attack'

    df = pd.DataFrame([pre_thresholdless_stats, post_thresholdless_stats])

    # order columns
    df = df[['name', 'median-bg', 'median-decoy', 'median-gt','visibility-dc', 'visibility-gt', 'auc_dec', 'auc_gt']]

    return df


def extract_thresholded_stats(file_path):
    thresholded_stats = {
        'original_forgery_f1s',
        'target_forgery_f1s',
        'dr_bg_f1',
        'dr_decoy_f1',
        'dr_gt_f1',
        'original_forgery_mcc',
        'target_forgery_mcc',
    }

    results = []

    for i in range(0, 8):

        thresholded_stats_exp = {}
        for key in thresholded_stats:

            k = key
            if i >= 4:
                # Momentary fix to compute stastistics without having to re-run the experiments
                # Remove once the metrics generations pipelines will be corrected
                k = key.replace('_gt_f1', '_gt').replace('_gt_f1s', '_gt').replace('_bg_f1', '_bg').replace('_decoy_f1','_decoy')

            thresholded_stats_exp[f'{k}_{i}'] = key

        experiment_stats = extract_statistics(file_path, relevant_stats=thresholded_stats_exp)
        results.append(experiment_stats)

    df = pd.DataFrame(results)

    # order columns
    df = df[['target_forgery_f1s', 'original_forgery_f1s', 'dr_bg_f1', 'dr_decoy_f1','dr_gt_f1','target_forgery_mcc', 'original_forgery_mcc']]
    return df


def extract_statistics_from_console_file(file_path):
    """
    Read a console.log file containing the statistics of an experiment and extract the relevant statistics
    @param file_path: path of the console.log file
    @return: None
    """
    # Extract threshold statistics
    pre_threshold_stats_df = extract_thresholded_stats(file_path)

    # Extract thresholdless statistics
    pre_thresholdless_stats_df = extract_thresholdless_stats(file_path)

    # Save the statistics in a new excel file
    with pd.ExcelWriter(f"{os.path.dirname(file_path)}/refined_statistics.xlsx") as writer:
        pre_threshold_stats_df.to_excel(writer, sheet_name="Threshold stats", index=False)
        pre_thresholdless_stats_df.to_excel(writer, sheet_name="Thresholdless stats", index=False)


def group_statistics(statistics_root_folder):

    for attack_category in ['SingleAttacks', 'DoubleAttacks']:
        attack_categoy_folder = f'{statistics_root_folder}/{attack_category}'

        for attack_name in os.listdir(attack_categoy_folder):

            if attack_name.startswith('.'):
                continue

            attack_console_file = f'{attack_categoy_folder}/{attack_name}/console.log'

            print(f"Processing {attack_console_file}")
            extract_statistics_from_console_file(attack_console_file)


if __name__ == "__main__":
    group_statistics(file_path)
