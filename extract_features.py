# -*- coding: UTF-8 -*-
import os
import pysam
import numpy as np
import pandas as pd

def calculate_gc_content(sequence):
    g = sequence.upper().count('G')
    c = sequence.upper().count('C')
    return (g + c) / len(sequence) if len(sequence) > 0 else 0

def extract_features(bam_file, window_size=5000, threshold=0.2, normal_bam='normal.bam', targets_bed='targets.bed',
                     reference='reference.fasta', output_dir='output'):
    bam = pysam.AlignmentFile(bam_file, "rb")
    coverage = []
    #gc_content = []

    for pileupcolumn in bam.pileup():
        coverage.append(pileupcolumn.n)
        #seq = ''
        #for pileupread in bam.fetch(reference=pileupcolumn.reference_name, start=pileupcolumn.pos, end=pileupcolumn.pos + 1):
            #seq += pileupread.query_sequence if pileupread.query_sequence else ''
        #gc_content.append(calculate_gc_content(seq))

    bam.close()

    coverage = np.array(coverage)
    mean_coverage = np.mean(coverage)
    stddev_coverage = np.std(coverage)
    peak_coverage = np.max(coverage)
    repeat_fraction = np.sum(coverage > mean_coverage * 2) / len(coverage)
    #mean_gc_content = np.mean(gc_content)

    features = {
        #'bam_file': bam_file,
        'mean_coverage': mean_coverage,
        'stddev_coverage': stddev_coverage,
        'peak_coverage': peak_coverage,
        'repeat_fraction': repeat_fraction,
        #'gc_content': mean_gc_content,
    }

    return features

# ����������һ������100��BAM�ļ�·�����б�
bam_files = [f'data/50x/bam/s_{i}.bam' for i in range(0, 30)]

# ��鵱ǰ����Ŀ¼
print(f"Current working directory: {os.getcwd()}")

# ��ȡ����BAM�ļ�������
features_list = []
for bam in bam_files:
    features = extract_features(bam)
    if features is not None:
        features_list.append(features)

# ����Ƿ�ɹ���ȡ����
if len(features_list) == 0:
    print("No features were extracted. Please check the BAM file paths and contents.")
else:
    # �������б�ת��ΪPandas DataFrame
    features_df = pd.DataFrame(features_list)

    # ȷ�����·��
    output_path = os.path.join(os.getcwd(), 'data/50x/bam_features.csv')
    print(f"Output CSV file path: {output_path}")

    # ����������д��CSV�ļ�
    features_df.to_csv(output_path, index=False)

    print(f"Features of {len(bam_files)} BAM files have been written to {output_path}")
