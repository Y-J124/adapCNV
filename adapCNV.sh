#!/bin/bash
#SBATCH -N 1                     # ʹ�õĽڵ���
#SBATCH -n 28                     # �ܹ�������������
#SBATCH --ntasks-per-node=28      # ÿ���ڵ��������
#SBATCH --partition=6248      # ָ������
#SBATCH --output=%j.out          # ��׼����ļ�
#SBATCH --error=%j.err           # ��׼�����ļ�

python AdapCNV_pro.py ICGC_21.bam GRCh38_hla_decoy_ebv.fa.gz out 1000 0.1