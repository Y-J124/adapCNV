#!/bin/bash
#SBATCH -N 1                     # 使用的节点数
#SBATCH -n 28                     # 总共的运行任务数
#SBATCH --ntasks-per-node=28      # 每个节点的任务数
#SBATCH --partition=6248      # 指定分区
#SBATCH --output=%j.out          # 标准输出文件
#SBATCH --error=%j.err           # 标准错误文件

python AdapCNV_pro.py ICGC_21.bam GRCh38_hla_decoy_ebv.fa.gz out 1000 0.1