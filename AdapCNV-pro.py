# -*- coding: ISO-8859-1 -*-
# AdapCNV-pro
# Author: YJ Yuan

import numpy as np
import pysam
import sys
import pandas as pd
from numba import njit
import subprocess
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from scipy.optimize import minimize
import joblib
import os
import gzip
import re  # ????????
from pyod.models.iforest import IForest

def system_model(binsize, threshold, step_size, rd_diff):
    regularization = 0.1 * (binsize - 500)**2 + 0.1 * (threshold - 50)**2 + 0.1 * (step_size - 0.1)**2
    return np.sum(np.abs(rd_diff)) * binsize - threshold + regularization


def online_optimizer(rd_diff, binsize, threshold, step_size):
    binsize = np.clip(binsize, 500, 10000)
    threshold = np.clip(threshold, 1, 2)
    step_size = np.clip(step_size, 0.01, 1)

    def objective(params):
        binsize, threshold, step_size = params
        return system_model(binsize, threshold, step_size, rd_diff)

    bounds = [(500, 10000), (1, 2), (0.01, 1)]
    result = minimize(objective, [binsize, threshold, step_size], bounds=bounds, method='L-BFGS-B', options={'maxiter': 100})

    print(f"???: binsize={result.x[0]}, threshold={result.x[1]}, step_size={result.x[2]}")
    return result.x


class AMPCController:
    def __init__(self, binsize, threshold, step_size):
        self.binsize = binsize
        self.threshold = threshold
        self.step_size = step_size

    def update(self, rd_diff):
        print(f"¸before: binsize={self.binsize}, threshold={self.threshold}, step_size={self.step_size}")
        self.binsize, self.threshold, self.step_size = online_optimizer(rd_diff, self.binsize, self.threshold, self.step_size)
        print(f"¸after: binsize={self.binsize}, threshold={self.threshold}, step_size={self.step_size}")

def get_chrlist(filename):
    """
    ?? BAM ????????????????????
    ???['chr1', 'chr2', 'chr3', ?, 'chrX', 'chrY']
    """
    samfile = pysam.AlignmentFile(filename, "rb", ignore_truncation=True)
    return list(samfile.references)


def get_RC(filename, ReadCount, Mapq, target_chr, chrLen):
    """
    ??????? target_chr ???????????????????? read ???
    ???
      filename   -- BAM ????
      ReadCount  -- ????????????????? numpy ?????????
      Mapq       -- ????? numpy ??????? mapq?
      target_chr -- ?????????? "chr19"
      chrLen     -- ?????????????
    ???
      ???? ReadCount??????Bpoint?? Mapq ??
    """
    qname, flag, q_mapq, cigar = [], [], [], []
    pos, direction, isize, qlen = [], [], [], []
    samfile = pysam.AlignmentFile(filename, "rb", ignore_truncation=True)

    # ????????????????? BAM ??????????? target_chr ???
    for read in samfile.fetch(target_chr):
        posList = read.positions
        # ???posList ???????? 0 ~ (chrLen-1) ???
        ReadCount[posList] += 1
        Mapq[posList] += read.mapq
        if read.cigarstring is not None:
            # ?? read ? cigar ????? M????????? mapq >= 10????????
            cigarstring = str(read.query_length) + 'M'
            if read.cigarstring != cigarstring and read.mapq >= 10:
                qname.append(read.qname)
                flag.append(read.flag)
                q_mapq.append(read.mapq)
                cigar.append(read.cigarstring)
                pos.append(read.pos)
                direction.append(read.is_read1)
                isize.append(read.isize)
                qlen.append(read.qlen)
    # ?? DataFrame?????????
    SR = pd.DataFrame(list(zip(qname, flag, direction, q_mapq, cigar, pos, qlen)),
                      columns=['name', 'flag', 'dir', 'mapq', 'cigar', 'pos', 'len'])
    # ?????? get_breakpoint2????? chrLen
    Bpoint = get_breakpoint2(SR, chrLen)
    return ReadCount, Bpoint, Mapq


def get_breakpoint(read_cigar, read_pos, read_len):
    """
    ?? read ? cigar ????????????????
    """
    breakpoint = []
    for i in range(len(read_cigar)):
        if 'S' in read_cigar[i] and 'M' in read_cigar[i]:
            S_index = read_cigar[i].index('S')
            M_index = read_cigar[i].index('M')
            if S_index < M_index:
                breakpoint.append(read_pos[i])
            elif S_index > M_index:
                breakpoint.append(read_pos[i] + read_len[i])
        elif 'H' in read_cigar[i] and 'M' in read_cigar[i]:
            H_index = read_cigar[i].index('H')
            M_index = read_cigar[i].index('M')
            if H_index < M_index:
                breakpoint.append(read_pos[i])
            elif H_index > M_index:
                breakpoint.append(read_pos[i] + read_len[i])
    return breakpoint


def get_breakpoint2(SR, chrLen):
    """
    ?? SR??? read ??? DataFrame????????
    ?? chrLen ???????????
    ?????????????? 0 ? chrLen-1?
    """
    breakpoint1 = []
    breakpoint2 = [0, chrLen - 1]
    for name, group in SR.groupby('name'):
        num_g = group['name'].count()
        if num_g == 1:
            read_cigar = np.array(group['cigar'])
            read_pos = np.array(group['pos'])
            read_len = np.array(group['len'])
            breakpoint1 += get_breakpoint(read_cigar, read_pos, read_len)
        elif num_g >= 2:
            index1 = np.array(group['dir']) == True
            read1_count = np.sum(index1)
            index2 = np.array(group['dir']) == False
            read2_count = np.sum(index2)

            if read1_count > 1:
                read1_cigar = np.array(group['cigar'][index1])
                read1_pos = np.array(group['pos'][index1])
                read1_len = np.array(group['len'][index1])
                breakpoint2 += get_breakpoint(read1_cigar, read1_pos, read1_len)

            if read2_count > 1:
                read2_cigar = np.array(group['cigar'][index2])
                read2_pos = np.array(group['pos'][index2])
                read2_len = np.array(group['len'][index2])
                breakpoint2 += get_breakpoint(read2_cigar, read2_pos, read2_len)

    breakpoint2 = sorted(list(set(breakpoint2)))
    return breakpoint2


def read_ref_file(filename):
    """
    ???? fasta ????? .fasta ? .fasta.gz?????????
      refDict[ref_name] = ?????
    """
    refDict = {}
    if os.path.exists(filename):
        print("Reading reference file: " + str(filename))
        if filename.endswith('.gz'):
            with gzip.open(filename, 'rb') as f:
                content = f.read().decode('utf-8')
        else:
            with open(filename, 'r') as f:
                content = f.read()
        lines = content.splitlines()
        current_ref = None
        seq_list = []
        for line in lines:
            if line.startswith('>'):
                if current_ref is not None:
                    refDict[current_ref] = "".join(seq_list)
                current_ref = line[1:].split()[0]
                seq_list = []
            else:
                seq_list.append(line.strip())
        if current_ref is not None:
            refDict[current_ref] = "".join(seq_list)
    else:
        print("Reference file not found: " + str(filename))
    return refDict


def ReadDepth(ReadCount, ref, pos):

    start = pos[:len(pos) - 1]
    end = pos[1:]
    for i in range(len(pos) - 1):
        if end[i] - start[i] < 500:
            pos.remove(end[i])
    pos = np.array(pos)
    start = pos[:len(pos) - 1]
    end = pos[1:]
    length = end - start
    with open('pos.txt', 'w') as f:
        for i in range(len(start)):
            linestrlist = ['1', '1', str(start[i]), str(end[i]-1), str(length[i])]
            f.write('\t'.join(linestrlist) + '\n')
    bin_start,bin_end,bin_len,s = re_segfile('pos.txt', 'bin.txt', binSize)
    binNum = len(bin_start)
    bin_RD = np.full(binNum, 0.0)
    bin_GC = np.full(binNum, 0)
    bin_gc = np.full(binNum,0.0)
    for i in range(binNum):
        bin_RD[i] = np.mean(ReadCount[bin_start[i]:bin_end[i]])
        cur_ref = ref[bin_start[i]:bin_end[i]]
        N_count = cur_ref.count('N') + cur_ref.count('n')
        if N_count == 0:
            gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
        else:
            bin_RD[i] = -10000
            gc_count = 0
        bin_GC[i] = int(round(gc_count / bin_len[i], 3) * 1000)
        bin_gc[i] = round(gc_count / bin_len[i],2)
    bin_end -= 1

    index = bin_RD > 0
    bin_RD = bin_RD[index]
    bin_GC = bin_GC[index]
    bin_gc = bin_gc[index]
    bin_len = bin_len[index]
    bin_start = bin_start[index]
    bin_end = bin_end[index]
    bin_RD = gc_correct(bin_RD, bin_GC)
    return bin_start,bin_end,bin_len, bin_RD, bin_gc


def gc_correct(RD, GC):

    bincount = np.bincount(GC)
    global_rd_ave = np.mean(RD)
    for i in range(len(RD)):
        if bincount[GC[i]] < 2:
            continue
        mean_RD = np.mean(RD[GC == GC[i]])
        RD[i] = global_rd_ave * RD[i] / mean_RD
    return RD


def prox_tv1d(step_size: float, w: np.ndarray) -> np.ndarray:

    if w.dtype not in (np.float32, np.float64):
        raise ValueError('argument w must be array of floats')
    w = w.copy()
    output = np.empty_like(w)
    _prox_tv1d(step_size, w, output)
    return output

@njit
def _prox_tv1d(step_size, input, output):
    """low level function call, no checks are performed"""
    width = input.size + 1
    index_low = np.zeros(width, dtype=np.int32)
    slope_low = np.zeros(width, dtype=input.dtype)
    index_up  = np.zeros(width, dtype=np.int32)
    slope_up  = np.zeros(width, dtype=input.dtype)
    index     = np.zeros(width, dtype=np.int32)
    z         = np.zeros(width, dtype=input.dtype)
    y_low     = np.empty(width, dtype=input.dtype)
    y_up      = np.empty(width, dtype=input.dtype)
    s_low, c_low, s_up, c_up, c = 0, 0, 0, 0, 0
    y_low[0] = y_up[0] = 0
    y_low[1] = input[0] - step_size
    y_up[1] = input[0] + step_size
    incr = 1

    for i in range(2, width):
        y_low[i] = y_low[i-1] + input[(i - 1) * incr]
        y_up[i] = y_up[i-1] + input[(i - 1) * incr]

    y_low[width-1] += step_size
    y_up[width-1] -= step_size
    slope_low[0] = np.inf
    slope_up[0] = -np.inf
    z[0] = y_low[0]

    for i in range(1, width):
        c_low += 1
        c_up += 1
        index_low[c_low] = index_up[c_up] = i
        slope_low[c_low] = y_low[i]-y_low[i-1]
        while (c_low > s_low+1) and (slope_low[max(s_low, c_low-1)] <= slope_low[c_low]):
            c_low -= 1
            index_low[c_low] = i
            if c_low > s_low+1:
                slope_low[c_low] = (y_low[i]-y_low[index_low[c_low-1]]) / (i-index_low[c_low-1])
            else:
                slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])

        slope_up[c_up] = y_up[i]-y_up[i-1]
        while (c_up > s_up+1) and (slope_up[max(c_up-1, s_up)] >= slope_up[c_up]):
            c_up -= 1
            index_up[c_up] = i
            if c_up > s_up + 1:
                slope_up[c_up] = (y_up[i]-y_up[index_up[c_up-1]]) / (i-index_up[c_up-1])
            else:
                slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

        while (c_low == s_low+1) and (c_up > s_up+1) and (slope_low[c_low] >= slope_up[s_up+1]):
            c += 1
            s_up += 1
            index[c] = index_up[s_up]
            z[c] = y_up[index[c]]
            index_low[s_low] = index[c]
            slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])
        while (c_up == s_up+1) and (c_low>s_low+1) and (slope_up[c_up]<=slope_low[s_low+1]):
            c += 1
            s_low += 1
            index[c] = index_low[s_low]
            z[c] = y_low[index[c]]
            index_up[s_up] = index[c]
            slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

    for i in range(1, c_low - s_low + 1):
        index[c+i] = index_low[s_low+i]
        z[c+i] = y_low[index[c+i]]
    c = c + c_low-s_low
    j, i = 0, 1
    while i <= c:
        a = (z[i]-z[i-1]) / (index[i]-index[i-1])
        while j < index[i]:
            output[j * incr] = a
            output[j * incr] = a
            j += 1
        i += 1
    return


@njit
def prox_tv1d_cols(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along columns of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_cols):
        _prox_tv1d(stepsize, A[:, i], out[:, i])
    return out.ravel()


@njit
def prox_tv1d_rows(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along rows of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_rows):
        _prox_tv1d(stepsize, A[i, :], out[i, :])
    return out.ravel()


def Read_seg_file(binstart,binlen,binend,bingc):
    """
    read segment file (Generated by DNAcopy.segment)
    seg file: col, chr, start, end, num_mark, seg_mean
    """
    seg_start = []
    seg_end = []
    seg_len = []
    location = []
    seg_pos = []
    for i in range(len(binstart) - 1):
        if binstart[i] + binlen[i] != binstart[i + 1]:
            location.append(i+1)
    count = 0
    with open("seg", 'r') as f1,\
            open('seg2.txt', 'w') as f2:
        for line in f1:
            linestrlist = line.strip().split('\t')
            start = int(linestrlist[2])
            end = int(linestrlist[3])
            seg_pos.append([])
            seg_pos[-1].append(start)
            seg_pos[-1].append(end)
            for j in location:
                if j >= seg_pos[-1][0] and j < seg_pos[-1][-1]:
                    seg_pos[-1].append(j)
                    seg_pos[-1].append(j + 1)
                    seg_pos[-1] = sorted(seg_pos[-1])
            for k in range(0,len(seg_pos[count]),2):
                start = seg_pos[count][k] - 1
                end = seg_pos[count][k+1]-1
                linestrlist[2] = str(binstart[seg_pos[count][k] - 1])
                seg_start.append(seg_pos[count][k]-1)
                linestrlist[3] = str(binend[seg_pos[count][k+1]-1])#
                linestrlist[4] = str(np.sum(binlen[seg_pos[count][k] - 1:seg_pos[count][k+1]]))
                seg_end.append(seg_pos[count][k+1]-1)
                seg_len.append(np.sum(binlen[start:end + 1]))
                linestrlist.append('')
                linestrlist[6] = (str(np.mean(bingc[start:end+1])))
                f2.write('\t'.join(linestrlist) + '\n')
            count += 1
    reseg_Start,reseg_End,reseg_Len,reseg_gc = re_segfile('seg2.txt','reseg.txt',reseg_len)
    return reseg_Start, reseg_End, reseg_Len, reseg_gc


def PCC(data):
    rdmq_1 = np.array(data[['rd','gc','mq']])
    clf = IForest()  # ??IsolationForest??????
    clf.fit(rdmq_1)
    scores_1 = clf.decision_function(rdmq_1)  # ??????
    data['scores'] = scores_1
    threshold_1 = Otsu(scores_1)
    CNV_1 = get_CNV(data, threshold_1)
    CNV_stage1 = combineCNV(CNV_1)
    new_data = data.drop(CNV_1.index)
    new_data.index = range(new_data.shape[0])
    rdmq_2 = np.array(new_data[['rd', 'gc', 'mq']])
    clf.fit(rdmq_2)
    scores_2 = clf.decision_function(rdmq_2)
    new_data['scores'] = scores_2
    threshold_2 = Otsu(scores_2)
    CNV_2 = get_CNV(new_data, threshold_2)
    CNV_stage2 = combineCNV(CNV_2)
    allCNV = pd.concat([CNV_1, CNV_2]).sort_values(by='start').reset_index(drop=True)
    final_CNV = combineCNV(allCNV)

    return final_CNV


def resegment_RD(RD, MQ, start, end):

    reseg_RD = np.full(len(start),0.0)
    reseg_MQ = np.full(len(start),0.0)
    reseg_start = []
    reseg_end = []

    for i in range(len(start)):
        reseg_RD[i] = np.mean(RD[start[i]:end[i]])
        reseg_MQ[i] = np.mean(MQ[start[i]:end[i]])
        reseg_start.append(start[i] + 1)
        reseg_end.append(end[i])

    return reseg_RD, reseg_MQ, reseg_start, reseg_end


def re_segfile(filname,savefile,reseg_length):
    with open(filname,'r') as f1, \
            open(savefile,'w') as f2:
        for line in f1:
            linestrlist = line.strip().split('\t')
            length = int(linestrlist[4])
            if length > reseg_length:
                l = length % reseg_length
                if l:
                    if l >= binSize:
                        reseg_num = length // reseg_length + 1
                        for i in range(reseg_num):
                            if i == 0:
                                linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1)
                                linestrlist[4] = str(reseg_length)
                                f2.write('\t'.join(linestrlist) + '\n')
                            elif i+1 != reseg_num:
                                linestrlist[2] = str(int(linestrlist[2]) + 1 * reseg_length)
                                linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1 )
                                linestrlist[4] = str(reseg_length)
                                f2.write('\t'.join(linestrlist) + '\n')
                            else:
                                linestrlist[2] = str(int(linestrlist[2]) + 1 * reseg_length)
                                linestrlist[3] = str(int(linestrlist[2])  + l - 1)
                                linestrlist[4] = str(l)
                                f2.write('\t'.join(linestrlist) + '\n')

                    else:
                        bin_num = length // reseg_length
                        for i in range(bin_num):
                            if i == 0:
                                if i+1 != bin_num:
                                    linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1)
                                    linestrlist[4] = str(reseg_length)
                                else:
                                    linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1 + l)
                                    linestrlist[4] = str(reseg_length + l)
                                f2.write('\t'.join(linestrlist) + '\n')
                            elif i+1 != bin_num:
                                linestrlist[2] = str(int(linestrlist[2]) + 1 * reseg_length)
                                linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1 )
                                linestrlist[4] = str(reseg_length)
                                f2.write('\t'.join(linestrlist) + '\n')
                            else:
                                linestrlist[2] = str(int(linestrlist[2]) + 1 * reseg_length)
                                linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1 + l)
                                linestrlist[4] = str(reseg_length + l)
                                f2.write('\t'.join(linestrlist) + '\n')

                else:
                    bin_num = length // reseg_length
                    for i in range(bin_num):
                        if i==0:
                            linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1)
                            linestrlist[4] = str(reseg_length)
                            f2.write('\t'.join(linestrlist) + '\n')
                        else:
                            linestrlist[2] = str(int(linestrlist[2]) + 1 * reseg_length)
                            linestrlist[3] = str(int(linestrlist[2]) + reseg_length - 1)
                            linestrlist[4] = str(reseg_length)
                            f2.write('\t'.join(linestrlist) + '\n')
            else:
                f2.write('\t'.join(linestrlist) + '\n')

    tran_start = []
    tran_end = []
    tran_len = []
    tran_gc = []
    with open(savefile, 'r') as f:
        for line in f:
            linestrinfo = line.strip().split('\t')
            tran_start.append(int(linestrinfo[2]))
            tran_end.append(int(linestrinfo[3]) + 1)
            tran_len.append(int(linestrinfo[4]))
            if len(linestrinfo) > 5:
                tran_gc.append(round(float(linestrinfo[6]),2))

    tran_start = np.array(tran_start)
    tran_end = np.array(tran_end)
    tran_len = np.array(tran_len)
    os.remove(filname)
    os.remove(savefile)
    return tran_start, tran_end, tran_len,tran_gc


def get_newbins(new_data):
    new_chr = np.array(new_data['chr'])
    new_start = np.array(new_data['start'])
    new_end = np.array(new_data['end'])
    new_rd = np.array(new_data['rd'])
    new_gc = np.array(new_data['gc'])
    new_mq = np.array(new_data['mq'])

    return new_chr,new_start,new_end,new_rd,new_gc,new_mq


def Otsu(S):
    S = np.round(S,2)
    min_S = np.min(S)
    median_S = np.median(S)
    lower_S = np.quantile(S,0.35,interpolation='lower')
    higer_S = np.quantile(S,0.85,interpolation='higher')
    if(lower_S == min_S):
        lower_S += 0.1
    final_threshold = median_S
    max_var = 0.0
    D_labels = np.full(len(S), 0)
    for i in np.arange(lower_S,higer_S,0.01):
        cur_threshold = round(i,2)
        D0_index = (S < cur_threshold)
        D1_index = (S >= cur_threshold)

        D_labels[D0_index] = 0
        D_labels[D1_index] = 1
        D0 = S[D0_index]
        D1 = S[D1_index]
        S_resample = S.reshape(-1,1)

        new_D,new_label = RandomUnderSampler(random_state=42).fit_resample(S_resample,D_labels)
        new_D0 = new_D.ravel()[new_label==0]
        new_D1 = new_D.ravel()[new_label==1]

        D0_mean = np.mean(new_D0)
        D1_mean = np.mean(new_D1)
        p0 = len(D0)/(len(D0)+len(D1))
        p1 = (1 - p0)
        S_mean = p0*D0_mean + p1*D1_mean
        cur_var = p0*(D0_mean - S_mean)**2 + p1*(D1_mean - S_mean)**2
        if cur_var > max_var:
            final_threshold = cur_threshold
            max_var = cur_var

    return final_threshold


def get_CNV(data,threshold):

    CNVindex = data[np.round(data['scores'],2) >= threshold].index
    Normalindex = data[np.round(data['scores'],2) < threshold].index
    Normalmean = (data['rd'].iloc[Normalindex]).mean()
    gc_mean = (data['gc'].iloc[Normalindex]).mean()
    base = Normalmean * 0.25
    base2 = Normalmean * 0.5
    r1 = data['rd'].iloc[CNVindex] < Normalmean - base2
    r2 = data['rd'].iloc[CNVindex] > Normalmean + base
    real_CNVindex_loss = CNVindex[r1.values]
    real_CNVindex_gain = CNVindex[r2.values]
    cnv_loss = data.iloc[real_CNVindex_loss]
    cnv_gain = data.iloc[real_CNVindex_gain]
    CNVtype_loss = np.full(cnv_loss.shape[0], 'loss')
    cnv_loss.insert(6, 'type', CNVtype_loss)
    CNVtype_gain = np.full(cnv_gain.shape[0], 'gain')
    cnv_gain.insert(6, 'type', CNVtype_gain)
    allCNV = pd.concat([cnv_loss, cnv_gain])
    CNV_length = allCNV['end'] - allCNV['start'] + 1
    allCNV.insert(3, 'length', CNV_length)

    return allCNV


def combineCNV(CNVdata):

    CNVchr,CNVstart,CNVend,CNVRD,CNVgc,CNVmq = get_newbins(CNVdata)
    CNVlen = CNVend - CNVstart + 1
    typeCNV = np.array(CNVdata['type'])
    for i in range(len(CNVRD) - 1):
        if typeCNV[i] == typeCNV[i + 1]:
            len_n = CNVstart[i + 1] - CNVend[i] - 1
            if len_n / (CNVlen[i] + CNVlen[i + 1] + len_n) == 0:
                CNVstart[i + 1] = CNVstart[i]
                CNVlen[i + 1] = CNVend[i + 1] - CNVstart[i + 1] + 1
                typeCNV[i] = 0
                CNVRD[i + 1] = (CNVRD[i] + CNVRD[i+1]) / 2
                CNVmq[i + 1] = (CNVmq[i] + CNVmq[i + 1]) / 2
                CNVgc[i + 1] = (CNVgc[i] + CNVgc[i + 1]) / 2
    index = typeCNV != 0

    CNVRD = CNVRD[index]
    CNVchr = CNVchr[index]
    CNVstart = CNVstart[index]
    CNVend = CNVend[index]
    CNVlen = CNVlen[index]
    CNVmq = CNVmq[index]
    CNVgc = CNVgc[index]
    CNVtype = typeCNV[index]

    CNVdata = [*zip(CNVchr, CNVstart, CNVend, CNVlen, CNVtype)]
    final_CNV = pd.DataFrame(CNVdata, columns=['chr', 'start', 'end', 'size', 'type'])

    return final_CNV

'''
 use rd, gc and mq as features

'''


def calculate_gc_content(sequence):
    g = sequence.upper().count('G')
    c = sequence.upper().count('C')
    return (g + c) / len(sequence) if len(sequence) > 0 else 0


def extract_features(bam_file, window_size=5000, threshold=0.2, normal_bam='normal.bam', targets_bed='targets.bed',
                     reference='reference.fasta', output_dir='output'):
    bam_file = pysam.AlignmentFile(bam_file, "rb")
    coverage = []
    #gc_content = []

    for pileupcolumn in bam_file.pileup():
        coverage.append(pileupcolumn.n)
        #seq = ''
        #for pileupread in bam.fetch(reference=pileupcolumn.reference_name, start=pileupcolumn.pos, end=pileupcolumn.pos + 1):
            #seq += pileupread.query_sequence if pileupread.query_sequence else ''
        #gc_content.append(calculate_gc_content(seq))

    bam_file.close()

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


if __name__ == "__main__":
    # ??????bam_file, reference, outfile, reseg_len
    bam_file = sys.argv[1]
    reference = sys.argv[2]
    outfile = sys.argv[3]
    reseg_len = int(sys.argv[4])

    # ???????????????????????
    target_chr = "chr19"

    print("Current working directory:", os.getcwd())

    # ?? BAM ????????? extract_features ?????????
    features = extract_features(bam_file)
    features_list = []
    if features is not None:
        features_list.append(features)
    else:
        print("No features extracted. Please check BAM file.")

    if features_list:
        features_df = pd.DataFrame(features_list)
        output_path = os.path.join(os.getcwd(), 'data/bam_features.csv')
        print("Output CSV file path:", output_path)
        features_df.to_csv(output_path, index=False)
        print("Features have been written to", output_path)

    # ????????? binSize, threshold, step_size, rd????????????????????
    rf_window_size = joblib.load('rf_window_size_model.pkl')
    rf_threshold = joblib.load('rf_threshold_model.pkl')
    rf_step_size = joblib.load('rf_step_size_model.pkl')
    rf_rd_size = joblib.load('rf_rd_model.pkl')

    feature_vector = [[features['mean_coverage'], features['stddev_coverage'],
                       features['peak_coverage'], features['repeat_fraction']]]
    predicted_window_size = int(rf_window_size.predict(feature_vector)[0])
    predicted_threshold = float(rf_threshold.predict(feature_vector)[0])
    predicted_step_size = float(rf_step_size.predict(feature_vector)[0])
    predicted_rd = float(rf_rd_size.predict(feature_vector)[0])

    # ????????????
    binSize = predicted_window_size
    threshold = predicted_threshold
    alpha = predicted_step_size

    # ????????????????????
    refDict = read_ref_file(reference)
    if target_chr in refDict:
        refSeq = refDict[target_chr]
        chrLen = len(refSeq)
    else:
        raise ValueError("Reference file does not contain " + target_chr)

    # ?????????????????????
    ReadCount = np.full(chrLen, 0)
    Mapq = np.full(chrLen, 0)

    # ?????????? read ?????????????bin_pos?
    ReadCount, bin_pos, Mapq = get_RC(bam_file, ReadCount, Mapq, target_chr, chrLen)

    # ?? ReadDepth??? ReadDepth ??????????? refSeq ? bin_pos?
    bin_start, bin_end, bin_len, bin_RD, bin_gc = ReadDepth(ReadCount, refSeq, bin_pos)

    print("bin_RD length:", len(bin_RD))
    print("bin_RD contains NaN:", np.isnan(bin_RD).any())
    print("bin_RD positive count:", np.count_nonzero(bin_RD > 0))

    if len(bin_RD) == 0:
        print("Warning: bin_RD is empty!")
    else:
        nan_count = np.isnan(bin_RD).sum()
        inf_count = np.isinf(bin_RD).sum()
        print(f"bin_RD has NaNs: {nan_count}, Infs: {inf_count}")

    if len(bin_RD) > 0 and np.any(~np.isnan(bin_RD)):
        mean_rd = np.mean(bin_RD[~np.isnan(bin_RD)])
    else:
        mean_rd = float('nan')
    print("mean_rd:", mean_rd)

    if np.isnan(mean_rd):
        print("Detailed check of bin_RD:", bin_RD)

    # ???? R ???? CBS ??
    with open(outfile + '.txt', 'w') as file:
        for value in bin_RD:
            file.write(str(value) + '\n')
    subprocess.call('Rscript CBS_data.R ' + outfile, shell=True)
    os.remove(outfile + '.txt')

    # ??????? Read_seg_file?resegment_RD?prox_tv1d ? PCC ???
    seg_start, seg_end, seg_len, reseg_gc = Read_seg_file(bin_start, bin_len, bin_end, bin_gc)
    reseg_count, reseg_mq, reseg_start, reseg_end = resegment_RD(ReadCount, Mapq, seg_start, seg_end)
    reseg_mq /= reseg_count
    res_rd = prox_tv1d(alpha, reseg_count)
    reseg_count = res_rd
    reseg_chr = [target_chr] * len(reseg_count)
    data = list(zip(reseg_chr, reseg_start, reseg_end, reseg_count, reseg_gc, reseg_mq))
    data = pd.DataFrame(data, columns=['chr', 'start', 'end', 'rd', 'gc', 'mq'])

    called_CNVs = PCC(data)
    print('CNVs:\n', called_CNVs)
    with open(outfile + '.result.txt', 'w') as Outfile:
        called_CNVs.to_string(Outfile)