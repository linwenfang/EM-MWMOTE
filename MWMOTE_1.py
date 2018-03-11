#coding=utf-8
import bisect
import linecache
import logging
import math
import random

import numpy as np
import pandas as pd

from EM import MVNImputer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(message)s')


def MVNImputer(df, epsilon=1e-5, maxiter=100):
    data = df.values
    n_obs, n_var = data.shape
    mu_init = np.nanmean(data, axis=0)
    sigma_init = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(i, n_var):
            vecs = data[:, [i, j]]
            vecs = vecs[~np.any(np.isnan(vecs), axis=1), :].T
            if len(vecs) > 0:
                cov = np.cov(vecs)
                cov = cov[0, 1]
                sigma_init[i, j] = cov
                sigma_init[j, i] = cov
            else:
                sigma_init[i, j] = 1.0
                sigma_init[j, i] = 1.0

    print 'Start EasyEnsemble iteration.'
    pre_mu = mu_init
    pre_sigma = sigma_init
    pre_lik = -np.inf
    for n_iter in range(maxiter):
        # E step
        temp_data = np.copy(data)
        for i in range(n_obs):
            if np.any(np.isnan(temp_data[i, :])):
                nans = np.isnan(temp_data[i, :])
                # conditional distribution of multivariate normal
                offset_mu = np.dot(pre_sigma[nans, :][:, ~nans], np.dot(np.linalg.inv(pre_sigma[~nans, :][:, ~nans]),
                                                                        (temp_data[i, ~nans] - pre_mu[~nans])[:,
                                                                        np.newaxis]))
                temp_data[i, nans] = pre_mu[nans] + offset_mu
        # M step
        new_mu = np.mean(temp_data, axis=0)
        new_sigma = np.cov(temp_data.T)
        new_lik = -0.5 * n_obs * (n_var * np.log(2 * np.pi) + np.log(np.linalg.det(new_sigma)))
        for i in range(n_obs):
            new_lik -= 0.5 * np.dot((temp_data[i, :] - new_mu),
                                    np.dot(np.linalg.inv(new_sigma), (temp_data[i, :] - new_mu)[:, np.newaxis]))
        print 'ITER =', n_iter, '\tLog likelihood =', new_lik
        if new_lik - pre_lik < epsilon:
            imputed = temp_data
            break
        pre_mu = new_mu
        pre_sigma = new_sigma
        pre_lik = new_lik
    else:
        imputed = temp_data
    print 'End EasyEnsemble iteration.\n'
    return pd.DataFrame(imputed, index=df.index, columns=df.columns)


def dic_trans_txt(a, b):
    line=a+1
    theline = linecache.getline(r'knn.txt', line)
    arr= theline.split(',')
    arr=arr[:-1]
    return arr[b]

class Knn:
    """docstring for Knn"""

    def __init__(self):
        self.data = []
        self.dic = {}

    def fit(self, data):
        self.data = data
        self.real_indices = range(len(data))
        with open('knn.txt', 'w') as writer:
            for i in range(len(data)):
                for j in range(len(data)):
                    dis = math.sqrt(math.fsum(((a - b) ** 2 for a, b in zip(self.data[i], self.data[j]))))
                    writer.write(str(dis)+',')
                writer.write('\n')
                print i


    def fit_subset(self, indices):
        self.real_indices = indices

    def get_dis(self, a, b):

        return dic_trans_txt(a,b)

    def kneighbors(self, instance_index, n_neighbors, return_distance=False):
        result = []
        #存储result,最大为1*len(X)
        for i in self.real_indices:
            distance = dic_trans_txt(instance_index,i)
            result.append((distance, i))
        result = sorted(result)[:n_neighbors]

        if return_distance:
            return ([i[1] for i in result], [i[0] for i in result])
        else:
            return [i[1] for i in result]



class WeightedSampleRandomGenerator(object):
    def __init__(self, indices, weights):
        self.totals = []
        self.indices = indices
        running_total = 0

        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = random.random() * self.totals[-1]
        return self.indices[bisect.bisect_right(self.totals, rnd)]

    def __call__(self):
        return self.next()





def MWMOTE(X, Y, N, k1=5, k2=3, k3=0.5, C_th=5, CMAX=2, C_p=3, return_mode='only'):
    # logger.debug('MWMOTE: Starting with %d instances' % len(Y))
    # Generating indices of S_min, S_maj
    S_min, S_maj = [], []
    for index, i in enumerate(Y):#将多数类和少数类的下标分开
        if i == 1:
            S_min.append(index)
        else:
            S_maj.append(index)
    if type(k3) == float:
        k3 = int(round(len(S_min) * k3))#确定k3，距离每个边界多数类来说，计算其最近的K3个少数类样本
    k = Knn()

    # logger.debug(' Step   0: Computing Knn table')
    k.fit(X)
    print 'step1-2'
    # Step 1~2: Generating S_minf
    S_minf = []
    for i in S_min:
        neighbors = k.kneighbors(i, k1 + 1)  # remove itself from neighbors
        neighbors.remove(i)
        if not all((neighbor in S_maj) for neighbor in neighbors):
            S_minf.append(i)

    print S_minf
    # logger.debug(' Step 1~2: %d in S_minf' % len(S_minf))
    print 'step3-4'
    # Step 3~4: Generating S_bmaj
    k.fit_subset(S_maj)
    S_bmaj = []
    for i in S_minf:
        neighbors = k.kneighbors(i, k2)
        S_bmaj.extend(neighbors)
    S_bmaj = list(set(S_bmaj))
    # logger.debug(' Step 3~4: %d in S_bmaj' % len(S_bmaj))
    print 'step5-6'
    # Step 5~6: Generating S_imin

    k.fit_subset(S_min)
    S_imin = []
    N_min = {}
    for i in S_bmaj:
        neighbors = k.kneighbors(i, k3)
        S_imin.extend(neighbors)
        N_min[i] = neighbors
    S_imin = list(set(S_imin))


    # logger.debug(' Step 5~6: %d in S_imin' % len(S_imin))
    print 'step7-9'
    # Step 7~9: Generating I_w, S_w, S_p
    I_w = {}
    for y in S_bmaj:
        sum_C_f = 0.
        for x in S_imin:
            # closeness_factor
            if x not in N_min[y]:
                closeness_factor = 0.
            else:
                distance_n = math.sqrt(math.fsum(((a - b) ** 2 for a, b in zip(X[x], X[y])))) / len(X[x])
                '''抛出distance_n为0的情况'''
                try:
                    closeness_factor = min(C_th, (1 / distance_n)) / C_th * CMAX
                except ZeroDivisionError as c:

                    closeness_factor = min(C_th, (1 / (distance_n+1e-6))) / C_th * CMAX
            I_w[(y, x)] = closeness_factor
            sum_C_f += I_w[(y, x)]
        for x in S_imin:
            closeness_factor = I_w[(y, x)]
            density_factor = closeness_factor / sum_C_f
            I_w[(y, x)] = closeness_factor * density_factor

    S_w = {}
    for x in S_imin:
        S_w[x] = math.fsum((I_w[(y, x)]) for y in S_bmaj)

    S_p = {}  # actually useless
    WeightSum = math.fsum(S_w.values())
    for x in S_w:
        S_p[x] = float(S_w[x]) / WeightSum
    # logger.debug(' Step 7~9: %d in I_w' % len(I_w))


    print 'step10'
    print 'step11'
    # Step 11: Generating X_gen, Y_gen
    X_gen = []
    sample = WeightedSampleRandomGenerator(S_w.keys(), S_w.values())
    #data0 = np.loadtxt('abalone_0_18_9.csv', delimiter=',')
    #X = data0[:, :len(data0[0]) - 1]

    #N=20
    for z in xrange(N):
        hang = sample()  # 选取第x个样本
        lie = random.randint(0,len(X[0])-1)
        original = X[hang][lie]
        X[hang][lie] = np.nan

        d = {}
        for i in range(1, len(X[0]) + 1):
            m = X[:, i - 1].tolist()
            #print m
            d[str(i)] = m

        df = pd.DataFrame(d, index=range(1, len(X) + 1))
        imputed = MVNImputer(df)
        imputed_darray = np.array(imputed)  # np.ndarray()
        imputed_list = imputed_darray.tolist()  # list
        imputation_res = imputed_list[hang][lie]
        print original
        print imputation_res
        s=X[hang]
        s[lie]=imputation_res
        X_gen.append(s)

    Y_gen = [1 for z in xrange(N)]

    X = np.concatenate((X, X_gen), axis=0)
    Y = np.concatenate((Y, Y_gen), axis=0)
    return (X,Y)
