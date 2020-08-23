import numpy as np
import math
from scipy.spatial.distance import pdist
'''
Sampen function
From: Víctor Martínez-Cagigal (2018). Sample Entropy. Mathworks.
Avaliable on: https://www.mathworks.com/matlabcentral/fileexchange/69381-sample-entropy

CosEn function
from: E. M. Cirugeda-Roldán et al., "Customization of entropy estimation measures for human 
arterial hypertension records segmentation," 2012 Annual International Conference of the IEEE 
Engineering in Medicine and Biology Society, San Diego, CA, 2012, pp. 33-36, doi: 10.1109/EMBC.2012.6345864.
'''
'''data: ECG signal
   m = embedding distance (m<len(data))
   r = tolerance
   dist_type = Distance type, can be braycurtis, canberra, chebyshev, cityblock, correlation, 
   cosine, dice, euclidean, hamming, jaccard, jensenshannon, kulsinski, mahalanobis, 
   matching, minkowski, rogerstanimoto, russellrao, seuclidean, sokalmichener, 
   sokalsneath, sqeuclidean, yule    
'''


def cos_en(data, m=30, r=0.05, dist_type='chebyshev'):
    cosen = 0
    n = len(data)
    sigma = np.std(data)
    matches = np.nan*np.ones((m, n-1))
    for i in range(m):
        matches[i, 0:n-1-i] = data[i:n-1]

    matches = np.transpose(matches)
    d_m = pdist(matches[:, 1:m], dist_type)

    if len(d_m) == 0:
        cosen = math.inf
    else:
        d_m1 = pdist(matches[:, 0:m+1], dist_type)
        b = np.sum(d_m <= r*sigma)
        if b == 0:
            cosen = math.inf
        a = np.sum(d_m1 <= r*sigma)
        cosen = -np.log((a/b)*((n-m+1)/(n-m-1))) - np.log(2*r) - np.log(np.abs(np.mean(data)))

    if cosen == math.inf:
        cosen = -np.log(2/((n-m-1)*(n-m)))

    return round(cosen, 6)
