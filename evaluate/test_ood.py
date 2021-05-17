import numpy as np
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt    

class Metric(object):
    def __init__(self, ind, ood, gap=30000):
        
        self.ind = np.sort(ind)
        self.ood = np.sort(ood)
        self.ind_len = len(self.ind)
        self.ood_len = len(self.ood)
        self.total = np.concatenate((self.ind,ood))
        self.total = np.sort(self.total)
        self.gap = (self.total[-1] - self.total[0])/gap

        self.metric()
        
    def metric(self):
        self.fpr_tpr95()
        self.auroc()
        self.auprIn()
        self.auprOut()

    def fpr_tpr95(self):
        # calculate the falsepositive error when tpr is 95%
        delta = self.ind[self.ind_len//20]

        tpr = np.sum(np.sum(self.ind >= delta)) / np.float(self.ind_len)
        fpr = np.sum(np.sum(self.ood >= delta)) / np.float(self.ood_len)
            
        if tpr <= 0.9505 and tpr >= 0.9495:
            self.fpr = fpr * 100
        else:
            print("Can't find delta")
            self.fpr = 100
        
    def auroc(self):
        
        fpr_list, tpr_list = [], []
        ind_idx, ood_idx = 0,0
        
        for i in range(self.ind_len + self.ood_len):
            if self.ind[ind_idx] >= self.ood[ood_idx] :
                if ood_idx != self.ood_len -1 :
                    ood_idx += 1
            else:
                if ind_idx != self.ind_len -1 :
                    ind_idx += 1

            # coord.append((fpr,tpr))
            tpr_list.append((self.ind_len - ind_idx) / self.ind_len)
            fpr_list.append((self.ood_len - ood_idx) / self.ood_len)
        
        # ascending order
        fpr_list = fpr_list[::-1]
        tpr_list = tpr_list[::-1]

        auroc = 0
        for idx in range(len(fpr_list)):
            if idx >=1 and fpr_list[idx] != fpr_list[idx-1]:
                auroc += tpr_list[idx]*(fpr_list[idx] - fpr_list[idx-1])
        
        self.aur = auroc * 100

        # Save Roc curve image
        plt.axis([0,1,0,1])
        plt.plot(fpr_list, tpr_list, label = "ROC Curve", color ='darkred')
        plt.savefig('auroc.png', dpi=300)
        plt.clf()

    def auprIn(self):
        # calculate the AUPR
        auprIn = 0.0
        recallTemp = 1.0
        for delta in tqdm(np.arange(self.total[0], self.total[-1], self.gap)):
            tp = np.sum(np.sum(self.ind >= delta)) / np.float(len(self.ind))
            fp = np.sum(np.sum(self.ood >= delta)) / np.float(len(self.ood))
            if tp + fp == 0: continue
            precision = tp / (tp + fp)
            recall = tp
            auprIn += (recallTemp-recall)*precision
            recallTemp = recall
        auprIn += recall * precision
        self.auprin = auprIn * 100

    def auprOut(self):
        # calculate the AUPR
        auprOut = 0.0
        recallTemp = 1.0
        for delta in tqdm(np.arange( self.total[-1],self.total[0], -self.gap)):
            fp = np.sum(np.sum(self.ind < delta)) / np.float(len(self.ind))
            tp = np.sum(np.sum(self.ood < delta)) / np.float(len(self.ood))
            if tp + fp == 0: break
            precision = tp / (tp + fp)
            recall = tp
            auprOut += (recallTemp-recall)*precision
            recallTemp = recall
        auprOut += recall * precision
        self.auprout = auprOut * 100
        

if __name__ == "__main__":
    test_auroc(120,gap=50000)