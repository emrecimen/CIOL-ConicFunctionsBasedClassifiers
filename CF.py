import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
import random

import math
from sklearn import metrics
np.random.seed(0)
import pandas as pd
from gurobipy import *

class PCF(BaseEstimator):
    def __init__(self, norm):
        self.norm = norm
        self.G=[]


    def split_AB(self,X,y,cls):
        A=X[(y==cls)]
        B=X[(y!=cls)]
        return A, B

    def fit_PCF(self,A, B):
        Ajr=A.values
        Bjr=B.values
        # Create optimization model
        m = Model('PCF')
        f_size=Ajr.shape[1]
        cjr = np.mean(Ajr, axis=0)

        # Create variables
        gamma = m.addVar(vtype=GRB.CONTINUOUS, lb=1, name='gamma')
        w = [None]*f_size
        for a in range(f_size):
            w[a] = m.addVar(vtype=GRB.CONTINUOUS, lb=-1000, name='w[%s]' % a)

        ksi = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='ksi')


        hataA = {}
        hataB = {}


        for i in range(len(Ajr)):
            hataA[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='hataA[%s]' % i)


            m.addConstr(quicksum((Ajr[i][j] - cjr[j]) * w[j] for j in range(len(cjr))) + (
                        ksi * np.linalg.norm((Ajr[i,:] - cjr), ord=self.norm)) - gamma + 1.0 <= hataA[
                            i])

        for z in range(len(Bjr)):
            hataB[z] = m.addVar(vtype=GRB.CONTINUOUS, lb=0,  name='hataB[%s]' % z)


            m.addConstr(quicksum((Bjr[z][r] - cjr[r]) * -w[r] for r in range(len(cjr))) - (
                        ksi * np.linalg.norm((Bjr[z,:] - cjr), ord=self.norm)) + gamma + 1.0 <= hataB[
                            z])




        m.setObjective((quicksum(hataA[k] for k in range(len(hataA))) / len(hataA)) + (quicksum(hataB[l] for l in range(len(hataB))) / len(hataB)), GRB.MINIMIZE)
        m.setParam("OutputFlag",False)
        m.update()
        # Compute optimal solution
        m.optimize()


        ww = []
        for i in range(f_size):
            ww.append(w[i].X)

        return {'w': ww, 'gamma': gamma.x, 'ksi': ksi.x, 'c': cjr}

    def displayTerminal(self,ith,total):

        percent = ("{0:." + str(2) + "f}").format(100 * (ith / float(total)))
        filledLength = int(100 * float(ith) / float(total))
        bar = '█' * filledLength + '-' * (100 - filledLength)
        print('\r%s |%s| %s%% %s' % ('', bar, percent, ''),end='')
        if ith == total:
            print()

    def pcfVal(self, g, x):
        w=g['w']
        ksi=g['ksi']
        gamma=g['gamma']
        c=g['c']
        val = np.dot(w, x - c) + ksi * np.linalg.norm((x - c), ord=self.norm) - gamma
        return val

    def accuracy(self, predicted, real):
        return round(100.0 * (np.sum(real == predicted)) / len(predicted), 2)


class ClusterbasePCF(PCF):
    def __init__(self, norm=1, clustering='k-means++',n_cluster=2,n_init=1):
        PCF.__init__(self, norm)
        self.clustering=clustering
        self.n_cluster=n_cluster
        self.n_init=n_init


    def fit(self, X, y):
        self.G=[]

        n_class = max(y) + 1
        ith=1


        for cls in range(n_class):
            g=[]
            A, B = self.split_AB(X, y, cls)


            if self.clustering=='k-means++':
                kmeans = KMeans(init='k-means++', n_clusters=self.n_cluster, n_init=self.n_init)
                kmeans.fit(A)
                clusters = kmeans.labels_
            elif self.clustering=="AgglomerativeClustering":
                hierarchial = AgglomerativeClustering( n_clusters=self.n_cluster).fit(A)
                clusters = hierarchial.labels_

            for i in range(self.n_cluster):

                Ai = A[(clusters==i)]
                gi = self.fit_PCF(Ai, B)
                g.append(gi)
                self.displayTerminal(ith,self.n_cluster*n_class)
                ith=ith+1
            self.G.append(g)

        return self

    def predict(self, X,y=None):
        prd = np.zeros(X.shape[0])

        for cnt in range(X.shape[0]):
            x=X.values[cnt]
            t = 0
            minVal = float('inf')
            for g in self.G:
                for gi in g:
                    val = self.pcfVal(gi, x)

                    if val < minVal:
                        minVal = val
                        prd[cnt] = t
                t = t + 1


        return np.asarray(prd, dtype=np.intp)

    def score(self, X, y=None):

        prd = self.predict(X)
        return self.accuracy(prd, y)


class EnsambleCF(PCF):

    def __init__(self, norm=1, member=10, maxRatio = 0.5,alfa=0.01):

        PCF.__init__(self, norm)
        self.member = member
        self.maxRatio = maxRatio
        self.featureMap=[None]*member
        self.n_class=0
        self.weights=np.ones(self.member)
        self.alfa=alfa

    def fit_Ensamble_PCF(self, A, B):
        Ajr = A.values
        Bjr = B.values
        # Create optimization model
        m = Model('PCF')
        f_size = Ajr.shape[1]
        cjr = np.mean(Ajr, axis=0)

        # Create variables
        gamma = m.addVar(vtype=GRB.CONTINUOUS, lb=1, name='gamma')
        w = [None] * f_size
        w2 = [None] * f_size
        for a in range(f_size):
            w[a] = m.addVar(vtype=GRB.CONTINUOUS, name='w[%s]' % a)
            w2[a] = m.addVar(vtype=GRB.CONTINUOUS, name='w2[%s]' % a)

        ksi = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='ksi')

        hataA = {}
        hataB = {}



        for i in range(len(Ajr)):
            hataA[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='hataA[%s]' % i)

            m.addConstr(quicksum((Ajr[i][j] - cjr[j]) * (w[j]-w2[j]) for j in range(len(cjr))) + (
                    ksi * np.linalg.norm((Ajr[i, :] - cjr), ord=self.norm)) - gamma + 1.0 <= hataA[
                            i])

        for z in range(len(Bjr)):
            hataB[z] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name='hataB[%s]' % z)

            m.addConstr(quicksum((Bjr[z][r] - cjr[r]) * -(w[r]-w2[r]) for r in range(len(cjr))) - (
                    ksi * np.linalg.norm((Bjr[z, :] - cjr), ord=self.norm)) + gamma + 1.0 <= hataB[
                            z])

        m.setObjective((quicksum(hataA[k] for k in range(len(hataA))) / len(hataA)) + (
                    quicksum(hataB[l] for l in range(len(hataB))) / len(hataB))+self.alfa*(quicksum(w[k]+w2[k] for k in range(f_size))+ksi+gamma), GRB.MINIMIZE)
        m.setParam("OutputFlag", False)
        m.update()
        # Compute optimal solution
        m.optimize()

        ww = []
        for i in range(f_size):
            ww.append(w[i].X-w2[i].X)

        return {'w': ww, 'gamma': gamma.x, 'ksi': ksi.x, 'c': cjr}

    def fit(self, X, y):
        ith=1
        self.n_class = max(y) + 1
        self.G = [[0] * self.member for i in range(self.n_class)]

        f_size=X.shape[1]
        max_feature= math.ceil(f_size*self.maxRatio)
        self.featureMap = [None] * self.member

        for i in range(self.member):
            #member_f_size=np.random.randint(low=1,high=max_feature+1)
            member_f_size =max_feature
            member_features= random.sample(range(0, f_size), member_f_size)
            Xi=X.iloc[:,member_features]
            self.featureMap[i]=member_features

            for j in range(self.n_class):

                Aij, B = self.split_AB(Xi, y, j)
                gji = self.fit_Ensamble_PCF(Aij, B)
                self.G[j][i]=gji
                self.displayTerminal(ith, self.member * self.n_class)
                ith = ith + 1
        #self.calcMemberWeights(X, y)
        return self

    def predict(self, X,y=None):

        prd = np.zeros((X.shape[0],self.n_class))

        for i in range(self.member):
            Xi=X.iloc[:,self.featureMap[i]]
            for cnt in range(X.shape[0]):
                x=Xi.values[cnt]

                minVal = float('inf')
                for j in range(self.n_class):
                    val = self.pcfVal(self.G[j][i], x)
                    if val < minVal:
                        minVal = val
                        pTem = j

                prd[cnt,pTem]=prd[cnt,pTem]+self.weights[i]
        prdFin=np.argmax(prd,axis=1)


        return np.asarray(prdFin, dtype=np.intp)

    def calcMemberWeights(self, X,y):

        prd = np.zeros(X.shape[0])


        for i in range(self.member):
            Xi=X.iloc[:,self.featureMap[i]]
            for cnt in range(X.shape[0]):
                x=Xi.values[cnt]

                minVal = float('inf')
                for j in range(self.n_class):
                    val = self.pcfVal(self.G[j][i], x)
                    if val < minVal:
                        minVal = val
                        pTem = j
                prd[cnt]=pTem
            self.weights[i]=self.accuracy(prd, y)
        return

    def score(self, X, Y):

        ac=self.accuracy(self.predict(X,Y),Y)

        return ac




class OPCF(PCF):

    def __init__(self, lmbda= 0.1 ,norm=1, clustering='k-means++', n_cluster=1, n_init=1):

        PCF.__init__(self, norm)
        self.clustering = clustering
        self.n_cluster = n_cluster
        self.n_init = n_init
        self.lmbda = lmbda



    def fit_OPCF(self, A):

        Ajr=A.values
        # Create optimization model
        m = Model('OPCF')
        f_size=Ajr.shape[1]
        cjr = np.mean(Ajr, axis=0)

        # Create variables
        gamma = m.addVar(vtype=GRB.CONTINUOUS, name='gamma')
        w = [None]*f_size
        for a in range(f_size):
            w[a] = m.addVar(vtype=GRB.CONTINUOUS, lb=-100, ub=1, name='w[%s]' % a)

        ksi = m.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=10, name='ksi')


        hataA = {}

        for i in range(len(Ajr)):
            hataA[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='hataA[%s]' % i)


            m.addConstr(quicksum((Ajr[i][j] - cjr[j]) * w[j] for j in range(len(cjr))) + (
                        ksi * np.linalg.norm((Ajr[i,:] - cjr), ord=self.norm)) - gamma + 1 <= hataA[i])


        m.setObjective((quicksum(quicksum((Ajr[i][j] - cjr[j]) * -w[j] for j in range(len(cjr))) - (
                    ksi *  np.linalg.norm((Ajr[i,:] - cjr), ord=self.norm)) + gamma for i in
                                 range(len(Ajr)))) + self.lmbda * (quicksum(hataA[k] for k in range(len(hataA)))),
                       GRB.MINIMIZE)
        m.setParam("OutputFlag", False)
        m.update()
        # Compute optimal solution
        m.optimize()
        m.write('model.sol')
        ww = []
        for i in range(f_size):
            ww.append(w[i].X)

        return {'w': ww, 'gamma': gamma.x, 'ksi': ksi.x, 'c': cjr}

    def fit(self, X, y):
        self.G=[]

        ith=1

        if self.clustering=='k-means++':
            kmeans = KMeans(init='k-means++', n_clusters=self.n_cluster, n_init=self.n_init)
            kmeans.fit(X)
            clusters = kmeans.labels_


        for i in range(self.n_cluster):

            Ai = X[(clusters==i)]
            g = self.fit_OPCF(Ai)
            self.displayTerminal(ith,self.n_cluster)
            ith = ith + 1
            self.G.append(g)

        return self

    def predict(self, X,y=None):
        prd = np.zeros(X.shape[0])
        values = np.zeros(X.shape[0])

        for cnt in range(X.shape[0]):
            x=X.values[cnt]
            minVal = float('inf')
            for g in self.G:

                val = self.pcfVal(g, x)
                if val < minVal:
                    minVal = val
            values[cnt]=minVal

            if minVal<=0 :
                prd[cnt]=1

        return np.asarray(prd, dtype=np.intp), np.asarray(values)

    def score(self, X, y=None):

        prd, values = self.predict(X)


        return self.accuracy(prd, y), f1_score(y, prd)


