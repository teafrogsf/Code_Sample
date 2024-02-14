import igraph as ig
from random import uniform, shuffle
from random import randint as rd
from random import gauss as nm
import heapq as hq
from collections import deque
from matplotlib import pyplot as plt

class DAs:
    def __init__(self):
        self.out_seller = deque()
        self.out_buyer = deque()
        self.q_seller = []
        self.q_buyer = []
        self.p_s = -1
        self.p_b = -1

    def gen(self, N, SL, SR, VL, VR, REWIRING_PROB, CONNECT_PROB, INITIAL_PARTICIPANT_NUM, opt):
        # Setting Variant
        self.n = N
        self.val = [0 for i in range(self.n)]
        self.invited = [False for i in range(self.n)]
        self.ori_invited = [False for i in range(self.n)]
        self.s = rd(SL, SR)
        self.b = self.n - self.s

        for i in range(self.n):
            self.val[i] = rd(VL, VR)

        # Initialization of Graph

        # Randomly construct a graph

        self.g = ig.GraphBase.Watts_Strogatz(dim=1, size=self.n, nei=int(CONNECT_PROB*self.n//2), p=REWIRING_PROB)
        self.edges = self.g.get_adjacency()
        
        # Randomly choose sellers and buyers

        lis = [i for i in range(self.n)]
        shuffle(lis)
        for i in range(min(INITIAL_PARTICIPANT_NUM, self.n)):
            self.ori_invited[lis[i]] = self.invited[lis[i]] = True

    def align(self):
        while len(self.q_seller) != len(self.q_buyer):
            if len(self.q_seller) > len(self.q_buyer):
                self.out_seller.append(hq.heappop(self.q_seller))
            else:
                self.out_buyer.append(hq.heappop(self.q_buyer))

    def match(self):
        if len(self.q_seller) != 0:
            x, y = self.q_seller[0], self.q_buyer[0]
            while -x[0] > y[0]:
                self.out_seller.append(hq.heappop(self.q_seller))
                self.out_buyer.append(hq.heappop(self.q_buyer))
                if len(self.q_seller) == 0:
                    break
                x, y = self.q_seller[0], self.q_buyer[0]

    def review(self):
        assert len(self.q_seller) == len(self.q_buyer)
        sold = [False for u in range(self.n)]
        sw = 0

        for x in self.q_seller:
            sold[x[1]] = True
        for u in range(self.s):
            if not sold[u]:
                sw += self.val[u]
        for x in self.q_buyer:
            sw += x[0]

        return sw
        
    def Optimal(self):
        self.q_seller = []
        for u in range(self.s):
            hq.heappush(self.q_seller, (-self.val[u], u)) # max heap

        # Use min heap to pop unmatchable buyers
        self.q_buyer = []
        for u in range(self.s, self.n):
            hq.heappush(self.q_buyer, (self.val[u], u)) # min heap

        self.align()
        self.match()
        return self.review()

    def MTRForInit(self):
        # Use max heap to pop unmatchable sellers
        self.q_seller = []
        for u in range(self.s):
            if self.invited[u]:
                hq.heappush(self.q_seller, (-self.val[u], u)) # max heap

        # Use min heap to pop unmatchable buyers
        self.q_buyer = []
        for u in range(self.s, self.n):
            if self.invited[u]:
                hq.heappush(self.q_buyer, (self.val[u], u)) # min heap

        # Align sellers and buyers
        self.align()
        # Pop and store the unmatchbale sellers and buyers
        self.match()
        
        # calculate p_0
        if len(self.q_seller) != 0 and len(self.q_buyer) != 0:
            flag = False
            if len(self.out_seller) != 0 and len(self.out_buyer) != 0:
                x, y = self.out_seller[len(self.out_seller)-1], self.out_buyer[len(self.out_buyer)-1]
                p_0 = (x[0] + y[0]) / 2
                if -self.q_seller[0][0] <= p_0 <= self.q_buyer[0][0]:
                    flag = True # k pairs
            if not flag: # k-1 pairs
                self.out_seller.append(hq.heappop(self.q_seller))
                self.out_buyer.append(hq.heappop(self.q_buyer))

            self.p_s, self.p_b = -self.q_seller[0][0], self.q_buyer[0][0] # Initialize reserve price

        return self.review()

    def TRP(self):
        # Step 1 and Step 2
        while len(self.q_seller) != 0 and -self.q_seller[0][0] > self.p_s:
            self.out_seller.append(hq.heappop(self.q_seller))
        while len(self.q_buyer) != 0 and self.q_buyer[0][0] < self.p_b:
            self.out_buyer.append(hq.heappop(self.q_buyer))
        q_s = len(self.q_seller)
        q_b = len(self.q_buyer)

        # Step 3
        if q_s > q_b:
            tmp_lis = []
            for i in range(q_s-q_b-1): # get the q_b+1 th seller
                tmp_lis.append(hq.heappop(self.q_seller))
            self.p_s = -self.q_seller[0][0]

            for i in range(q_s-q_b-1):
                hq.heappush(self.q_seller, tmp_lis[i])
        elif q_s < q_b:
            tmp_lis = []
            for i in range(q_b-q_s-1): # get the q_s+1 th buyer
                tmp_lis.append(hq.heappop(self.q_buyer))
            self.p_b = self.q_buyer[0][0]

            for i in range(q_b-q_s-1):
                hq.heappush(self.q_buyer, tmp_lis[i])

    def solve(self):
        '''
        Calculate optimal SW for reference
        '''
        res0 = self.Optimal()

        self.out_seller = deque()
        self.out_buyer = deque()

        '''
        Stage 1: Do McAfee's Trade Reduction Mechanism
        '''
    
        res1 = self.MTRForInit()
        if self.p_s == -1 and self.p_b == -1: # If no trade in MTR
            res_s = 1e18 # INF
            res_b = -1
            for u in range(self.s):
                if invited[u]:
                    res_s = min(res_s, self.val[u])
            for u in range(self.s, self.n):
                if invited[u]:
                    res_b = max(res_b, self.val[u])

            self.p_s = min(res_s, res_b)
            self.p_b = max(res_s, res_b)

        '''
        Stage 2: Invite people, do TRP
        '''

        flag = 1
        while flag:
            flag = 0
            # Inviting
            while len(self.out_seller) != 0:
                x = self.out_seller.popleft()
                for v in range(self.n):
                    if self.edges[x[1]][v] and not self.invited[v]:
                        flag = 1 # mark that it is not empty
                        self.invited[v] = True # mark v as invited
                        if v < self.s:
                            hq.heappush(self.q_seller, (-self.val[v], v)) # max heap
                        else:
                            hq.heappush(self.q_buyer, (self.val[v], v)) # min heap


            while len(self.out_buyer) != 0:
                x = self.out_buyer.popleft()
                for v in range(self.n):
                    if self.edges[x[1]][v] and not self.invited[v]:
                        flag = 1 # mark that it is not empty
                        self.invited[v] = True
                        if v < self.s:
                            hq.heappush(self.q_seller, (-self.val[v], v)) # max heap
                        else:
                            hq.heappush(self.q_buyer, (self.val[v], v)) # min heap

            # do TRP
            self.TRP()

        # Summarize the result

        self.align()
        res2 = self.review()

        return res0, res1, res2


def experiment(VARL, VARR, VARSTEP, LOOP, opt):
    N = 1000
    SL = N//2
    SR = N//2
    VL = 0
    VR = 10000
    REWIRING_PROB = 0.3
    CONNECT_PROB = 0.3
    INITIAL_PARTICIPANT_NUM = 300
    EXP_CNT = 3

    DA = DAs()

    x = []
    y = [[] for j in range(EXP_CNT)]
    VAR = VARL
    while VAR <= VARR:
        print(VAR)
        if opt == 1:
            CONNECT_PROB = VAR/1000
        elif opt == 2:
            INITIAL_PARTICIPANT_NUM = VAR
            
        res = [0 for j in range(EXP_CNT)]
        for i in range(LOOP):
            DA.gen(N, SL, SR, VL, VR, REWIRING_PROB, CONNECT_PROB, INITIAL_PARTICIPANT_NUM, opt)
            now = DA.solve()
            for j in range(EXP_CNT):
                res[j] += now[j]/now[0]
        x.append(VAR)
        for j in range(EXP_CNT):
            y[j].append(res[j]/LOOP)

        if int(VARSTEP) == VARSTEP:
            VAR += VARSTEP
        else:
            VAR = int(VAR*VARSTEP)

    '''
    labels = ["Optimal Social Welfare", "MTR for initial traders", "DTR Mechanism"]
    for j in range(EXP_CNT):
        plt.plot(x, y[j], label=labels[j])
    plt.legend()
    plt.show()
    '''

    with open("data.out","a") as f:
        f.write("opt={}\n".format(opt))
        for val in x:
            f.write("{} ".format(val))
        f.write('\n')
        for j in range(EXP_CNT):
            for val in y[j]:
                f.write("{} ".format(val))
            f.write('\n')
        f.write('\n')

# experiment(1, 100, 5, 1000, 1)
