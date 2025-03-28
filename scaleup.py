from common import *
import networkx as nx
from logging import getLogger
from scipy import sparse

fix = lambda x: np.array(list(dict(x).values()))

class ScaleUp:
    def __init__(self, network, groups, known, know_prob=1.0, directed=False):
        """
        Initialize ScaleUp estimator.
        
        Args:
            network: Either a NetworkX graph, numpy adjacency matrix, or scipy sparse matrix
            groups: List of sets of nodes
            known: List of indices of known groups
            know_prob: Probability of knowing each group (float or list)
            directed: Whether the network is directed
        """
        # Handle different input types
        if isinstance(network, nx.Graph):
            self.A = nx.to_scipy_sparse_matrix(network)
        elif isinstance(network, (np.ndarray, np.matrix)):
            self.A = sparse.csr_matrix(network)
        elif isinstance(network, sparse.spmatrix):
            self.A = network
        else:
            raise TypeError("Network must be NetworkX graph, numpy array, or scipy sparse matrix")
            
        self.nodes = list(range(self.A.shape[0]))
        self.groups = [set(x) for x in groups]
        self.known = known

        if isinstance(know_prob, float):
            know_prob = [know_prob for _ in self.groups]

        # Pre-allocate arrays for better performance
        n_nodes = len(self.nodes)
        n_groups = len(self.groups)
        
        # Create boolean mask for edges - works directly on sparse matrix
        edge_mask = self.A > 0
        
        # Initialize egonet and egonetr as lists of sets for each group
        self.egonet = [defaultdict(set) for _ in range(n_groups)]
        self.egonetr = [defaultdict(set) for _ in range(n_groups)]
        
        # Vectorized random number generation
        rands = np.random.random((n_groups, n_nodes, n_nodes))
        
        # Process each group
        for gi in range(n_groups):
            # Get mask for this group's probability
            prob_mask = rands[gi] < know_prob[gi]
            
            # Get edges that meet both conditions
            valid_edges = edge_mask.multiply(sparse.csr_matrix(prob_mask))
            
            # Get sender and receiver indices using sparse matrix operations
            senders, receivers, _ = sparse.find(valid_edges)
            
            # Add to egonet and egonetr
            for s, r in zip(senders, receivers):
                self.egonet[gi][s].add(r)
                self.egonetr[gi][r].add(s)
            
            # If undirected, add reverse edges
            if isinstance(network, nx.Graph) or not directed:
                for s, r in zip(senders, receivers):
                    self.egonet[gi][r].add(s)
                    self.egonetr[gi][s].add(r)

        # Convert sets to lists for faster access
        self.senders = list(set().union(*[set(d.keys()) for d in self.egonet]))
        self.receivers = list(set().union(*[set(d.keys()) for d in self.egonetr]))

    def sample(self, N):
        """Sample N random nodes from the network."""
        return np.random.choice(self.senders, size=N, replace=False)

    def sample_and_identify(self):
        """
        Sample a random node and identify their friends in each group.
        
        Returns:
            List of lists containing friends in each group
        """
        # first choose a random node
        sampled = choice(self.senders)

        # now identify their num of friends in each group
        enets = [
            self.egonet[gi][sampled]
            for gi in range(len(self.groups))
        ]

        lists = [
            list(g & enets[gi])
            for gi, g in enumerate(self.groups)
        ]

        return lists

    def sample_and_count(self):
        """Sample and count friends in each group."""
        return [len(l) for l in self.sample_and_identify()]
    
    def classic_estimate(self, N=20):
        """
        Classic estimation method attributed to Killworth et al. (1998).
        """
        ls = []
        for _ in range(20):
            ls.append(self.sample_and_count())
        return np.array(ls)
        
    def groupc(self, gi=None):
        """
        Calculate the group coefficient of variation.
        
        Args:
            gi: Group index. If None, computes overall C.
            
        Returns:
            Coefficient of variation for the specified group
        """
        if gi is None:
            # Calculate degrees directly from sparse matrix
            deg = np.array(self.A.sum(axis=1)).flatten()
        else:
            deg = np.array([len(self.egonetr[gi][si]) for si in self.groups[gi]])
        M = deg.mean()
        V = deg.var()
        return 1 + V / M**2
    
    def expected_num(self, s, k, gi):
        """
        Calculate expected number of connections.
        
        Args:
            s: Number of samples
            k: Order of expectation
            gi: Group index
            
        Returns:
            Expected number of connections
        """
        n_edges = np.sum([len(self.egonetr[gi][ri]) for ri in self.groups[gi]])
        if k == 1:
            # mean of senders' degree
            M = n_edges / len(self.senders)
            return M*s
    
        # mean of receivers' degree
        deg = np.array([len(self.egonetr[gi][ri]) for ri in self.groups[gi]])
        M = deg.mean()
        V = deg.var()
        N = len(self.senders)
        
        return self.expected_num(s, k-1, gi) * (
            (s-1) / (N-1) * (M + V/M - 1)
        )

sample_logger = getLogger('sample')

class Sample:
    def __init__(self, su:ScaleUp, N:int):
        """
        Initialize a sample from the ScaleUp wrapper.
        
        Args:
            su: ScaleUp object
            N: Sample size
        """
        self.su = su
        self.N = N

        self.s = su.sample(N)
        sample_logger.debug(f'Sample: {self.s}')

        self.sn = np.array([
            [len(g & self.su.egonet[gi][si]) for si in self.s]
            for gi, g in enumerate(self.su.groups)
        ])

        for gi, g in enumerate(self.su.groups):
            sample_logger.debug(f'Group {gi}')
            sample_logger.debug(f'Group {gi} counts: {[x for x in self.sn[gi] if x > 0]}')
            sample_logger.debug(f'Group {gi} twice mentioned: {self.count_twice_mentioned(gi)}')

    def count_k_mentioned(self, gi, k, type='perm'):
        """
        Count nodes mentioned k times in a group.
        
        Args:
            gi: Group index
            k: Number of mentions
            type: Type of counting ('perm', 'comb', or 'count')
            
        Returns:
            Count of nodes mentioned k times
        """
        from math import perm, comb
        c = defaultdict(int)
        for si in self.s:
            for f in self.su.egonet[gi][si]:
                if f in self.su.groups[gi]:
                    c[f] += 1

        sample_logger.debug(f'Group {gi} count {k} mentioned: {[v for x,v in c.items()]}')

        if type == 'comb':
            return np.sum([comb(x, k) for x in c.values()])
        elif type == 'perm':
            return np.sum([perm(x, k) for x in c.values()])
        elif type == 'count':
            return np.sum([1 for x in c.values() if x == k])
        else:
            raise ValueError(f'Unknown type {type}')
    
    def count_twice_mentioned(self, gi, type='perm'):
        """Count nodes mentioned twice in a group."""
        return self.count_k_mentioned(gi, 2, type=type)

    def JointReferral(self, ri:int, ti:int, correction=None):
        """
        Joint-referral method for estimating network size.
        
        Args:
            ri: Reference group index
            ti: Target group index
            correction: Type of correction to apply ('true', 'sample', or None)
            
        Returns:
            Estimated network size
        """
        if correction == 'true':
            Ct = self.su.groupc(ti)
            Cr = self.su.groupc(ri)
            correction_val = Ct / Cr
        else:
            correction_val = 1

        Nr = len(self.su.groups[ri])
        uncorrected = Nr * self.JointReferral_mono(ti) / self.JointReferral_mono(ri)
        estimate = correction_val * uncorrected

        if correction == 'sample':
            E1t = np.sum(self.sn[ti, :])
            E1r = np.sum(self.sn[ri, :])
            E2t = self.count_twice_mentioned(ti)
            E2r = self.count_twice_mentioned(ri)
            f = self.N / len(self.su.nodes)

            Nt = estimate
            Nr = len(self.su.groups[ri])

            Ct = Nt*(E2t + f*E1t) / E1t**2
            Cr = Nr*(E2r + f*E1r) / E1r**2

            estimate *= Ct / Cr

        return estimate

    def JointReferral_mono(self, ti:int):
        """
        Monotonic version of the joint-referral method.
        
        Args:
            ti: Target group index
            
        Returns:
            Estimated network size
        """
        E1t = np.sum(self.sn[ti, :])
        E2t = self.count_twice_mentioned(ti)
        f = self.N / len(self.su.nodes)
        return E1t ** 2 / (E2t + f*E1t)
    
    def JointReferral_simplified(self, ri:int, ti:int):
        """
        Simplified version of the joint-referral method.
        
        Args:
            ri: Reference group index
            ti: Target group index
            
        Returns:
            Estimated network size
        """
        E1t = np.sum(self.sn[ti, :])
        E1r = np.sum(self.sn[ri, :])
        E2t = self.count_twice_mentioned(ti)
        E2r = self.count_twice_mentioned(ri)
        Nr = len(self.su.groups[ri])
        return Nr * (E1t/E1r)**2 * (E2r/E2t)
    
    def Katzir2012(self, gi):
        """
        Katzir et al. (2012) method for estimating network size.
        
        Args:
            gi: Group index
            
        Returns:
            Estimated network size
        """
        egonets = [self.su.egonet[gi][si] for si in self.s]
        all_degrees = np.array([len(self.su.egonetr[gi][ri]) for e in egonets for ri in e])
        R = all_degrees.sum() * (1/all_degrees).sum() - all_degrees.shape[0]
        C = self.count_twice_mentioned(gi, type='comb')
        return R/C
    
    def Chao1987(self, gi):
        """
        Chao (1987) estimator for network size.
        
        Args:
            gi: Group index
            
        Returns:
            Estimated network size
        """
        D = np.sum(self.sn[gi, :] > 0)  # total number of people mentioned
        D1 = np.sum(self.sn[gi, :] == 1)  # mentioned exactly once
        D2 = np.sum(self.sn[gi, :] == 2)  # mentioned exactly twice
        return D + D1**2 / (2*D2) if D2 > 0 else D
    

    def Lanumteang2011(self, gi):
        """
        Lanumteang et al. (2011) lower bound estimator.
        
        Args:
            gi: Group index
            
        Returns:
            Estimated network size
        """
        D = np.sum(self.sn[gi, :] > 0)  # total number of people mentioned
        D1 = np.sum(self.sn[gi, :] == 1)  # mentioned exactly once
        D2 = np.sum(self.sn[gi, :] == 2)  # mentioned exactly twice

        ratio = (3 * D1) / (2 * D2) if D2 > 0 else float('inf')

        if ratio <= 1:
            return D + (D1**2) / (2 * D2)
        elif 1 < ratio < 2:
            return D + (D1**2 / (2 * D2)) * ratio
        else:
            return D + (D1 / D2) * D1 if D2 > 0 else D

    def Khan2018(self, gi):
        """
        Khan et al. (2018) estimator.
        
        Args:
            gi: Group index
            
        Returns:
            Estimated network size or None if computation fails
        """
        g = set(self.su.groups[gi])
        egonets = [set(self.su.egonet[gi][si]) & g for si in self.s]
        n_in_sample = sum(j in self.s for enet in egonets for j in enet)
        n_total = sum(len(enet) for enet in egonets)

        if not n_in_sample:
            return None
        
        return len(set(self.s) & g) * n_total / n_in_sample
        
    def Khan2018_partial(self, gi):
        """
        Partial version of Khan et al. (2018) estimator.
        
        Args:
            gi: Group index
            
        Returns:
            Estimated network size or None if computation fails
        """
        g = set(self.su.groups[gi])
        egonets = [set(self.su.egonet[gi][si]) & g for si in self.s if si in g]
        n_in_sample = sum(j in self.s and j in g for enet in egonets for j in enet)
        n_total = sum(len(enet) for enet in egonets)

        if not n_in_sample:
            return None
        
        return len(set(self.s) & g) * n_total / n_in_sample

    def C_naive(self, gi):
        """
        Naive estimator of C = 1 + V/M^2.
        Just calculates this statistic for the sample.
        
        Args:
            gi: Group index
            
        Returns:
            Estimated coefficient of variation
        """
        deg = np.array([len(self.su.egonet[gi][si]) for si in self.s])
        M = deg.mean()
        V = deg.var()
        return 1 + V / M**2

    def C_LuLi2013(self, gi):
        """
        Lu and Li (2013) estimator of C = 1 + V/M^2.
        Assumes we have the degrees of the people we sample.
        
        Args:
            gi: Group index
            
        Returns:
            Estimated coefficient of variation
        """
        deg = np.array([len(self.su.egonetr[gi][si]) for si in self.s])
        S1 = deg.mean()
        S2 = (1/deg).mean()
        return S1 * S2

    def MLE_deg(self, known=None):
        """
        Equation (6) in Laga et al. 2021.
        This is the MLE of the degree of the people we sample.

        Args:
            known: List of known group indices
            
        Returns:
            Estimated degrees
        """
        if known is None:
            known = self.su.known

        N = len(self.su.nodes)
        known_counts = self.sn[known, :]
        known_Ns = np.array([len(self.su.groups[gi]) for gi in known])
        return N * known_counts.sum(axis=0) / known_Ns.sum()
        
    def PIMLE(self, known=None):
        """
        Plug-in maximum likelihood estimator.
        
        Args:
            known: List of known group indices
            
        Returns:
            Estimated network sizes
        """
        if known is None:
            known = self.su.known

        N = len(self.su.nodes)
        d = self.MLE_deg(known=known)
        ix = d > 0
        d = d[ix]
        sn = self.sn[:, ix]
        ratios = sn / d
        return N * np.mean(ratios, axis=1)
        
    def MLE(self, known=None):
        """
        Maximum likelihood estimator.
        
        Args:
            known: List of known group indices
            
        Returns:
            Estimated network sizes
        """
        if known is None:
            known = self.su.known

        N = len(self.su.nodes)
        known_counts = self.sn[known, :]
        known_Ns = np.array([len(self.su.groups[gi]) for gi in known])
        factor = known_Ns.sum() / known_counts.sum()
        return self.sn.sum(axis=1) * factor
    
    def MLE_stderr(self):
        """
        Standard error of the MLE.
        
        Returns:
            Standard error
        """
        N = len(self.su.nodes)
        d = self.MLE_deg()
        return np.sqrt(N * self.MLE_N() / d.sum())

    def MLE_eval(self, highlight=None):
        """
        Evaluate MLE performance with visualization.
        
        Args:
            highlight: List of indices to highlight
        """
        from matplotlib import pyplot as plt
        x,y = (
            self.MLE_N(),
            [len(x) for x in self.su.groups]
        )

        x += np.random.random(len(x))*10

        plt.errorbar(x, y,
            yerr=self.MLE_stderr(),
            linewidth=0,
            elinewidth=2,
        )

        plt.scatter(x,y)

        if highlight is not None:
            if type(highlight) != list: 
                highlight = [highlight]

            xh = np.array(x)[highlight]
            yh = np.array(y)[highlight]

            plt.scatter(xh,yh, color='green')

        plt.xlabel('MLE')
        plt.ylabel('True')

        m = min(plt.xlim()[1], plt.ylim()[1])
        plt.plot([0, m], [0, m], color='black', linestyle='--');


def method_comparison_plot(
        G, focal_group, reference_group, 
        N_samples=100, sample_size=500, 
        confidence=0.90, logy=True,
        methods = None,
        **scaleup_kwargs
):
    """
    Generate comparison plot of different estimation methods.
    
    Args:
        G: Network graph
        focal_group: Group to estimate
        reference_group: Reference group
        N_samples: Number of samples
        sample_size: Size of each sample
        confidence: Confidence interval
        logy: Whether to use log scale for y-axis
        methods: List of methods to compare
        **scaleup_kwargs: Additional arguments for ScaleUp
    """
    
    groups = [
        focal_group,
        reference_group
    ]

    if methods is None:
        methods = [
            'Killworth et al. (1998)',
            'Joint report correction',
            'JRC, using sample Ct/Cr',
            'JRC, without reference',
            'JRC, using true Ct/Cr'
        ]

    assert all(x in G.nodes for x in focal_group)
    assert all(x in G.nodes for x in reference_group)

    known = [1]

    su = ScaleUp(G, groups=groups, known=known, **scaleup_kwargs)

    print(f'Focal group size: {len(focal_group)}')

    method_map = {
        'MLE 3': lambda s: s.MLE_corrected()[0],
        'Killworth et al. (1998)': lambda s: s.MLE_N()[0],
        'MLE overall': lambda s: s.MLE_N_2()[0],
        'Joint report correction': lambda s: s.JointReferral(ri=1, ti=0, correction=None),
        'JRC, without reference': lambda s: s.JointReferral_mono(ti=0),
        'JRC, using sample Ct/Cr': lambda s: s.JointReferral(ri=1, ti=0, correction='sample'),
        'JRC, using true Ct/Cr': lambda s: s.JointReferral(ri=1, ti=0, correction='true')
    }

    vals = [
        np.array([method_map[m](s) for si in range(N_samples) for s in [Sample(su, sample_size)]])
        for mi, m in enumerate(methods)
        if (print('Processing', methods[mi]) or True)
    ]

    vals = [
        v[~np.isnan(v)]
        for v in vals
    ]

    means = [v.mean() for v in vals]

    bit = (1-confidence)/2

    yerr = [
        [ # lower error
            means[vi] - np.quantile(v, bit)
            for vi,v in enumerate(vals)
        ], [ # upper error
            np.quantile(v, 1-bit) - means[vi]
            for vi,v in enumerate(vals)
        ]
    ]

    plt.bar(methods, means, yerr=yerr, capsize=5, color='grey', edgecolor='black')
    plt.scatter(methods, means, color='black')

    # and a horizontal line for the true value
    true_val = len(su.groups[0])
    plt.hlines([true_val], *plt.xlim(), color='red', linestyle='--')

    # plot minimum and maximum values as open circles
    mins = [v.min() for v in vals]
    maxs = [v.max() for v in vals]

    plt.plot(methods, mins, 'o', color='black', markerfacecolor='white')
    plt.plot(methods, maxs, 'o', color='black', markerfacecolor='white')

    # log y axis
    if logy:
        plt.yscale('log')

    if logy:
        # maximum order of magnitude in means
        allvals = np.array(vals).flatten()
        allvals = allvals[allvals > 0]

        ytmax = np.ceil(np.log10(np.max(allvals)))
        ytmin = np.floor(np.log10(np.min(allvals)))

        if ytmax-ytmin == 1:
            yticks = np.arange(0, 11)
            yticks = (10**ytmin) * yticks
            yticklabels = [f'{int(yt):0,.0f}' for yt in yticks]

            mn, mx = yticks[0], yticks[-1]

        else:
            yticks = np.arange(ytmin, ytmax+1)
            yticklabels = [f'{int(10**yt):0,.0f}' for yt in yticks]
            yticks = 10**yticks

            mn, mx = np.min(allvals), np.max(allvals)
            unit = (mx-mn) / 20

            mn = max(1, mn - unit)
            mx = max(1, mx + unit)

        if true_val < mn:
            mn = true_val - unit

        if true_val > mx:
            mx = true_val + unit

        plt.yticks(yticks, yticklabels)
        plt.ylim(mn,mx)

    for n,m,v in zip(methods, means, vals):
        print(f'{n}: {m:0.2f} [{np.quantile(v, bit):0.2f}, {np.quantile(v, 1-bit):0.2f}]')