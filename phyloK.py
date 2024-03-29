# by Art FY Poon, 2012

from Bio import Phylo

from numpy import zeros
from numpy.linalg import cholesky
from numpy.random import lognormal as lognorm

import math, sys, random


class PhyloKernel:
    def __init__(self, 
                kmat=None, 
                rotate='ladder',
                rotate2='none',
                subtree=False,
                normalize='mean', 
                sigma=1, 
                gaussFactor=1, 
                withLengths=True, 
                decayFactor=0.1, 
                verbose=False, 
                resolve_poly=False):
        """
        requires a list of Phylo.Tree objects
        can cast iterator returned by Phylo.parse() as list
        """
        self.resolve_poly = resolve_poly
        self.normalize = normalize
        
        self.trees = []
        
        self.kmat = []
        self.is_kmat_computed = False
        
        # using **kwargs would probably make this cleaner
        self.rotate = rotate
        self.rotate2 = rotate2
        self.subtree = subtree
        self.normalize = normalize
        self.sigma = sigma
        self.gaussFactor = gaussFactor
        self.withLengths = withLengths
        self.decayFactor = decayFactor
        self.verbose = verbose
        self.resolve_poly = resolve_poly
        
        self.pcache = {}
        self.subtrees = {} # used for matching polytomies
        #self.cache_productions()
        
        self.sigma = sigma
        self.gaussFactor = gaussFactor
        self.decayFactor = decayFactor
        self.withLengths = withLengths
        
        self.verbose = verbose
        
        if self.verbose:
            print('creating PhyloKernel with settings')
            print('sigma = %f' % (self.sigma))
            print('gaussFactor = %f' % (self.gaussFactor))
            print('decayFactor = %f' % (self.decayFactor))
        
    @property
    def ntrees (self):
        return len(self.trees)

    def load_trees_from_file (self, handle):
        """
        Parse a file containing Newick tree strings
        """
        self.trees = []
        
        tree_iter = Phylo.parse(handle, 'newick')
        
        for t in tree_iter:
            if self.rotate=='ladder':
                t.ladderize()
            elif rotate=='random':
                scramble(t)
            else:
                pass
        
            if self.rotate2 == 'none':
                pass
            else:
                gravitate(t, subtree=subtree, mode=rotate2)
       
            if self.normalize != 'none': self.normalize_tree(t, mode=self.normalize)
            if self.resolve_poly:
                collapse_polytomies(t)
            self.trees.append(self.annotate_tree(t))
            
        self.kmat = zeros( (self.ntrees, self.ntrees) )
        self.is_kmat_computed = False
        self.delta_values = {}
    
    def normalize_tree (self, t, mode='median'):
        """
        Normalize branch lengths in tree by mean branch length.
        This helps us compare trees of different overall size.
        Ignore the root as its branch length is meaningless.
        """
        
        # compute number of branches in tree
        branches = t.get_nonterminals() + t.get_terminals()
        nbranches = len(branches) - 1
        
        if mode == 'mean':  
            tree_length = t.total_branch_length() - t.root.branch_length
            mean_branch_length = tree_length / nbranches
            
            for branch in branches[int(not t.rooted):]:
                branch.branch_length /= mean_branch_length
            
        elif mode == 'median':
            branch_lengths = [branch.branch_length for branch in branches[int(not t.rooted):]]
            branch_lengths.sort()
            
            if nbranches%2 == 0:
                median_branch_length = (branch_lengths[(nbranches/2)-1] + 
                                        branch_lengths[nbranches/2]) / 2.
            else:
                median_branch_length = branch_lengths[nbranches/2]
            
            for branch in branches[int(not t.rooted):]:
                branch.branch_length /= median_branch_length

    def annotate_tree(self, t):
        for tip in t.get_terminals():
            tip.production = 0
        for i, node in enumerate(t.get_nonterminals(order='postorder')):
            children = node.clades
            nterms = sum( [c.production == 0 for c in children] )
            node.production = nterms + 1
            node.index = i
            branch_lengths = [c.branch_length for c in node.clades]
            node.sqbl = sum([bl**2 for bl in branch_lengths])
        return t

    def compute_matrix(self):
        for i in range(self.ntrees):
            for j in range(i, self.ntrees):
                self.kmat[i,j] = self.kmat[j,i] = self.kernel(self.trees[i], self.trees[j])
                    
                if self.verbose:
                    print('%d\t%d\t%f' % (i, j, self.kmat[i,j]))
        
        self.is_kmat_computed = True

    def quick_kernel(self, t1, t2):
        """
        Compute kernel function for two trees without loading
        as member variables.
        """
        self.trees = []
        for t in [t1, t2]:
            if self.rotate=='ladder':
                t.ladderize()
            elif rotate=='random':
                scramble(t)
            else:
                pass
        
            if self.rotate2 == 'none':
                pass
            else:
                gravitate(t, subtree=subtree, mode=rotate2)
       
            if self.normalize != 'none': self.normalize_tree(t, mode=self.normalize)
            if self.resolve_poly:
                collapse_polytomies(t)
            self.trees.append(self.annotate_tree(t))
        
        return self.kernel(self.trees[0], self.trees[1])
        
    def kernel(self, t1, t2):
        """
        Recursive function for computing tree convolution
        kernel.  Adapted from Moschitti (2006) Making tree kernels
        practical for natural language learning. Proceedings of the 
        11th Conference of the European Chapter of the Association 
        for Computational Linguistics.
        """
        nodes1 = t1.get_nonterminals(order='postorder')
        nodes2 = t2.get_nonterminals(order='postorder')
        k = 0

        dp_matrix = [[0 for n2 in nodes2] for n1 in nodes1]
        
        # iterate over non-terminals
        for n1 in nodes1:
            for n2 in nodes2:
                if n1.production == n2.production:
                    bl1 = [c1.branch_length for c1 in n1.clades]
                    bl2 = [c2.branch_length for c2 in n2.clades]
                    res = self.decayFactor * math.exp( -1. / self.gaussFactor 
                            * (n1.sqbl + n2.sqbl - 2*sum([(bl1[i]*bl2[i]) for i in range(len(bl1)) ]) ))
                        
                    for cn1 in range(2):
                        c1 = n1.clades[cn1]
                        c2 = n2.clades[cn1]
                        
                        # Both terminal?
                        if c1.production == c2.production == 0:
                            res *= self.sigma + self.decayFactor
            
                        # Same number of children and tips?
                        elif c1.production == c2.production:
                            res *= self.sigma + dp_matrix[c1.index][c2.index]
            
                    dp_matrix[n1.index][n2.index] = res
                    k += res
        return k
