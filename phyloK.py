# by Art FY Poon, 2012

from Bio import Phylo

from numpy import zeros
from numpy.linalg import cholesky
from numpy.random import lognormal as lognorm

import math, sys, random


class PhyloKernel:
    def __init__(self, trees, 
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
        for t in trees:
            if rotate=='ladder':
                t.ladderize()
            elif rotate=='random':
                scramble(t)
            else:
                pass
            
            if rotate2 == 'none':
                pass
            else:
                gravitate(t, subtree=subtree, mode=rotate2)
            
            if self.normalize != 'none': self.normalize_tree(t, mode=self.normalize)
            if self.resolve_poly:
                collapse_polytomies(t)
            self.trees.append(t)
        
        self.ntrees = len(self.trees)
        
        # kernel matrix already computed
        if kmat is None:
            self.kmat = zeros( (self.ntrees, self.ntrees) )
            self.is_kmat_computed = False
        else:
            self.kmat = kmat
            self.is_kmat_computed = True
        
        self.pcache = {}
        self.subtrees = {} # used for matching polytomies
        #self.cache_productions()
        
        self.sigma = sigma
        self.gaussFactor = gaussFactor
        self.decayFactor = decayFactor
        self.withLengths = withLengths
        
        self.verbose = verbose
        
        if self.verbose:
            print 'creating PhyloKernel with settings'
            print 'sigma = %f' % self.sigma
            print 'gaussFactor = %f' % self.gaussFactor
            print 'decayFactor = %f' % self.decayFactor
        
        
    
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
    
    
        
    def compute_matrix(self):
        for i in range(self.ntrees):
            for j in range(i, self.ntrees):
                self.kmat[i,j] = self.kmat[j,i] = self.kernel(self.trees[i], self.trees[j])
                if self.verbose:
                    print '%d\t%d\t%f' % (i, j, self.kmat[i,j])
        
        self.is_kmat_computed = True
    
    
    def extend_matrix(self, trees, normalized=True):
        """
        Given a set of trees and a precomputed kernel matrix, take 
        another set of trees and calculate ONLY the kernel entries 
        for pairs that involve this new set.  New kernel matrix is
        returned, NOT set as class member variable.
        """
        if not self.is_kmat_computed:
            return None
        
        """
        self.normalize_matrix() # assumes that old kernel matrix is normalized
                                # if it is already normalized than this has no effect
        """
        
        more_trees = []
        for t in trees:
            t.ladderize()
            if self.normalize: self.normalize_tree(t, mode=self.normalize)
            more_trees.append(t)
        
        mtrees = len(more_trees)
        newmat = zeros( (self.ntrees+mtrees, self.ntrees+mtrees) )
        
        # transfer entries to new matrix
        for i in range(self.ntrees):
            for j in range(self.ntrees):
                newmat[i,j] = self.kmat[i,j]
        
        """
        # update productions cache
        if self.verbose: print 'updating productions cache'
        for t in more_trees:
            nodes = t.get_nonterminals()
            for node in nodes:
                children = node.clades
                nterms = sum( [c.is_terminal() for c in children] )
                self.pcache.update({node: (nterms, len(children)) })
        """
        
        # calculate diagonal entries - note that the original kernel matrix is normalized
        if normalized:
            if self.verbose: print 'recalculating non-normalized diagonal entries'
            
            diag = []
            for i in range(0, self.ntrees):
                k = self.kernel(self.trees[i], self.trees[i])
                if self.verbose: print '%d %d' % (i,k)
                diag.append(k)
                
            for i in range(0, mtrees):
                k = self.kernel(more_trees[i], more_trees[i])
                if self.verbose: print '%d %d' % (i+self.ntrees, k)
                diag.append(k)
                        
            # calculate new entries and normalize
            if self.verbose: print 'computing new entries'
            
            for i in range(0, self.ntrees):
                for j in range(0, mtrees):
                    denom = math.exp ( 0.5 * (math.log(diag[i]) + math.log(diag[self.ntrees+j])) )
                    newmat[self.ntrees+j,i] = newmat[i,self.ntrees+j] = self.kernel(self.trees[i], more_trees[j]) / denom
                    if self.verbose: print '%s,%s' % (i,j+self.ntrees)
                    
            for i in range(0, mtrees):
                newmat[self.ntrees+i, self.ntrees+i] = 1.
                for j in range(i+1, mtrees):
                    denom = math.exp ( 0.5 * (math.log(diag[self.ntrees+i]) + math.log(diag[self.ntrees+j])) )
                    newmat[self.ntrees+i, self.ntrees+j] = self.kernel(more_trees[i], more_trees[j]) / denom
                    newmat[self.ntrees+j, self.ntrees+i] = newmat[self.ntrees+i, self.ntrees+j]
                    if self.verbose: print '%s,%s' % (i+self.ntrees, j+self.ntrees)
        
        else: # original matrix is not normalized
            if self.verbose: print 'computing new entries without normalization'
            for i in range(0, self.ntrees):
                for j in range(0, mtrees):
                    newmat[self.ntrees+j,i] = newmat[i,self.ntrees+j] = self.kernel(self.trees[i], more_trees[j])
                    if self.verbose: print '%s,%s' % (i,j+self.ntrees)
                    
            for i in range(0, mtrees):
                for j in range(i, mtrees):
                    newmat[self.ntrees+i, self.ntrees+j] = self.kernel(more_trees[i], more_trees[j])
                    newmat[self.ntrees+j, self.ntrees+i] = newmat[self.ntrees+i, self.ntrees+j]
                    if self.verbose: print '%s,%s' % (i+self.ntrees, j+self.ntrees)
            
        return newmat
    
    
    def cache_productions (self):
        """
        For all trees, for all nodes, calculate production, which is
        a tuple of the number of terminal children and number of children.
        """
        
        for t in self.trees:
            nodes = t.get_nonterminals()
            for node in nodes:
                children = node.clades
                nterms = sum( [c.is_terminal() for c in children] )
                self.pcache.update({node: (nterms, len(children)) })
    
    
    def kernel(self, t1, t2):
        nodes1 = t1.get_nonterminals()
        nodes2 = t2.get_nonterminals()
        k = 0
        # iterate over non-terminals
        for n1 in nodes1:
            children = n1.clades
            nterms = sum( [c.is_terminal() for c in children] )
            p1 = (nterms, len(children))
            for n2 in nodes2:
                children = n2.clades
                nterms = sum( [c.is_terminal() for c in children] )
                p2 = (nterms, len(children))
                #if self.pcache[n1] == self.pcache[n2]:
                if p1 == p2:
                    k += self.delta(n1, n2)
        return k
    
    
    def partitions(self, n, maxlen):
        """
        Returns an iterator that generates all partitions of a positive
        integer.  From http://homepages.ed.ac.uk/jkellehe/partitions.php
        Restrict to permutations of length less than or equal to (maxlen).
        """
        a = [0 for i in range(n + 1)]
        k = 1
        a[0] = 0
        y = n - 1
        while k != 0:
            x = a[k - 1] + 1
            k -= 1
            while 2*x <= y:
                a[k] = x
                y -= x
                k += 1
            l = k + 1
            while x <= y:
                a[k] = x
                a[l] = y        
                res = a[:k + 2]
                if len(res) <= maxlen:
                    yield (res + [0 for i in range(maxlen-len(res))])
                x += 1
                y -= 1
            
            a[k] = x + y
            y = x + y - 1
            res = a[:k + 1]
            if len(res) <= maxlen:
                yield (res + [ 0 for i in range(maxlen-len(res)) ] )
        
    
    def k_permuts (self, n, maxlen):
        """
        Return an iterator over all permutations of partitions of 
        positive integer N of max length.
        """
        res = []
        for partition in partitions(n, maxlen):
            permuts = itertools.permutations(partition)
            for permut in permuts:
                res.append(permut)
        
        res = set(res)
        return list(res)
    
    
    def count_subtrees (self, root, node_count):
        """
        How many subtrees starting from root of subtree are there that
        have N tips and D nodes?  Track results in dictionary {res} with 
        path as key and number of tips as value.
        """
        if node_count == 1:
            return int(root.is_terminal())
        
        n_clades = len(root.clades)
        res = []
        
        for kp in self.k_permuts(node_count, n_clades):
            for i, clade in enumerate(root.clades):
                res.append(self.count_subtrees(clade, kp[i]))
        
        return res
    
    
    def delta(self, n1, n2):
        """
        Recursive function for computing tree convolution
        kernel.  Adapted from Moschitti (2006) Making tree kernels
        practical for natural language learning. Proceedings of the 
        11th Conference of the European Chapter of the Association 
        for Computational Linguistics.
        
        delta() should never be called on a mixed terminal/non-
        terminal pair because productions of the parent are equal 
        and trees have been ladderized.
        """
        if n1.is_terminal() and n2.is_terminal():
            return self.decayFactor
            
        #if n1.is_terminal() or n2.is_terminal():
        #   return 0
        
        # count productions
        children = n1.clades
        nterms = sum( [c.is_terminal() for c in children] )
        p1 = (nterms, len(children))
        
        children = n2.clades
        nterms = sum( [c.is_terminal() for c in children] )
        p2 = (nterms, len(children))
        
        
        #if self.pcache[n1] == self.pcache[n2]:
        if p1 == p2:
            # calculate decay factor
            if self.withLengths:
                bl1 = [c1.branch_length for c1 in n1.clades]
                bl2 = [c2.branch_length for c2 in n2.clades]
                res = self.decayFactor * math.exp( -1. / self.gaussFactor 
                        * sum([ (bl1[i] - bl2[i])**2 for i in range(len(bl1)) ]) )
            else:
                res = self.decayFactor
            
            for cn1 in range(len(n1.clades)):
                res *= self.sigma + self.delta (n1.clades[cn1], n2.clades[cn1])
            
            return res
        
        return 0
        
    
    def is_positive (self, kmat=None):
        """
        Check that the kernel matrix is positive definite by
        Cholesky decomposition.
        """
        if kmat is None:
            try:
                res = cholesky(self.kmat)
            except:
                return False
            return True
        else:
            try:
                res = cholesky(kmat)
            except:
                return False
            return True
    
    
    def matrix_to_file (self, handle):
        """
        Write matrix to open file passed as argument in
        CSV format.
        """
        if self.is_kmat_computed:
            for row in self.kmat:
                handle.write(','.join([str(x) for x in row.tolist()])+'\n')
    
    def normalize_matrix (self):
        """
        From Collins and Duffy (2001):
        "First, the value of K(T1,T2) will depend greatly on the size of the tree. 
        One may normalize the kernel by using
            K'(T1,T2) = K(T1,T2) / sqrt(K(T1,T1) K(T2,T2))
        which also satisfies the essential Mercer conditions."
        """
        # apply to member object
        if self.is_kmat_computed:
            nmat = zeros( (self.ntrees, self.ntrees) )
            for i in range(self.ntrees):
                nmat[i][i] = 1.
                for j in range(i+1, self.ntrees):
                    denom = math.exp(0.5 * (math.log(self.kmat[i][i]) + math.log(self.kmat[j][j])))
                    nmat[j][i] = nmat[i][j] = self.kmat[i][j] / denom
            
            # does this need to be deep copy?
            self.kmat = nmat
        


# ================================================== #
def normalize_matrix(kmat):
    ntrees = len(kmat)
    nmat = zeros( (ntrees, ntrees) )
    for i in range(ntrees):
        nmat[i][i] = 1.
        for j in range(i+1, ntrees):
            denom = math.exp(0.5 * (math.log(kmat[i][i]) + math.log(kmat[j][j])))
            nmat[j][i] = nmat[i][j] = kmat[i][j] / denom
    return nmat


def scramble (t):
    """
    Instead of ladderizing the tree, randomize children in place.
    """
    nodes = t.get_nonterminals()
    for node in nodes:
        random.shuffle(node.clades)


def gravitate (t, mode='all', subtree=False, reverse=False):
    """
    Rotate branches based on their lengths, or the total length of 
    descendant subtrees.  Performed in-place, no return value.
    """
    nodes = t.get_nonterminals()
    for node in nodes:
        is_cherry = all([clade.is_terminal() for clade in node.clades])
        is_tie = (len(set([len(clade.get_terminals()) for clade in node.clades])) == 1)
        # every cherry is a tie, but not every tie is a cherry
        if (mode=='ties' and not is_tie): continue
        if (mode=='cherries' and not is_cherry): continue
        
        if (subtree):
            weights = [(clade.total_branch_length, index) for index, clade in enumerate(node.clades)]
        else:
            weights = [(clade.branch_length, index) for index, clade in enumerate(node.clades)]
        
        weights.sort(reverse=reverse) # lengthier branches last by default
        
        node.clades = [node.clades[index] for bl, index in weights]


def mean_path_length (t):
    tips = t.get_terminals()
    total = 0.
    for tip in tips:
        path = len(t.root.get_path(tip))
        total += path
    return total / len(tips)


def var_path_length(t):
    tips = t.get_terminals()
    total = 0.
    for tip in tips:
        path = len(t.root.get_path(tip))
        total += path**2
    
    # E[X^2] - E[X]^2
    return total / len(tips) - mean_path_length(t)


def colless (t, normalize=True):
    """
    Compute Colless' index for tree, the sum over all internal nodes
    of the absolute difference between the number of tips descending
    from its left and right children.  Note this requires a rooted tree.
    For an unrooted tree, the root is essentially a polytomy and we
    ignore it.  Normalization from Kirkpatrick and Slatkin (1993).
    """
    
    nodes = t.get_nonterminals()
    
    # is this a rooted tree?
    if len(t.root.clades) == 2:
        # yes
        pass
    else:
        # no
        nodes.remove(t.root)
    
    # iterate over internal nodes
    cindex = 0
    for node in nodes:
        cindex += abs( len(node.clades[0].get_terminals()) - len(node.clades[1].get_terminals()) )
    
    if normalize:
        # compute Colless' index for pectinate n-tree
        # this is simply the triangular number
        ntips = len(t.get_terminals())
        if ntips == 2:
            return 0
        else:
            cindex *= 2. / (ntips * (ntips-3) + 2)
    
    return cindex


def sackin (t, normalize=True):
    """
    Compute Sackin index, the sum of the depths of leaves.
    If normalized, use coalescent-based expectation as per Kirkpatrick
    and Slatkin (1993), although this is not necessarily the 
    appropriate null distribution.
    """
    
    ntips = len(t.get_terminals())
    depths = t.depths(unit_branch_lengths=True)
    res = sum(depths.values())
    if normalize:
        # expectation under the Yule model
        expect = 2 * sum([1./j for j in range(2,ntips+1)])
        return (res - expect) / ntips
    
    return res



def shao_sokal_b1 (t, normalize=True):
    """
    Compute Shao and Sokal's (1990) B1 statistic, which is the 
    sum over all internal nodes of 1./max(N_i), where N_i is the
    number of internal nodes from tip to root, which is included
    in the count.
    Normalization according to Shao and Sokal (1990) Syst Zool 39:
    B1(norm) = (B1-min(B1)) / (max(B1)-min(B1))
    """
    nodes = t.get_nonterminals()
    res = 0
    
    for node in nodes:
        tips = node.get_terminals()
        m = 0
        for tip in tips:
            path = node.get_path(tip)
            if len(path) > m:
                m = len(path)
        
        if m > 0:
            res += 1./m
    
    if normalize:
        max_b1 = sum( [ 1./i for i in range(1, len(nodes))] ) # pectinate tree
        """
        # in development
        n = len(t.get_terminals())
        max_depth = int(math.ceil(math.log(n,2)))
        depths = []
        for k in range(0, max_depth):
            left = min(2**k, (2**max_depth-1))
            for times in range(left):
                depths.append(max_depth-k)
        """
        min_b1 = shao_sokal_b1(bifurcate_tree( len(t.get_terminals()) ), False)
        
        res = (res - min_b1) / (max_b1 - min_b1)
        
    return res



def shao_sokal_b2 (t):
    """
    B2 = Sum(N_i/(2^N_i), 1, n)
    where N_i is the number of internal nodes between the i-th
    tip and the root
    """
    
    tips = t.get_terminals()
    total = 0.
    for tip in tips:
        ni = len(t.root.get_path(tip))
        total += ni / (2.**ni)
    return total


def fusco_cronk (t):
    nodes = t.get_nonterminals()
    res = []
    for node in nodes:
        # total number of descendant tips
        n = len(node.get_terminals())
        # size of larger subtree
        b = max([len(clade.get_terminals()) for clade in node.clades])
        
        # maximum possible value for B
        big_m = n-1
        # minimum possible value for B
        little_m = n/2
        try:
            res.append( float(b-little_m) / (big_m - little_m))
        except:
            # not possible to calculate
            pass
    
    mean = sum(res)/len(res)
    mean10 = sum(res[:10])/10.
    return sum(res), mean, mean10


    

def pectinate_tree (t, with_length = None, log_sd = None):
    """
    Generate a pectinate tree with the same number of tips as
    the given tree or integer argument.  
    Optionally randomize branch lengths, otherwise return constant 
    branch lengths scaled so that tree lengths are same.
    If randomizing and (t) is Integer, draw from log-normal distribution.
    """
    if type(t) is int:
        ntips = t
        nbranches = 2*ntips - 2
        
        if with_length:
            mean_length = with_length
        else:
            mean_length = 1.0
        
        if log_sd:
            branch_lengths = [lognorm(math.log(mean_length), log_sd) for i in range(nbranches)]
        else:
            branch_lengths = [mean_length for i in range(nbranches)]
        
    elif type(t) is Phylo.Newick.Tree:
        ntips = len(t.get_terminals())
        nbranches = 2*ntips - 2
        
        if with_length:
            # re-use branch lengths
            branch_lengths = [b.branch_length for b in t.get_terminals()] + [b.branch_length for b in t.get_nonterminals()]
        else:
            mean_length = t.total_branch_length() / (ntips + len(t.get_nonterminals()))
            if log_sd:
                branch_lengths = [lognorm(math.log(mean_length), log_sd) for i in range(nbranches)]
            else:
                branch_lengths = [mean_length for i in range(nbranches)]
    else:
        raise TypeError
    
    random.shuffle(branch_lengths) # random permutation in-place
    
    pt = Phylo.BaseTree.Tree()
    curnode = pt.root
    for i in range(ntips-1):
        curnode.split()
        for child in curnode.clades:
            child.branch_length = branch_lengths.pop()
        curnode = curnode.clades[0]
    
    #pt.ladderize() # this is causing a max recursion depth error
    return pt


def bifurcate_tree (t, with_length = None, log_sd = None):
    """
    Generate a bifurcating (balanced) tree.  This will only be a perfectly
    balanced tree if the requested number of tips is a power of 2.
    """
    if type(t) is int:
        ntips = t
        nbranches = 2*ntips - 2
        
        if with_length:
            mean_length = with_length
        else:
            mean_length = 1.0
        
        if log_sd:
            branch_lengths = [lognorm(math.log(mean_length), log_sd) for i in range(nbranches)]
        else:
            branch_lengths = [mean_length for i in range(nbranches)]
        
    elif type(t) is Phylo.Newick.Tree:
        ntips = len(t.get_terminals())
        nbranches = 2*ntips - 2
        
        if with_length:
            # re-use branch lengths
            branch_lengths = [b.branch_length for b in t.get_terminals()] + [b.branch_length for b in t.get_nonterminals()]
        else:
            mean_length = t.total_branch_length() / (ntips + len(t.get_nonterminals()))
            if log_sd:
                branch_lengths = [lognorm(math.log(mean_length), log_sd) for i in range(nbranches)]
            else:
                branch_lengths = [mean_length for i in range(nbranches)]
    else:
        raise TypeError
    
    random.shuffle(branch_lengths) # random permutation in-place
    
    bt = Phylo.BaseTree.Tree()
    curnodes = [bt.root]
    while 1:
        nextnodes = []
        exit = False
        for node in curnodes:
            node.split()
            nextnodes.extend(node.clades)
            if len(bt.get_terminals()) >= ntips:
                exit = True
                break
        if exit:
            break
        curnodes = nextnodes
    
    return bt



def collapse_polytomies (t, cutoff=1e-5):
    """
    Assume that branches with minimum branch length (that
    is below some threshold) are polytomies.  Collapse these 
    nodes to form proper polytomies.  Tree will no longer be a 
    binary tree.  Return value = number of collapsed nodes.
    """
    nodes = t.get_nonterminals()
    root = nodes.pop(0)
    branch_lengths = [node.branch_length for node in nodes]
    min_branch_length = min(branch_lengths)
    if min_branch_length > cutoff:
        return 0
    
    to_collapse = []
    for node in nodes:
        if node.branch_length == min_branch_length:
            to_collapse.append(node)
    
    for node in to_collapse:
        t.collapse(node)
    
    return len(to_collapse)
    


def prune_polytomies (t):
    """
    Remove branches from polytomies until only two remain.
    Remove in proportion to the number of tips descending from
    each branch.  Return True on success.
    """
    if t.is_bifurcating():
        return False
    
    nodes = t.get_nonterminals(order='postorder') # children before parents
    root = nodes.pop(0) # ignore the root
    
    for node in nodes:
        if len(node.clades) > 2:
            # count the number of tips per clade
            ntips = []
            for i, clade in enumerate(node.clades):
                for j in range(len(clade.get_terminals())):
                    ntips.append(i)
            
            # iteratively remove one 'tip' until the cardinality reaches 2
            while len(set(ntips)) > 2:
                remove = ntips.pop(random.randint(0, len(ntips)-1))
            
            # prune clades that are not in this set
            for i, clade in enumerate(node.clades):
                if not i in ntips:
                    tips = clade.get_terminals()
                    for tip in tips:
                        t.prune(tip)
    
    return True



def coalesce_nodes (parent, n1, n2, blen = 1E-6):
    """
    Insert common ancestor for given nodes.  New node will 
    be assigned previous parent.  Total branch length is conserved.
    Returns the new node.
    """
    clade_cls = type(parent)
    # create new ancestor
    new_node = clade_cls(name=None, branch_length = blen)
    new_node.clades.append(n1)
    new_node.clades.append(n2)
    # update original parent
    parent.clades.append(new_node)
    parent.clades.remove(n1)
    parent.clades.remove(n2)
    # rescale branch lengths
    n1.branch_length -= blen
    n2.branch_length -= blen

    

def randomize_polytomies(t):
    """
    Insert random bifurcating subset tree at all polytomies in given tree.
    """
    nodes = t.get_nonterminals()
    for node in nodes:
        while 1:
            clades = node.clades
            if len(clades) > 2:
                # randomly coalesce child nodes
                n1, n2 = random.sample(clades, 2)
                coalesce_nodes(node, n1, n2)
            else:
                break


def run_example ():
    if len(sys.argv) < 2:
        print 'need to provide an output path'
        print 'python phyloK.py [path]'
        sys.exit()
    
    print 'running example...'
    trees = Phylo.parse('Shankarappa.P9.nwk', 'newick')
    pk = PhyloKernel(trees, normalize=True, withLengths=True, sigma=1, decayFactor=0.1)
    kmat = pk.compute_matrix()
    outfile = open(sys.argv[1], 'w')
    pk.matrix_to_file(outfile)
    outfile.close()
    if pk.is_positive():
        print 'matrix is positive definite'
    else:
        print 'matrix is not positive definite!'


if __name__ == "__main__":
    run_example()

