import sys, time

if sys.version < '2.7':
	print 'Requires Python >= 2.7'
	sys.exit()


from Bio import Phylo
import phyloK
import argparse


parser = argparse.ArgumentParser (description = 'Generate a kernel matrix for phylogenetic trees')

parser.add_argument('infile', help='Input file containing trees.')
parser.add_argument('outfile', help='Path to write kernel matrix in CSV format.')

parser.add_argument('--scale', help='Turns off default behavior of rescaling branch lengths in trees.', choices=['mean', 'median', 'none'], default='mean')
parser.add_argument('--sst', help='Use original subset tree kernel without penalizing by branch lengths.', action='store_true')

parser.add_argument('--rotate', help='Rotate branches to a particular configuration.  [ladder] ladderize only; [none] do no rotations; [random] randomize rotations', choices=['ladder', 'none', 'random'], default='ladder')
parser.add_argument('--rotate2', help='Further rotation of branches based on lengths.', choices=['none', 'all', 'ties', 'cherries'], default='none')
parser.add_argument('--subtree', help='rotate2 performed with respect to subtrees, otherwise adjacent branches.', action='store_true')

parser.add_argument('-s', '--sigma', help='0 produces subtree kernel, 1 produces subset tree kernel (default).', type=float, default=1)
parser.add_argument('-d', '--decay', help='Decay factor to suppress large diagonal in kernel matrix.', type=float, default=0.1)
parser.add_argument('-g', '--gauss', help='Variance parameter of Gaussian penalty function on branch lengths.  Lower values enforce greater penalty.', type=float, default=1)
parser.add_argument('-v', '--verbose', help='Print additional messages to console.', action='store_true')


args = parser.parse_args()

outfile = open(args.outfile, 'w')

"""
notes

if we run -noscale with default gauss factor, it is pretty similar to
running with no branch lengths - i.e., there is limited penalization by
discordant branch lengths.  We end up with a huge diagonal.

"""

print 'reading in trees from %s' % args.infile

t0 = time.time()

trees = Phylo.parse(args.infile, 'newick')

pk = phyloK.PhyloKernel(trees, 
						rotate = args.rotate,
						rotate2 = args.rotate2,
						subtree = args.subtree,
						normalize = args.scale, 
						withLengths = not args.sst,
						sigma = args.sigma,
						decayFactor = args.decay,
						gaussFactor = args.gauss,
						verbose = args.verbose)

print 'computing matrix'
pk.compute_matrix()

if pk.is_positive():
	print 'matrix is positive definite'
else:
	print 'matrix is not positive definite!'


print 'writing out to file %s' % args.outfile
pk.matrix_to_file(outfile)
outfile.close()


pk.normalize_matrix() # this is important!

outfile = open(args.outfile+'.norm', 'w')
pk.matrix_to_file(outfile)
outfile.close()

print 'total processing time %1.2f seconds' % (time.time() - t0)

