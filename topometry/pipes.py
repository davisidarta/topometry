import matplotlib
matplotlib.use('Agg')  # plotting backend compatible with screen
import sys

filename = sys.argv[1]  # read filename from command line

def sc_standard(filename):
    """""""""
    Standard analysis usually employed for single-cell analysis.     
    
    """""""""
    import scanpy.api as sc

    adata = sc.read_10x_h5(filename)
    sc.pp.recipe_zheng17(adata)
    sc.pp.neighbors(adata)
    sc.tl.louvain(adata)
    sc.tl.paga(adata)
    sc.tl.umap(adata)
    sc.tl.rank_genes_groups(adata, 'louvain')
    adata.write('std_an_'+filename)

def sc_dbmap(filename):
    """""""""
    Standard analysis usually employed for single-cell analysis.     

    """""""""
    adata = sc.read_10x_h5(filename)
    sc.pp.recipe_zheng17(adata)
    sc.pp.neighbors(adata)
    sc.tl.louvain(adata)
    sc.tl.paga(adata)
    sc.tl.umap(adata)
    sc.tl.rank_genes_groups(adata, 'louvain')
    adata.write('./write/result.h5ad')
    # plotting
    sc.pl.paga(adata)
    sc.pl.umap(adata, color='louvain')
    sc.pl.rank_genes_groups(adata, save='.pdf')
