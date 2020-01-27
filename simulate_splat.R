library(splatter)
library(RcppCNPy)

setwd("D:/seqRNA")



simulate_batch = function(sim_id, n_sim=32, n_genes=20, n_cells=20, which='train', n_params=8) {
  # Simulates a batch of gene-cell matrices and save it to file
  
  if (which=='test') set.seed(42)
  else if (which == 'train') set.seed(43)
  else if (which == 'sbc') set.seed(44)
    
  sim_data = array(0, c(n_sim, n_genes, n_cells))
  sim_params = array(0., c(n_sim, n_params))
  
  # Sample parameters 
  mu = runif(n_sim, 1.0, 2.0)
  rate = runif(n_sim, 0.1, 5.0)
  out_loc = runif(n_sim, 0.5, 5)
  out_scale = runif(n_sim, 0.1, 2.0)
  lib_loc = runif(n_sim, 6, 10)
  lib_scale = runif(n_sim, 1.0, 2.0)
  out_prob = runif(n_sim, 0.01, 0.5)
  bcv_common = runif(n_sim, 0.1, 0.9)
  
  for (i in 1:n_sim) {
    
    
    # Generate a parameter instance
    params = newSplatParams(mean.shape=mu[i], 
                            mean.rate=rate[i], 
                            out.facLoc=out_loc[i], 
                            out.facScale=out_scale[i], 
                            lib.loc=lib_loc[i], 
                            lib.scale=lib_scale[i], 
                            out.prob=out_prob[i],
                            bcv.common=bcv_common[i])
    
    # Simulate and store
    sim = splatSimulate(params, nGenes = n_genes, batchCells=n_cells, verbose=F)
    sim_data[i, , ] = matrix(counts(sim), nrow=n_genes, ncol=n_cells)
    sim_params[i, ] = c(mu[i], rate[i], out_loc[i], out_prob[i], 
                        out_scale[i], lib_loc[i], lib_scale[i], bcv_common[i])
  }
  
  
  
  if (which == 'test') {
    npySave("test/rna_data_test.npy", sim_data)
    npySave("test/rna_params_test.npy", sim_params)
  } else if (which == 'train') {
    npySave(paste0("data/rna_data_", sim_id + offset, ".npy"), sim_data)
    npySave(paste0("params/rna_params_", sim_id + offset, ".npy"), sim_params)
  } else {
    npySave(paste0("sbc/rna_data_", sim_id + offset, ".npy"), sim_data)
    npySave(paste0("sbc/rna_params_", sim_id + offset, ".npy"), sim_params)
  }
  
  
}

### Simulation specs
n_genes=100 
n_cells=40
n_batches = 30
n_sim_batch = 10000
n_test = 500
n_sbc = 5000


### Run simulation (For training)
for (i in 1:n_batches) {
  
  simulate_batch(i, n_sim=n_sim_batch, n_genes=n_genes, which='train', n_cells=n_cells)
  print(paste('Simulated', i,  'batches...'))

  
}

### Run simulation (for validation)
simulate_batch(i+1, n_sim=n_test, n_genes=n_genes, which='test', n_cells=n_cells)


### Run simulation (for SBC)
simulate_batch(i+1, n_sim=n_sbc, n_genes=n_genes, which='sbc', n_cells=n_cells)
