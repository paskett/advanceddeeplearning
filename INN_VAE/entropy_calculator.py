import numpy as np
from scipy.stats import entropy


def hist_from_samples(x, num_bins=100):
    """Returns a histogram with the specified number of bins from the array of
    samples, x"""
    x_density = np.histogram(x.ravel(), bins=num_bins)[0] + 1
    return x_density / np.sum(x_density)


def get_entropy_data(all_data,num_bins=100):
    """Obtains estimates of entropy data.
    
      Inputs:
          data_loader (iterable): loads the data, possibly in batches

      Outputs:
          relative_entropy (float): The relative entropy (or K-L divergence) 
                                    between the input and output distributions.
    """
    num_layers = len(all_data)
    names = []
    individual_entropies = np.zeros(num_layers)
    relative_entropies = np.zeros(num_layers-1)

    for idx,(name,data) in enumerate(all_data):
        names.append(name)

        if idx == 0: # For one iteration, there's no relative entropy to compute.
            output_hist = hist_from_samples(data,num_bins)
        else: # Switch input/output histograms & compute their relative entropy.
            input_hist, output_hist = output_hist, hist_from_samples(data,num_bins)
            relative_entropies[idx-1] = entropy(input_hist, output_hist)
        
        # Compute the individual entropy of the output histogram.
        individual_entropies[idx] = entropy(output_hist)
    
    return names, individual_entropies, relative_entropies
    
