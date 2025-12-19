import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
deltax = 80 / 512
import numpy as np

def compute_energy(lattice1, lattice2, lattice3):
    """
    Computes energy based on three small lattices.
    
    Parameters:
        lattice1, lattice2, lattice3 : array-like
            Input lattices (numpy arrays or lists)
    
    Returns:
        float : computed energy
    """
    # Convert to numpy arrays in case they are lists
    lattice1 = np.array(lattice1)
    lattice2 = np.array(lattice2)
    lattice3 = np.array(lattice3)
    
    # Compute sums
    sum1 = np.sum(lattice1)
    sum23 = np.sum(lattice2) + np.sum(lattice3)
    
    # Apply the condition
    if sum23 > 0.5 * sum1:
        energy = sum23+sum1
    else:
        energy = 0.0
    
    return energy

energyDW =0
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 25})
results = []
dataset_numbers = []
scale = np.loadtxt('average_scale_factor.txt')
# Open the snapshot file
with h5py.File("kinetic_energy_snapshot_scalar.h5", 'r') as f:
    kin = f['E_S_K_0']
    with h5py.File("gradient_energy_snapshot_scalar.h5", 'r') as f:
        grad = f['E_S_G_0']
        with h5py.File("potential_energy_snapshot.h5", 'r') as f:
            pot= f['E_V_0']
#The idea is to increase the energy ONLY of the points for which gradient plus potential are more than half of the total energy
#Then we should be able to relate it with the total area of the walls and see if it is coherent
    for dataset_name in kin:
        kinetic = kin[dataset_name]
        gradient = grad[dataset_name]
        potential = pot[dataset_name]
        print(f"Processing dataset: {dataset_name}")
        
        # Convert dataset name to float for plotting
        dataset_num = float(dataset_name)
        dataind= 10*int(dataset_num) 
        dataset_numbers.append(dataset_num)
        
        # Load data as numpy array
        kin_field = kinetic[()]
        grad_field = gradient[()]
        pot__field = potential[()]
        
        energyDW = compute_energy(kin_field, grad_field, pot__field)
        results.append(np.array([scale[dataind, 0],energyDW]))



results = np.array(results)

# Sort rows by the first column
results = results[results[:, 0].argsort()]

# Save results to a text file
np.savetxt("resultsZ2_energyDW.txt", results)
