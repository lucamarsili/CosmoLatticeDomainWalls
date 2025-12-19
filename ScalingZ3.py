import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
deltax =  21/ 448

mu = 0.15
lambda1= 0.225 
lambda2 = 0.4

beta =3*lambda2/(np.sqrt(8*lambda1))
vac = (1/np.sqrt(2))*(beta+ np.sqrt(beta**2+1)) # mu/np.sqrt(2 * lambda1)*
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
rc('font', **{'family': 'normal', 'weight': 'bold', 'size': 25})
results = []
dataset_numbers = []
scale = np.loadtxt('average_scale_factor.txt')
# Open the snapshot file
def roll_field(field, axis):
    return np.roll(field, -1, axis=axis)

def compute_forward_derivatives(field, deltax):
    # +x direction
    field_x = np.roll(field, -1, axis=0)
    forward_derivative_x = np.abs((field_x - field) / deltax)
    # +y direction
    field_y = np.roll(field, -1, axis=1)
    forward_derivative_y = np.abs((field_y - field) / deltax)
    # +z direction
    field_z = np.roll(field, -1, axis=2)
    forward_derivative_z = np.abs((field_z - field) / deltax)
    return forward_derivative_x, forward_derivative_y, forward_derivative_z

def closest_value_indices(field1, field2, values1, values2):
    # field1, field2: 3D arrays
    # values1, values2: lists or arrays of 3 numbers each
    field1 = np.asarray(field1)
    field2 = np.asarray(field2)
    values1 = np.asarray(values1).reshape((3, 1, 1, 1))
    values2 = np.asarray(values2).reshape((3, 1, 1, 1))
    # Compute squared distances for each value
    diffs = (field1 - values1) ** 2 + (field2 - values2) ** 2
    # Find index of minimum squared distance along the value axis (0-based)
    min_indices = np.argmin(diffs, axis=0)
    # Convert to 1-based indexing as before
    return min_indices + 1

def compare_sites_along_axis(arr, axis, value1, value2):
    """
    Compare two lattice sites along a given axis using roll.
    Returns a boolean array where True if a site is value1 and its neighbor along axis is value2.
    """
    arr = np.asarray(arr)
    rolled = np.roll(arr, -1, axis=axis)
    return (arr == value1) & (rolled == value2)



def AreaTypeDirection(deltax, crossing, fdx_h, fdy_h, fdz_h, fdx_a, fdy_a, fdz_a):
    A  = np.sum((deltax**2) * (np.sqrt(
            fdx_h[crossing]**2 + 
            fdy_h[crossing]**2 +
            fdz_h[crossing]**2 + 
            fdx_a[crossing]**2 + 
            fdy_a[crossing]**2 +
            fdz_a[crossing]**2
        )) / (np.sqrt(fdx_h[crossing]**2+fdx_a[crossing]**2 ) + np.sqrt(fdy_h[crossing]**2+fdy_a[crossing]**2 ) + np.sqrt(fdz_h[crossing]**2+fdz_a[crossing]**2 )))
    return A

with h5py.File('snapshot_scalar.h5', 'r') as f:
    snap_group_h = f['S_C_0']
    snap_group_a = f['S_C_1']
    
    
    for dataset_name in snap_group_h:
        dataset_h = snap_group_h[dataset_name]
        dataset_a = snap_group_a[dataset_name]
        print(f"Processing dataset: {dataset_name}")
        
        # Convert dataset name to float for plotting
        dataset_num = float(dataset_name)
        dataind= 5*int(dataset_num) #to adjust but it should be ok for this case 
        dataset_numbers.append(dataset_num)
        print(scale[dataind,0])
        # Load data as numpy array
        field_h = dataset_h[()]
        field_a = dataset_a[()]
        values_h = np.array([vac, -0.5*vac, -0.5*vac])
        values_a = np.array([0, 0.5*np.sqrt(3)*vac, -0.5*np.sqrt(3)*vac])
        Int = closest_value_indices(field_h,field_a, values_h, values_a)
        
        # +x direction
        crossingTypeI_x = compare_sites_along_axis(Int, 0, 1, 2) 
        crossingTypeII_x = compare_sites_along_axis(Int, 0, 2, 3) 
        crossingTypeIII_x = compare_sites_along_axis(Int, 0, 1, 3) 

        crossingTypeI_y = compare_sites_along_axis(Int, 1, 1, 2) 
        crossingTypeII_y = compare_sites_along_axis(Int, 1, 2, 3) 
        crossingTypeIII_y = compare_sites_along_axis(Int, 1, 1, 3)

        crossingTypeI_z = compare_sites_along_axis(Int, 2, 1, 2) 
        crossingTypeII_z = compare_sites_along_axis(Int, 2, 2, 3) 
        crossingTypeIII_z = compare_sites_along_axis(Int, 2, 1, 3)
    
        # Example usage:
        # result_array = closest_value_indices(field_h, [val1, val2, val3])
        
        
        # Roll fields for field_h
        #field_h_x = roll_field(field_h, 0)
        #field_h_y = roll_field(field_h, 1)
        #field_h_z = roll_field(field_h, 2)

        # Roll fields for field_a
        #field_a_y = roll_field(field_a, 1)
        #field_a_x = roll_field(field_a, 0)
        #field_a_z = roll_field(field_a, 2)
        # Compute derivatives for field_h
        fdx_h, fdy_h, fdz_h = compute_forward_derivatives(field_h, deltax)
        # Compute derivatives for field_a
        fdx_a, fdy_a, fdz_a = compute_forward_derivatives(field_a, deltax)

        AIx = AreaTypeDirection(deltax, crossingTypeI_x, fdx_h, fdy_h, fdz_h, fdx_a, fdy_a, fdz_a)
        AIIx = AreaTypeDirection(deltax, crossingTypeII_x, fdx_h, fdy_h, fdz_h, fdx_a, fdy_a, fdz_a)
        AIIIx = AreaTypeDirection(deltax, crossingTypeIII_x, fdx_h, fdy_h, fdz_h, fdx_a, fdy_a, fdz_a)
        
        AIy = AreaTypeDirection(deltax, crossingTypeI_y, fdx_h, fdy_h, fdz_h, fdx_a, fdy_a, fdz_a)
        AIIy = AreaTypeDirection(deltax, crossingTypeII_y, fdx_h, fdy_h, fdz_h, fdx_a, fdy_a, fdz_a)
        AIIIy = AreaTypeDirection(deltax, crossingTypeIII_y, fdx_h, fdy_h, fdz_h, fdx_a, fdy_a, fdz_a)
        
        AIz = AreaTypeDirection(deltax, crossingTypeI_z, fdx_h, fdy_h, fdz_h, fdx_a, fdy_a, fdz_a)
        AIIz = AreaTypeDirection(deltax, crossingTypeII_z, fdx_h, fdy_h, fdz_h, fdx_a, fdy_a, fdz_a)
        AIIIz = AreaTypeDirection(deltax, crossingTypeIII_z, fdx_h, fdy_h, fdz_h, fdx_a, fdy_a, fdz_a)
        
        
        #result = ((Ax + Ay + Az) / ((2*80 ** 3 ) ))/(scale[dataind,3])
        resultI = (scale[dataind,0])*((AIx + AIy + AIz) / (( 21** 3 ) ))
        resultII = (scale[dataind,0])*((AIIx + AIIy + AIIz) / (( 21** 3 ) ))
        resultIII = (scale[dataind,0])*((AIIIx + AIIIy + AIIIz) / (( 21** 3 ) ))
        #result =(scale[dataind, 0]) *((Ax + Ay + Az) / ((16 ** 3 ) ))/(scale[dataind,1])
        #result =((Ax + Ay + Az) / ((2*80 ** 3 ) ))/scale[dataind, 3]
        #print(f"Dataset {dataset_name}: result = {result}")
        
	    #print(i)
        results.append(np.array([scale[dataind, 0],resultI, resultII, resultIII]))

    # Save results to a text file
# Convert list of results into a numpy array
results = np.array(results)

# Sort rows by the first column
results = results[results[:, 0].argsort()]

# Save results to a text file
np.savetxt("resultsZ3_448_21_020925_10.txt", results)

