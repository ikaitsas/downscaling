# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:31:26 2025

@author: yiann

For slope calculations, we need to caclulcate a new scaling factor for
each horizontal stripe, to minimize the top-bot strip calclation errors,
because the current approach takes the scaling factor of the center 
latitude. one approach that i have to implement is cutting the DEM into
horizontal stripes, and calculating slope in each one of them, using
the corresponding latitude-dependent scaling factor

this will probably be implemented in a separate slope computation script
i will also make the DEM and aspect extraction separate scripts
this one is a bit too large...

also, computing slope and aspect from aggregated DEM for large 
resolutions produces bullshit results
just run the script for 0.1deg resolution....
it might be best to aggregate the 1arcsec native resolution slope 
and aspect, as per this paper dictates:
https://www.sciencedirect.com/science/article/pii/S0098300415000254
maybe calculate the 0.01 deg slope and aspect, but aggregate those 
for the derivation of even coarser resolution morphometric parameters

aspect noData values are another problem...
gdal standard behaviour is assigning -9999 to flat terrain, be it bodies
of water of flat plains. but if i export the tif file as array, then only
large water bodies are "dark" in color, not flat plains. if i set -9999 to
NaNs though,both water bodies and flat terrain are blanked out. WHY??
this is not a problem if i set -9999 values to a smaller negative value, 
e.g. -1/-10 also, only if i mask negaative values as NaNs and plot...

"""

import subprocess
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib import colormaps

input_dem = "coco.tif"

#input_dem = 'coco.tif'
#input_dem = 'coco0.01deg.tif'
#input_dem = 'coco0.1deg.tif'

export_to_device = False
plot_bands = False
make_corrections = True  #slight filtering of data
save_memory = True  #"delete" some datasets - might help with memory?
merge_outputs = True  #merge DEM-Slope-Aspect for exportation
scale_factor = 111120  #meters


#needed for rounding lats/lons
def count_decimal_places(number):
    """Counts the number of decimal places in a given number."""
    str_num = str(number).rstrip('0')  # Remove trailing zeros
    if '.' in str_num:
        return len(str_num.split('.')[1])
    return 0


gdal.UseExceptions()



#%% Import dataset & extract DEM
print('Importing Dataset...\n')
dataset = gdal.Open(input_dem, gdal.GA_ReadOnly)
arrayDEM = dataset.GetRasterBand(1).ReadAsArray()

print('Extracting Metadata...\n')
width = dataset.RasterXSize
height = dataset.RasterYSize
geotransform = dataset.GetGeoTransform()

lat = np.arange(geotransform[3], 
                geotransform[3]+height*geotransform[5], 
                geotransform[5])
latRounded = np.round(lat, decimals=count_decimal_places(geotransform[1]))

lon = np.arange(geotransform[0], 
                geotransform[0]+width*geotransform[1], 
                geotransform[1]) #it produces width+1!!
lonRounded = np.round(lon, decimals=count_decimal_places(geotransform[1]))
lonMap,latMap = np.meshgrid(lonRounded, latRounded)


minGDAL, maxGDAL = dataset.GetRasterBand(1).ComputeRasterMinMax(False)
minArray, maxArray = arrayDEM.min(), arrayDEM.max()

if make_corrections == True:
    # -everything lower than -6m for Greece doesnt make sense
    arrayDEM[arrayDEM<-6] = -6


print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                            dataset.GetDriver().LongName))
print('\n')
print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                    dataset.RasterYSize,
                                    dataset.RasterCount))
print('\n')
print("Projection is\n{}".format(dataset.GetProjection()))
print('\n')
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
print('\n')
print("Band Type={}".format(gdal.GetDataTypeName(
    dataset.GetRasterBand(1).DataType)))
print('\n')
print("Min={:.3f}, Max={:.3f}".format(minGDAL,maxGDAL))


print('Exporting DEM as numpy array...')
output_dem = '-'
if export_to_device == True:
    if geotransform[1]<0.003:  #1arcsec Native
        output_dem = f'dem_{input_dem[:-4]}_1arcsecNative'
        np.save(f'{output_dem}.npy', arrayDEM)
        print('\n')
        print(f'Exported DEM as: {output_dem}.npy')
    elif (geotransform[1]>0.0003) & (geotransform[1]<0.001):  #3arcsec Native
        output_dem = f'dem_{input_dem[:-4]}_3arcsecNative'
        np.save(f'{output_dem}.npy', arrayDEM)
        print('\n')
        print(f'Exported DEM as: {output_dem}.npy')
    else:
        output_dem = f'dem_{input_dem[:-4]}_{geotransform[1]}deg'
        np.save(f'{output_dem}.npy', arrayDEM)
        print('\n')
        print(f'Exported DEM as: {output_dem}.npy')


#%% Extract slope
'''
better calculate slope from individual DEM puzzle pieces
scaling factor assumes a cetral latitude and applies it to all
latitudes of the .tif file
for large latitude variation files this will produce distortions
'''

#find center latitude for scaling factor
ymax = geotransform[3]
ymin = ymax + geotransform[5]*dataset.RasterYSize
center_lat = (ymin + ymax)/2
scale_factor = scale_factor / np.cos(np.radians(center_lat))


# Export with diffferent names for native resolutions
if geotransform[1]<0.003:  #1arcsec Native
    output_slope = f'slope_{input_dem[:-4]}_1arcsecNative.tif'
elif (geotransform[1]>0.0003) & (geotransform[1]<0.001):  #3arcsec Native
    output_slope = f'slope_{input_dem[:-4]}_3arcsecNative.tif'
else:
    output_slope = f'slope_{input_dem[:-4]}.tif'


print('\n')
print('Computing Slope...')
output_slopeCompute = subprocess.Popen(
    ["gdaldem", "slope", input_dem, output_slope, "-s", f"{scale_factor}"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
    )
for line in output_slopeCompute.stdout:
    # Print output in real time
    print(line, end="")
print(f'Slope file name: {output_slope}')


print('\n')
print('Exporting Slope as numpy array...')
datasetSlope = gdal.Open(output_slope, gdal.GA_ReadOnly)
arraySlope = datasetSlope.GetRasterBand(1).ReadAsArray()

if make_corrections == True:
    # Replace the edges with the adjacent values (values: -9999)
    arraySlope[:,0] = arraySlope[:,1]
    arraySlope[:,-1] = arraySlope[:,-2]
    arraySlope[0,:] = arraySlope[1,:]
    arraySlope[-1,:] = arraySlope[-2,:]

if export_to_device == True:
    np.save(f'{output_slope[:-4]}.npy', arraySlope)
    print(f'Exported as: {output_slope[:-4]}.npy')

if save_memory == True:
    datasetSlope = None

#%% Extract Aspect
# Export with diffferent names for native resolutions
if geotransform[1]<0.003:  #1arcsec Native
    output_aspect = f'aspect_{input_dem[:-4]}_1arcsecNative.tif'
elif (geotransform[1]>0.0003) & (geotransform[1]<0.001):  #3arcsec Native
    output_aspect = f'aspect_{input_dem[:-4]}_3arcsecNative.tif'
else:
    output_aspect = f'aspect_{input_dem[:-4]}.tif'


print('\n')
print('Computing Aspect...')
output_aspectCompute = subprocess.Popen(
    ["gdaldem", "aspect", input_dem, output_aspect],# "-zero_for_flat"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
    )
for line in output_aspectCompute.stdout:
    # Print output in real time
    print(line, end="")
print(f'Aspect file name: {output_aspect}')

print('\n')
print('Exporting Aspect as numpy array...')
datasetAspect = gdal.Open(output_aspect, gdal.GA_ReadOnly)
arrayAspect = datasetAspect.GetRasterBand(1).ReadAsArray()

if make_corrections == True:
    arrayAspect[arrayAspect == -9999] = -1
    arrayAspect = np.round(arrayAspect*2)/2  #rounds to 0.5

if export_to_device == True:
    np.save(f'{output_aspect[:-4]}.npy', arrayAspect)
    print(f'Exported as: {output_aspect[:-4]}.npy')

#mask = np.ones_like(arrayAspect)
#mask[arrayAspect == -1] = 0
#mask[arrayAspect > 0] = 2

if save_memory == True:
    datasetAspect = None


#%% Plot bands - Merge together
if save_memory == True:
    dataset = None  # if i want to close the dataset
    
    
print('\n')
if plot_bands == True:
    print('Plotting DEM...')
    #plt.figure(figsize=(14,14))
    plt.imshow(arrayDEM[:,:], cmap='inferno')
    gg=np.where(arrayDEM==arrayDEM.max())
    plt.scatter(gg[1][0], gg[0][0], marker='+', c='g')
    #plt.xticks(lon)
    #plt.yticks(lat)
    plt.axis('off')
    plt.savefig(f'{input_dem[:-4]}DEM.png', dpi=300)
    plt.show()
    
    print('Plotting Slope...')
    #plt.figure(figsize=(14,14))
    plt.imshow(arraySlope[:,:], cmap='magma')
    ggs=np.where(arraySlope==arraySlope.max())
    plt.scatter(ggs[1][0], ggs[0][0], marker='+', c='g')
    #plt.xticks(lon)
    #plt.yticks(lat)
    plt.axis('off')
    plt.savefig(f'{output_slope[:-4]}.png', dpi=300)
    plt.show()
    
    print('Plotting Aspect...')
    #plt.figure(figsize=(14,14))
    #import matplotlib as mpl
    cmap = colormaps.get_cmap('twilight')
    #cmap.set_bad(color = 'k')
    plt.imshow(arrayAspect[:,:], cmap=cmap)
    gga=np.where(arrayAspect==arrayAspect.max())
    #plt.scatter(gga[1][0], gga[0][0], marker='+', c='g')
    #plt.xticks(lon)
    #plt.yticks(lat)
    plt.axis('off')
    plt.savefig(f'{output_aspect[:-4]}.png', dpi=300)
    plt.show()


if merge_outputs == True:
    arrayMorphometry = np.stack((arrayDEM, arraySlope, arrayAspect), 
                                axis=-1)
    #stack along the last axis, like an RGB image
    print('\n')
    print('Exporting DEM, Slope, Aspect as a unified numpy array..')
    if geotransform[1]<0.003:  #1arcsec Native
        output_morphometry = f'demSLopeAspect_{input_dem[:-4]}_1arcsecNative.tif'
    elif (geotransform[1]>0.0003) & (geotransform[1]<0.001):  
        output_morphometry = f'demSLopeAspect_{input_dem[:-4]}_3arcsecNative.tif'
    else:
        output_morphometry = f'demSLopeAspect_{input_dem[:-4]}.tif' 
    np.save(output_morphometry, arrayMorphometry)
    print(f'Exported as: {output_morphometry[:-4]}.npy')


print('\n')
print('Done.')


