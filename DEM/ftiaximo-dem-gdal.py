# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:32:49 2024

@author: yiann

kalopismos tou DEM tou SRTM gia thn perioxh ths poloponnhsou arxika
gi athn ellada tha ginei se argotero xrono
skeftomai na xrhsimopoihsw kai qgis/arcgis gia thn douleia auth
paizei na einai pio eukolo...

agregation to lower resolution ginetai kai ws (sto CLI):
gdalwarp -tr 0.1 0.1 -r average input_dem.tif dem_0.1deg_average.tif

download gdal wheels from:
https://github.com/cgohlke/geospatial-wheels/releases/tag/v2024.9.22
this link contains other geospatial librariies wheels too
i personally downloaded the gdal wheel for my machine's python3.10
e.g. my cases name: GDAL-3.9.2-cp310-cp310-win_amd64.whl 
"""
import numpy as np
from osgeo import gdal
import skimage.measure #dokimase kai allo library gia aggregation
import matplotlib.pyplot as plt


# merge tiffs through gdal_merge.py command
# 
tif_file = 'coco.tif'
compute_slope = False  #set true if slope is being conputed

#options pou exw egw sto laptop mou:
#tif_file = 'coco.tif'
#tif_file = 'coco__0.01deg_average.tif'
#tif_file = 'coco__0.1deg_average.tif'

export_to_device = False
aggregate_values = False
export_aggregated = False
plot_bands = True


#needed for rounding
def count_decimal_places(number):
    """Counts the number of decimal places in a given number."""
    str_num = str(number).rstrip('0')  # Remove trailing zeros
    if '.' in str_num:
        return len(str_num.split('.')[1])
    return 0


#%% diavasma .tif mesw gdal
gdal.UseExceptions()

print('Importing Dataset...\n')
dataset = gdal.Open(tif_file, gdal.GA_ReadOnly)
for x in range(1, dataset.RasterCount + 1):
    band = dataset.GetRasterBand(x)
    array = band.ReadAsArray()
    
if compute_slope == True:
    # Replace the edges with the adjacent values
    array[:,0] = array[:,1]
    array[:,-1] = array[:,-2]
    array[0,:] = array[1,:]
    array[-1,:] = array[-2,:]

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


# giat ieinai diaforetika apo ta array.min(), array.max() gia True??
#gia false einai ta idia...
minGDAL, maxGDAL = band.ComputeRasterMinMax(False)
minArray, maxArray = array.min(), array.max()


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
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))
print('\n')
print("Min={:.3f}, Max={:.3f}".format(minGDAL,maxGDAL))

# maybe change this sometime later, -6 is the lowest point in Greece
if compute_slope != True:
    array[array<-6] = -6

if export_to_device == True:
    if geotransform[1]<0.003:  #1arcsec Native
        np.save(f'dem1arcsecNativeGDAL.npy', array)
    elif (geotransform[1]>0.0003) & (geotransform[1]<0.001):  #3arcsec Native
        np.save(f'dem3arcsecNativeGDAL.npy', array)
    else:
        np.save(f'dem{geotransform[1]}GDAL.npy', array)
    

#%% aggregate from for 1 arcsec DEM
if aggregate_values == True:
    
   # 1: 3600 arcseconds in a deg - reduction factor 360 for 0.1deg (~11km)
   array01 = array[:-1,:-1].copy()
   array01 = skimage.measure.block_reduce(array01, block_size=(360,360), 
                                          func=np.mean)
   array01 = np.round(array01 ,decimals=0)
   
   # 2: 3600 arcseconds in a deg - reduction factor 36 for 0.01deg (~1.1km)
   array001 = array[:-1,:-1].copy()
   array001 = skimage.measure.block_reduce(array001, block_size=(36,36), 
                                           func=np.mean)
   array001 = np.round(array001 ,decimals=0)

   
   if export_aggregated == True:
       np.save('dem0.1GDAL_agg.npy', array01)
       np.save('dem0.01GDAL_agg.npy', array001)
     


#%% plot bands
if plot_bands == True:
    print('Plotting...')
    #plt.figure(figsize=(14,14))
    plt.imshow(array[:,:], cmap='magma')
    gg=np.where(array==array.max())
    #plt.scatter(gg[1][0], gg[0][0], marker='+', c='g')
    #plt.xticks(lon)
    #plt.yticks(lat)
    plt.axis('off')
    #plt.savefig('peloponhsos_dem_1arcsecondNativeResolution_gdal.png', dpi=300)
    plt.show()
    
    if aggregate_values == True:
        #plt.figure(figsize=(14,14))
        plt.imshow(array001[:,:], cmap='inferno')
        ggLD2=np.where(array001==array001.max())
        #plt.scatter(ggLD2[1][0], ggLD2[0][0], marker='+', c='g')
        #plt.xticks(lon)
        #plt.yticks(lat)
        plt.axis('off')
        #plt.savefig('peloponhsos_dem_0.01deg_gdal.png', dpi=300)
        plt.show()
        
        #plt.figure(figsize=(14,14))
        plt.imshow(array01[:,:], cmap='inferno')
        ggLD1=np.where(array01==array01.max())
        plt.scatter(ggLD1[1][0], ggLD1[0][0], marker='+', c='g')
        #plt.xticks(lon)
        #plt.yticks(lat)
        plt.axis('off')
        #plt.savefig('peloponhsos_dem_0.1deg_gdal.png', dpi=300)
        plt.show()

    print('Done.')



'''
xamhlotero shmeio: epitalio ston nomo hleias, -6m
to min einai -92m(!) gia kapoio logo
kane mia ekatharhsh sta mikrotera twn -6
vres kai location twn arnhtikwn, na doume pou vgazeito dem tou srtm mlkies
'''






'''
#%% plotarisma tou DEM
#plt.figure(figsize=(14,14))
plt.imshow(array, cmap='inferno')
gg=np.where(array==array.max())
plt.scatter(gg[1][0], gg[0][0], marker='+', c='g')
#plt.xticks(lon)
#plt.yticks(lat)
plt.axis('off')
plt.show()
'''