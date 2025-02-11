# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 00:57:55 2024

@author: yiann
"""

import rasterio
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt

dataset = rasterio.open('coco.tif')
# "coco.tif" taken from SRTM 1 arcsecond DEM

width = dataset.width
height = dataset.height
transform = dataset.transform
indexes = dataset.indexes


print('Bands dtypes: {}'.format({i: dtype for i, 
                                 dtype in zip(dataset.indexes, 
                                              dataset.dtypes)}))
print('\n')
print('Bounds: {}'.format(dataset.bounds))
print('\n')
print('Transform: {}'.format(list(dataset.transform)))
print('\n')
print('Coordinate Reference System: {}'.format(dataset.crs))


array = dataset.read(indexes[0])

# 1: 3600 arcseconds in a deg - reduction factor 360 for 0.1deg (~11km)
arrayLD1 = array[:-1,:-1].copy()
arrayLD1 = skimage.measure.block_reduce(arrayLD1, block_size=(360,360), 
                                        func=np.mean)

# 2: 3600 arcseconds in a deg - reduction factor 36 for 0.01deg (~1km)
arrayLD2 = array[:-1,:-1].copy()
arrayLD2 = skimage.measure.block_reduce(arrayLD2, block_size=(36,36), 
                                        func=np.mean)

np.save('dem0.1deg.npy', arrayLD1)
np.save('dem0.01deg.npy', arrayLD2)

#%% plotarisma tou DEM
#plt.figure(figsize=(14,14))
plt.imshow(array[:,:], cmap='inferno')
gg=np.where(array==array.max())
#plt.scatter(gg[1][0], gg[0][0], marker='+', c='g')
#plt.xticks(lon)
#plt.yticks(lat)
plt.axis('off')
#plt.savefig('peloponhsos_dem_1arcsecondNativeResolution.png', dpi=300)
plt.show()

#plt.figure(figsize=(14,14))
plt.imshow(arrayLD2[:,:], cmap='inferno')
ggLD2=np.where(arrayLD2==arrayLD2.max())
#plt.scatter(ggLD2[1][0], ggLD2[0][0], marker='+', c='g')
#plt.xticks(lon)
#plt.yticks(lat)
plt.axis('off')
#plt.savefig('peloponhsos_dem_0.01deg.png', dpi=300)
plt.show()

#plt.figure(figsize=(14,14))
plt.imshow(arrayLD1[:,:], cmap='inferno')
ggLD1=np.where(arrayLD1==arrayLD1.max())
plt.scatter(ggLD1[1][0], ggLD1[0][0], marker='+', c='g')
#plt.xticks(lon)
#plt.yticks(lat)
plt.axis('off')
#plt.savefig('peloponhsos_dem_0.1deg.png', dpi=300)
plt.show()



'''
dataset.xy(dataset.height // 2, dataset.width // 2)
dinei ta lat lon gia sygkekrimena pixel

x, y = (dataset.bounds.left, dataset.bounds.top)
dinei ta lat lon tou upper left pixel - tou bound ekei

row, col = dataset.index(x, y)
dinei ta row kai column tou lat lon, pou einai ta x, y

array[row, col]
dinei to value sta epilegmena row kai column 
'''

'''
gia ta ticks:
plt.figure(1)     
ax = plt.subplot(111)
#... do your stuff
#need to figure out your image size divided by the number of labels you want
#FOR EXample, if image size was 180, and you wanted every second coordinate labeled:
ax.set_xticks([i for i in range(0,180,2)]) #python3 code to create 90 tick marks
ax.set_xticklabels([-i for i in range(-90,90,2)]) #python3 code to create 90 labels
#DO SAME FOR Y
'''
