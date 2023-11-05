## This script generates rasters with all available bands and masks for each location.

import os
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
import json
from pyproj import CRS
from rasterio.mask import mask

df_116 = pd.read_csv('../df_scenes_downloaded.csv')
df = df_116

location_list = list(df['location'])

for location in location_list:

    entityId = df[df['location']==location]['entityId'].iloc[0]

    ## Stacking bands of LandSat Imagery

    if len(df[df['entityId']==entityId]) == 1:
        folder = location
    else:
        folder = df[df['entityId']==entityId]['location'].iloc[0]

    PATH_FOLDER = 'D:/brazilian_dataset/2010_landsat/'+\
        str(folder) + "/" + entityId
    raster_list = [x for x in os.listdir(PATH_FOLDER) if '.TIF' in x]
    # print(raster_list)

    bands_paths = []
    for band_number in range(1,8):
        # print(band_number)
        band_path = [x for x in raster_list if '_B{}'.format(band_number) in x][0]
        # print(band_path)
        bands_paths.append(band_path)

    # B1_blue_path = [x for x in raster_list if '_B1' in x][0]
    # B2_green_path = [x for x in raster_list if '_B2' in x][0]
    B3_red_path = [x for x in raster_list if '_B3' in x][0]
    # B4_path = [x for x in raster_list if '_B4' in x][0]
    # B5_path = [x for x in raster_list if '_B5' in x][0]
    # B6_path = [x for x in raster_list if '_B6' in x][0]
    # B7_path = [x for x in raster_list if '_B7' in x][0]

    bands_rasters = []
    for band_path in bands_paths:
        band_raster = rasterio.open(PATH_FOLDER+"/"+band_path, mode='r')
        bands_rasters.append(band_raster)

    band_red=rasterio.open(PATH_FOLDER+"/"+B3_red_path, mode='r')
    # band_green=rasterio.open(PATH_FOLDER+"/"+B2_green_path)
    # band_blue=rasterio.open(PATH_FOLDER+"/"+B1_blue_path)

    # rasterio.plot.show(band_red.read(1), cmap='terrain')
    # plt.imshow(band_red.read(1), cmap='terrain')

    landsat_epsg = band_red.profile['crs']
    # print(landsat_epsg)

    ## Clipping the raster with location bbox

    crs_4326 = CRS.from_user_input(4326)
    # crs_32624 = CRS.from_user_input(32624)

    # metropolitan_locations = pd.read_csv('../metropolitan_locations.csv')
    # name_region = metropolitan_locations[metropolitan_locations['RME_CD']==float(RME_CD)]['RME_NM'].unique()[0]
    # print(name_region)

    # df_metro_locations = pd.read_csv('../metro_regions_bbox.csv')

    # Reading coordinates for search
    # df_complete_bbox_municipalities = pd.read_csv('../Localities_Slums_Coordinates/bbbox_all_municipalities.csv')
    # df_complete_bbox_municipalities = df_complete_bbox_municipalities.drop(['Unnamed: 0','id_in_shapefile','NM_MUNICIP'], axis=1)

    # df_bbox = df_complete_bbox_municipalities[df_complete_bbox_municipalities['CD_GEOCODI'] == location]

    # minx_4326=df_bbox['lowerLeft_longitude'].iloc[0]
    # miny_4326=df_bbox['lowerLeft_latitude'].iloc[0]
    # maxx_4326=df_bbox['upperRight_longitude'].iloc[0]
    # maxy_4326=df_bbox['upperRight_latitude'].iloc[0]

    # New bbox

    df_new_boxes = pd.read_csv('../new_bboxes_locations.csv')
    df_new_boxes = df_new_boxes.drop(['Unnamed: 0'], axis=1)

    df_bbox = df_new_boxes[df_new_boxes['location_ID'] == location]

    minx_4326=df_bbox['lowerLeft_lon'].iloc[0]
    miny_4326=df_bbox['lowerLeft_lat'].iloc[0]
    maxx_4326=df_bbox['upperRight_lon'].iloc[0]
    maxy_4326=df_bbox['upperRight_lat'].iloc[0]

    bbox_region_4326 = shapely.geometry.box(minx_4326, miny_4326, maxx_4326, maxy_4326)

    geo_4326 = gpd.GeoDataFrame({'geometry': bbox_region_4326}, index=[0], crs=crs_4326)
    geo_landsat = geo_4326.to_crs(landsat_epsg)
    # print(geo_landsat)

    # Checking intersection of the two rasters (now in the same coordinate system)

    (minx_r, miny_r, maxx_r, maxy_r) = band_red.bounds
    bbox_raster = shapely.geometry.box(minx_r, miny_r, maxx_r, maxy_r)
    # raster_shape = shapely.geometry.shape(bbox_raster)

    (minx, miny, maxx, maxy) = geo_landsat['geometry'][0].bounds
    bbox_region = shapely.geometry.box(minx, miny, maxx, maxy)
    # region_shape = shapely.geometry.shape(bbox_region)

    overlap_region = bbox_region.intersection(bbox_raster)
    (minx_overlap_box, miny_overlap_box, maxx_overlap_box, maxy_overlap_box) = overlap_region.bounds
    bbox_overlap_region = shapely.geometry.box(minx_overlap_box, miny_overlap_box, maxx_overlap_box, maxy_overlap_box)

    overlap = overlap_region.area/bbox_region.area
    print('The location of interest {} is {}% in the raster.'\
        .format(location,round(overlap,3)*100))

    geo_bbox_overlap_region = gpd.GeoDataFrame({'geometry': bbox_overlap_region}, \
        index=[0], crs=landsat_epsg)

    def getFeatures(gdf):
        """Function to parse features from GeoDataFrame in such a 
        manner that rasterio wants them"""
        return [json.loads(gdf.to_json())['features'][0]['geometry']]

    coords = getFeatures(geo_bbox_overlap_region)
    # print(coords)

    # Masking all bands and concatenating the results in a single array

    out_img_red, out_transform_red = mask(dataset=band_red, shapes=coords, crop=True)

    clipped_rasters = []
    for band in bands_rasters:
        out_img_band, _ = mask(dataset=band, shapes=coords, crop=True)
        clipped_rasters.append(out_img_band)

    # out_img_green, out_transform_green = mask(dataset=band_green, shapes=coords, crop=True)
    # out_img_blue, out_transform_blue = mask(dataset=band_blue, shapes=coords, crop=True)

    # print(out_transform_red == out_transform_green)
    # print(out_transform_green == out_transform_blue)

    out_img = np.concatenate([clipped_rasters[0],clipped_rasters[1],clipped_rasters[2],\
        clipped_rasters[3],clipped_rasters[4],clipped_rasters[5],clipped_rasters[6]        
        ])

    # Saving the file

    PATH_RASTER = '../../BrazilianDataset/intermediary_rasters/'+str(location)+'/'
    if os.path.exists(PATH_RASTER)==False:
        os.makedirs(PATH_RASTER)

    out_meta = band_red.meta
    out_meta.update({"count": out_img.shape[0],
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform_red})
    # print(out_meta)

    COMPLETE_PATH_RASTER = PATH_RASTER+str(location)+'.tif'

    with rasterio.open(COMPLETE_PATH_RASTER, "w", **out_meta) as dest:
        dest.write(out_img)

    ## Mask the slums 

    band_red_clipped=rasterio.open(COMPLETE_PATH_RASTER, mode='r')

    PATH_SLUMS_SHAPE = "../Localities_Slums_Coordinates/SlumsLocation_epsg4326_v2.shp"
    os.path.exists(PATH_SLUMS_SHAPE)

    slums_shape = gpd.read_file(PATH_SLUMS_SHAPE)
    # print(slums_shape.crs)
    slums_shape_landsat_epsg = slums_shape.to_crs(landsat_epsg)
    # print(geo_bbox_overlap_region.crs)
    # print(slums_shape_landsat_epsg.crs)

    clipped_slums = gpd.clip(slums_shape_landsat_epsg, geo_bbox_overlap_region)
    json_file = clipped_slums.to_json()
    json_file2 = json.loads(json_file)

    shapes_slums = []
    for i in range(len(clipped_slums)):
        extracted_features = json_file2['features'][i]['geometry']
        shapes_slums.append(extracted_features)

    # with fiona.open(PATH_SLUMS_SHAPE, "r") as shapefile:
    #     shapes = [feature["geometry"] for feature in shapefile]

    out_img_mask, out_transform_mask = mask(dataset=band_red_clipped, shapes=shapes_slums, crop=False)

    out_meta_mask = band_red_clipped.meta
    out_meta_mask.update({"count": 1,
                    "transform": out_transform_mask,
                    "nodata": 2.0
                    })
    # print(out_meta_mask)

    # Processing numpy array
    # print(np.unique(out_img_mask[0],return_counts=True))
    out_img_mask_mod = np.where(out_img_mask[0] > 0, 1, 0).astype('uint16')
    # print(np.unique(out_img_mask_mod,return_counts=True))
    out_img_mask_final = np.reshape(out_img_mask_mod,(1,out_img_mask_mod.shape[0], out_img_mask_mod.shape[1]))
    # sum(sum(out_img_mask_final[0,:,:] == out_img_mask_mod))==3229*3312

    PATH_MASK = '../../BrazilianDataset/masks/'
    if os.path.exists(PATH_MASK)==False:
        os.makedirs(PATH_MASK)

    COMPLETE_PATH_MASK = PATH_MASK+'mask_'+str(location)+'.tif'

    with rasterio.open(COMPLETE_PATH_MASK, "w", **out_meta_mask) as dest:
        dest.write(out_img_mask_final)

    generated_mask=rasterio.open(COMPLETE_PATH_MASK, mode='r')
    np.unique(generated_mask.read(1), return_counts=True)

    print('Finished location {}'.format(location))

## Closing all files
band_red.close
band_red_clipped.close
generated_mask.close