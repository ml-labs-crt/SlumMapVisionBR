# This script selects the imagery that will be downloaded

import pandas as pd
import json
import fiona
from shapely.geometry import shape, box

# New bbox area

locations_with_slum = pd.read_csv("../code_locations_slums.csv")
list_locations = locations_with_slum["CD_GEOCODM"].astype(str).to_list()

PATH_SHAPE_LOCATIONS = "../BR_Localidades_2010_v1/Locations_Brazil_2010_epsg4326.shp"
data_locations = fiona.open(PATH_SHAPE_LOCATIONS)
data_locations.crs
shapefile_content = []
for item in range(len(data_locations)):
    if data_locations[item]["properties"]["CD_GEOCODM"] in list_locations:
        dictionary_properties = data_locations[item]["properties"]
        data = {
            "id_in_shapefile": item,
            "TIPO": dictionary_properties["TIPO"],
            "NM_BAIRRO": dictionary_properties["NM_BAIRRO"],
            "CD_GEOCODM": dictionary_properties["CD_GEOCODM"],
            "NM_MUNICIP": dictionary_properties["NM_MUNICIP"],
            "NM_UF": dictionary_properties["NM_UF"],
            "CD_NIVEL": dictionary_properties["CD_NIVEL"],
            "NM_CATEGOR": dictionary_properties["NM_CATEGOR"],
            "NM_LOCALID": dictionary_properties["NM_LOCALID"],
            "LONG": dictionary_properties["LONG"],
            "LAT": dictionary_properties["LAT"],
        }
        shapefile_content.append(data)

df_locations = pd.DataFrame(shapefile_content)
df_locations = df_locations[df_locations["NM_CATEGOR"] == "CIDADE"]
df_locations.to_csv("../coordinates_locations_slums.csv")

PATH_SHAPE_SLUMS = "../Localities_Slums_Coordinates/SlumsLocation_epsg4326_v2.shp"
data_slums = fiona.open(PATH_SHAPE_SLUMS)
data_slums.crs
data_locations.crs == data_slums.crs
shapefile_content = []
for item in range(len(data_slums)):
    dictionary_properties = data_slums[item]["properties"]
    bbox_coord = shape(data_slums[item]["geometry"]).bounds
    data = {
        "id_in_shapefile": item,
        "CodAGSN": dictionary_properties["CodAGSN"],
        "NM_AGSN": dictionary_properties["NM_AGSN"],
        "CD_GEOCODM": dictionary_properties["CD_GEOCODM"],
        "NM_MUNICIP": dictionary_properties["NM_MUNICIP"],
        "uf": dictionary_properties["uf"],
        "lowerLeft_lon": bbox_coord[0],
        "lowerLeft_lat": bbox_coord[1],
        "upperRight_lon": bbox_coord[2],
        "upperRight_lat": bbox_coord[3],
    }
    shapefile_content.append(data)
df_slums = pd.DataFrame(shapefile_content)
df_slums.to_csv("../coordinates_6329_slums.csv")

bboxes_slum_list = []
for location in list_locations:
    df = df_slums[df_slums["CD_GEOCODM"] == location]
    df_location = df_locations[df_locations["CD_GEOCODM"] == location]
    data = {
        "location_ID": location,
        "lowerLeft_lon": min(min(df["lowerLeft_lon"]), df_location["LONG"].iloc[0]),
        "lowerLeft_lat": min(min(df["lowerLeft_lat"]), df_location["LAT"].iloc[0]),
        "upperRight_lon": max(max(df["upperRight_lon"]), df_location["LONG"].iloc[0]),
        "upperRight_lat": max(max(df["upperRight_lat"]), df_location["LAT"].iloc[0]),
    }
    bboxes_slum_list.append(data)
df_new_bboxes = pd.DataFrame(bboxes_slum_list)
df_new_bboxes.to_csv("../new_bboxes_locations.csv")

# Continue analysing the intersection of images

df_new_boxes = pd.read_csv("../new_bboxes_locations.csv")
df_all_scenes = pd.read_csv("../df_2010_landsat_323_complete.csv")
locations_with_slum = pd.read_csv("../code_locations_slums.csv")
list_locations = locations_with_slum["CD_GEOCODM"].astype(str).to_list()

# Checking overlap of the scenes with location of interest

overlaps = []

for scene in range(len(df_all_scenes)):
    line = df_all_scenes.iloc[scene]
    location = line.location
    df_info_coord = df_new_boxes[df_new_boxes["location_ID"] == location]
    i = 0
    polygon = box(
        df_info_coord["lowerLeft_lon"].iloc[i],
        df_info_coord["lowerLeft_lat"].iloc[i],
        df_info_coord["upperRight_lon"].iloc[i],
        df_info_coord["upperRight_lat"].iloc[i],
    )
    myNewAOI = {"type": "Polygon", "coordinates": [list(polygon.exterior.coords)]}
    new_bbox = shape(myNewAOI)
    line_spatial_coverage = line.spatialCoverage.replace("'", '"')
    footprint = json.loads(line_spatial_coverage)
    raster_coverage = shape(footprint)
    overlap = 100.0 * (new_bbox.intersection(raster_coverage).area / new_bbox.area)
    overlaps.append(overlap)

df_all_scenes["overlap_new_bbox"] = pd.Series(overlaps, index=df_all_scenes.index)

# Only select scenes that contain 100% of the area new bbox area and zero cloud cover

# These images will be downloaded and used to create the dataset

# Checking the overlap of the new bbox with downloadead images

df_116 = pd.read_csv("../df_scenes_downloaded.csv")
df_new_boxes = pd.read_csv("../new_bboxes_locations.csv")

overlaps = []

for scene in range(len(df_116)):
    line = df_116.iloc[scene]
    location = line.location
    df_info_coord = df_new_boxes[df_new_boxes["location_ID"] == location]
    i = 0
    polygon = box(
        df_info_coord["lowerLeft_lon"].iloc[i],
        df_info_coord["lowerLeft_lat"].iloc[i],
        df_info_coord["upperRight_lon"].iloc[i],
        df_info_coord["upperRight_lat"].iloc[i],
    )
    myNewAOI = {"type": "Polygon", "coordinates": [list(polygon.exterior.coords)]}
    new_bbox = shape(myNewAOI)
    line_spatial_coverage = line.spatialCoverage.replace("'", '"')
    footprint = json.loads(line_spatial_coverage)
    raster_coverage = shape(footprint)
    overlap = 100.0 * (new_bbox.intersection(raster_coverage).area / new_bbox.area)
    overlaps.append(overlap)

df_116["overlap_new_bbox"] = pd.Series(overlaps, index=df_116.index)

df_116["overlap_new_bbox"].value_counts()

# Scenes contain 100% of the area.
