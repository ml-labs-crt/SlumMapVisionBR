Status: Archive (code is provided as-is, no updates expected)

## SlumMapVisionBR: A Dataset for Mapping Slums with Medium-Resolution Satellite Imagery

This repo contains the code to recreate the data set and reproduce the results of the paper "SlumMapVisionBR: A Dataset for Mapping Slums
with Medium-Resolution Satellite Imagery". 

## Dataset 

- The SlumMapVisionBR dataset is available at:

## Files

`info_locations.csv` contains information about each location.<br>
`info_locations_data_dictionary.txt` contains a description of each field in the `info_locations.csv` file.<br>

### Files in the `dataset_creation` folder:

`generating_dataset.py` generates rasters with all available bands and masks for each location.<br>
`df_scenes_downloaded.csv` lists all satellite imagery scenes downloaded.<br>
`new_bboxes_locations.csv` the coordinates of the bounding boxes used to clip raster of each location.<br>
`SlumsLocation_epsg4326_v2` contains polygons with locations of the slums in EPSG4326. Census data was obtained in the link listed in the 'Data' section below. <br>

`landsat_scene_search.py` generates a DataFrame with the information about the satellite imagery avalailable for the locations of interest.<br>
`code_locations_slums.csv` code and name of all locations/municipalities that contain slums. Census data was obtained in the link listed in the 'Data' section below. <br>

`landsat_scene_selection.py` selects the imagery that will be downloaded.<br>
`Locations_Brazil_2010_epsg4326.shp` data of the central location of each munipality. Original data was obtained in the link listed in the 'Data' section below. <br>

`landsat_download.py` downloads the scenes that were selected `landat_scene_selection.py`.<br>
`landsat_untar.py` untars files that were downloaded using `landsat_download.py`.<br>

### Files in the `benchmark` folder:

`multispectral.py` trains a random forest model with multispectral data (random sampling) as features for each 116 locations. <br>

`multispectral_halves.py` trains a random forest model with multispectral data as features for each 32 locations in the `info_locations_32.csv` file. Data is split in half and the model is trained on one half and tested on the other half (both halves).<br>
`info_locations_32.csv` list of the 32 locations were the multispectral models convolutional neural network model was evaluated.<br>

`segmentation_halves.py` trains models with half of the image and test it with the other half (both halves) for the locations in the `info_capitals.csv` file .<br>
`info_capitals.csv` list of locations were the convolutional neural network model (`segmentation_halves.py`) was evaluated.<br>

### Files in the `results` folder:

`cnn_7_bands.csv` results CNN capitals. Models are trained with half of the image and tested with the other half (both halves).<br>
`multispectral.csv` results multispectral random sampling for all 116 locations.<br>
`multispectral_halves.csv` Models are trained with half of the image and tested with the other half (both halves).<br>
`visualisations` folder contains the visualisations of imagery, ground-truth and predictions for each location.<br>

## Census Data 

- The original census information used to generate the ground-truth data was obtained on [this link](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/tipologias-do-territorio/15788-aglomerados-subnormais.html) on 03-Nov-2019. 
- The data of the central location of each municipality was obtained on [this link](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/estrutura-territorial/27385-localidades.html?=&t=acesso-ao-produto) on 11-Mar-2022.

## Citation

```
TBD

```