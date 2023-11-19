# This script generates a DataFrame with the information about the satellite imagery avalailable for the locations of interest.

import json
import requests
import sys
import pandas as pd
from shapely.geometry import shape, box
from landsat_login_info import my_username, my_password


# Send http request
def sendRequest(url, data, apiKey=None):
    json_data = json.dumps(data)

    if apiKey == None:
        response = requests.post(url, json_data)
    else:
        headers = {"X-Auth-Token": apiKey}
        response = requests.post(url, json_data, headers=headers)

    try:
        httpStatusCode = response.status_code
        if response == None:
            print("No output from service")
            sys.exit()
        output = json.loads(response.text)
        if output["errorCode"] != None:
            print(output["errorCode"], "- ", output["errorMessage"])
        if httpStatusCode == 404:
            print("404 Not Found")
            sys.exit()
        elif httpStatusCode == 401:
            print("401 Unauthorized")
            sys.exit()
        elif httpStatusCode == 400:
            print("Error Code", httpStatusCode)
            sys.exit()
    except Exception as e:
        response.close()
        print(e)
        sys.exit()
    response.close()

    return output


# Login
payload = {"username": my_username, "password": my_password}
serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
response = sendRequest(serviceUrl + "login", payload)
if response["errorCode"] == None:
    apiKey = response["data"]
else:
    sys.exit()
print("API Key: " + apiKey + "\n")

# New coordinates for search

df_locations = pd.read_csv("../code_locations_slums.csv")
df_locations = df_locations.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)

df_new_boxes = pd.read_csv("../new_bboxes_locations.csv")
df_new_boxes = df_new_boxes.drop(["Unnamed: 0"], axis=1)

df = df_new_boxes[
    df_new_boxes["location_ID"].isin(df_locations["CD_GEOCODM"].to_list())
]

df = df.rename(columns={"location_ID": "CD_MUNICIP"})

df_for_payload = df

locations_no_scene = []

# Creating df to store complete results
df_all_scenes = pd.DataFrame(
    columns=[
        "entityId",
        "displayId",
        "cloudCover",
        "spatialCoverage",
        "spatialBounds",
        "startDate",
        "overlap",
        "location",
    ]
)

# Searches all scenes for each location in df_for_payload
# Also, checks the overlapping %

for i in range(0, len(df_for_payload)):
    payload = {
        "maxResults": 100,
        "datasetName": "landsat_tm_c2_l2",
        "sceneFilter": {
            "cloudCoverFilter": {"max": 20, "min": 0, "includeUnknown": False},
            "acquisitionFilter": {"end": "2010-10-30", "start": "2010-08-01"},
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {
                    "latitude": df_for_payload["lowerLeft_lat"].iloc[i],
                    "longitude": df_for_payload["lowerLeft_lon"].iloc[i],
                },
                "upperRight": {
                    "latitude": df_for_payload["upperRight_lat"].iloc[i],
                    "longitude": df_for_payload["upperRight_lon"].iloc[i],
                },
            },
        },
    }

    results_images = sendRequest(serviceUrl + "scene-search", payload, apiKey)

    all_data = results_images["data"]

    if len(all_data["results"]) > 0:
        # Organising relevant data in a dataframe
        scenes_list = []
        for item in all_data["results"]:
            dict_scenes = {}
            dict_scenes["entityId"] = item.get("entityId")
            dict_scenes["displayId"] = item.get("displayId")
            dict_scenes["cloudCover"] = item.get("cloudCover")
            dict_scenes["spatialCoverage"] = item.get("spatialCoverage")
            dict_scenes["spatialBounds"] = item.get("spatialBounds")
            dict_scenes["startDate"] = item.get("temporalCoverage").get("startDate")
            scenes_list.append(dict_scenes)

        df_scenes = pd.DataFrame(data=scenes_list)

        # Checking overlap of the scenes with location of interest

        # polygon = box(df_for_payload['lowerLeft_longitude'].iloc[i],df_for_payload['lowerLeft_latitude'].iloc[i],df_for_payload['upperRight_longitude'].iloc[i],df_for_payload['upperRight_latitude'].iloc[i])
        polygon = box(
            df_for_payload["lowerLeft_lon"].iloc[i],
            df_for_payload["lowerLeft_lat"].iloc[i],
            df_for_payload["upperRight_lon"].iloc[i],
            df_for_payload["upperRight_lat"].iloc[i],
        )
        myNewAOI = {"type": "Polygon", "coordinates": [list(polygon.exterior.coords)]}
        my_location_shape = shape(myNewAOI)

        overlaps = []
        for footprint in df_scenes["spatialCoverage"].tolist():
            s = shape(footprint)
            # footprints.append(s)
            overlap = 100.0 * (
                my_location_shape.intersection(s).area / my_location_shape.area
            )
            overlaps.append(overlap)

        df_scenes["overlap"] = pd.Series(overlaps, index=df_scenes.index)
        # scenes['footprint'] = pd.Series(footprints, index=scenes.index)

        # Adding location name
        df_scenes = df_scenes.assign(
            location=len(df_scenes) * [df_for_payload["CD_MUNICIP"].iloc[i]]
        )

        df_scenes = pd.DataFrame(df_scenes)

        df_all_scenes = pd.concat([df_all_scenes, df_scenes], axis=0)

        print(
            "Finished adding info #{} about {}. Scenes found {}.".format(
                i, df_for_payload["CD_MUNICIP"].iloc[i], len(df_scenes)
            )
        )

    else:
        locations_no_scene.append(str(df_for_payload["CD_MUNICIP"].iloc[i]))
        print("No info #{} about {}.".format(i, df_for_payload["CD_MUNICIP"].iloc[i]))

df_all_scenes.to_csv("../df_2010_landsat_323_complete.csv")
