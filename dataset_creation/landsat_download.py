# This script downloads the scenes that were selected landat_scene_selection.py

import json
import requests
import sys
import pandas as pd
import pathlib
from landsat_login_info import my_username, my_password

# Loading df with scenes to download

PATH_DOWNLOAD = "D:/brazilian_dataset/2010_landsat"

df = pd.read_csv("../df_scenes_downloaded.csv")
print("There are {} unique scenes.".format(len(df["entityId"].unique())))

entityIds = list(df["entityId"])


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

# Find the download options for these scenes

payload = {"datasetName": "landsat_tm_c2_l2", "entityIds": entityIds}
downloadOptions = sendRequest(serviceUrl + "download-options", payload, apiKey)

# Aggregate a list of available products

downloads = []
for product in downloadOptions["data"]:
    if product["available"] == True:
        downloads.append({"entityId": product["entityId"], "productId": product["id"]})

print(
    "{} scenes out of {} are available for download.".format(
        len(downloads), len(df["entityId"].unique())
    )
)

df["entityId"].isin(pd.DataFrame(downloads)["entityId"])

requestedDownloadsCount = len(downloads)

# Requesting URLs for download

label = "download-sample"
payload = {"downloads": downloads, "label": label}

requestResults = sendRequest(serviceUrl + "download-request", payload, apiKey)

downloadUrls = []
for download in requestResults["data"]["availableDownloads"]:
    downloadUrls.append(download["url"])

# Downloading the available scenes

for i in range(0, len(downloads)):
    if len(downloadUrls) == len(downloads):
        entity_to_download = downloads[i]["entityId"]
        location_download = df[df["entityId"] == entity_to_download]["location"].iloc[0]
        response = requests.get(downloadUrls[i])
        PATH_TO_FILE = (
            PATH_DOWNLOAD
            + "/"
            + str(location_download)
            + "/"
            + entity_to_download
            + "/"
        )
        pathlib.Path(PATH_TO_FILE).mkdir(parents=True, exist_ok=True)
        open(PATH_TO_FILE + "{}.tar".format(entity_to_download), "wb").write(
            response.content
        )
        print("Finished #{}".format(i))

# Logging out

endpoint = "logout"
if sendRequest(serviceUrl + endpoint, None, apiKey) == None:
    print("Logged Out\n")
else:
    print("Logout Failed\n")
