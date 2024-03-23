import requests

def download_model_from_url(url, output_path):
    # Send a GET request to download the file
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content of the response to a file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    else:
        print("Failed to download model. Status code:", response.status_code)

# Example usage:
models = ["https://storage.googleapis.com/sandeep_personal/Bone.pt", "https://storage.googleapis.com/sandeep_personal/TB.keras", "https://storage.googleapis.com/sandeep_personal/best_model_pn.keras", "https://storage.googleapis.com/sandeep_personal/covid_sequential%20.h5","https://storage.googleapis.com/sandeep_personal/PN.h5","https://storage.googleapis.com/sandeep_personal/retina%20(1).pt","https://storage.googleapis.com/sandeep_personal/Model_XRAY_final.h5"]
output_path = ['Bone.pt', "TB.keras","PN.keras","covid.h5","PN.h5","retina.pt","x_ray.h5"]
i = -1
download_model_from_url(models[i], output_path[i])
