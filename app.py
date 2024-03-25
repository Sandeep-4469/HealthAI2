import uvicorn
from fastapi import FastAPI, UploadFile, File, Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import io
from PIL import Image
import os
import json
import shutil
from fastapi.responses import RedirectResponse
from bone import bone_main
from chat import get_completion_from_messages,collect_messages_text
from all_models import xray_type
from lung import test_lung
from retina import test_retina
# from chat import collect_messages_text
import dotenv
app = FastAPI()
app.mount("/static", StaticFiles(directory = "static"), name = "static")
data = {}
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
context = []
templates = Jinja2Templates(directory="templates")
data = {}
def get_dict():
    return data
class Message(BaseModel):
    content: str
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message:Message):
    user_message = message.content
    print(context)
    response = collect_messages_text(user_message,context)
    return {"message": response}

@app.post("/upload")
async def upload_image(request: Request, name: str = Form(...), age: str = Form(...), gender: str = Form(...), image: UploadFile = File(...)):
    if image is not None and image.filename != '':
        data["name"] = name 
        data["age"] = age 
        data["gender"] = gender
        # Saving the uploaded image to the specified directory
        image_path = os.path.join('static', image.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        x = xray_type(image_path)
        print(x)
        if x==0:
            d = bone_main(image_path)
        elif x==1:
            d = test_lung(image_path)
        else:
            d = test_retina(image_path)
        data.update(d)
        context.append({"role": "system", "content": f"""Consider you are a doctorbot.
The details of the patient are as follows and/
The user provides an x-ray and the details from the x-ray are as follows:
{str(data)}.
Now act as a chatbot and answer questions asked by the user.
First, give the user the X-ray report, then ask the user whether he/she has any questions.
Answer the questions wisely in short form.
Use the name of the user to interact.
Ask for BMI and blood glucose levels in case of diabetes.
If no BMI is known, ask for height and weight and calculate.
Use BMI and blood glucose levels before providing the X-ray report only for Diabetic Retinopathy.
If the user asks questions out of context - Simply warn him.
Now act as a chatbot and answer questions asked by the user.
First of all, greet the user with his name and show the result given in the x-ray data."""})
        
        # Return a response with the processed image (and any other data)
        return templates.TemplateResponse("index.html", {"request": request, "image_url":f'{image_path}',"path": '/static/output_image.jpg'})
    return templates.TemplateResponse("index.html", {"request": request, "error": "No image selected."})



if __name__ == '__main__':
   uvicorn.run(app, host='0.0.0.0', port=8000)


   #apt-get install -y libgl1-mesa-dev
   #sudo apt-get install libgl1-mesa-glx
   #pip install python-multipart
   #python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl