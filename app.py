import tkinter as tk
import customtkinter as ctk 
from PIL import ImageTk
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud") 
ctk.set_appearance_mode("dark") 

# Create CTkEntry widget
prompt = ctk.CTkEntry(app, height=40, width=512, text_color="black", fg_color="white") 
prompt.place(x=10, y=10)

# Create CTkLabel widget
lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

# Load the Stable Diffusion model
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

# Function to generate image
def generate(): 
    with autocast(device): 
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 

# Create CTkButton widget
trigger = ctk.CTkButton(app, height=40, width=120, text_color="white", fg_color="blue", command=generate) 
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

# Run the application
app.mainloop()
