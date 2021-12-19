import streamlit as st
from PIL import Image
import cv2 
import pytesseract
import numpy as np #numpy
import tensorflow.keras  # Loading the keras models
from PIL import Image, ImageOps # pre-processing

# load the DL Model
model=tensorflow.keras.models.load_model('model.h5')

# Adding custom options

custom_config = r'--oem 3 --psm 6'
found=None

def load_image(image_file):
    im = Image.open(image_file)
    return im

def main():
    st.title("Plant Scanner")
    menu = ["Image"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Image":
        st.subheader("Image")
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        if image_file is not None:
            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
            st.write(file_details)
            # To View Uploaded Image
            st.image(load_image(image_file),width=250)
            # empty data array variable
            data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
            #open the image
            image=Image.open(image_file.name) # test.jpg
            # resize the image
            size=(224,224)
            image=ImageOps.fit(image,size,Image.ANTIALIAS)
            # convert this into numpy array
            image_array=np.asarray(image)
            # Normalise the Image - (0 to 255)
            normalise_image_array=(image_array.astype(np.float32)/127.0)-1
            # loading the image into the array
            data[0]=normalise_image_array
            # pass this data to model
            prediction=model.predict(data)
            print(prediction) # [[0.5,0.5,0.7,0.3]]
            # Decision Logic
            prediction=list(prediction[0])
            max_prediction=max(prediction)
            index_max=prediction.index(max_prediction)
            print(index_max)
            st.text("Expected Result")
            if(index_max==0):
                st.text("Identified Plant: senna auriculata")
                st.text("Uses: Diabetes, Joint & Muscle Pain, Eye Infection")
                st.text("Vitamins: Vitamin E, D, C")
            elif(index_max==1):
                st.text("Identified Plant: centella asiatica")
                st.text("Uses: Wound Healing,Stomach Aches, Reduce Weight Loss")
                st.text("Vitamins: Calcium, C, Potassium")
            elif(index_max==2):
                st.text("Identified Plant: Lemon Balm")
                st.text("Uses: Improves Concentration, Anti Viral Diseases,Ageing Brain Health")
                st.text("Vitamins: C, B")
            elif(index_max==3):
                st.text("Identified Plant: ocimum basilicum")
                st.text("Uses: Improves Oral Health,Relieves Stress,Treats Asthama")
                st.text("Vitamins: Calcium,Iron,Zinc")
            elif(index_max==4):
                st.text("Identified Plant: datura")
                st.text("Uses:Tooth Ache,Skin Diseases,Nutrients")
                st.text("Vitamins:C")
            elif(index_max==5):
                st.text("Identified Plant: catharanthus")
                st.text("Uses: Diabetes, High BP, Reduce Cough")
                st.text("Vitamins: B12,C")
            elif(index_max==6):
                st.text("Identified Plant: custured apple")
                st.text("Uses: High in Antioxidants,Eye Health,Healthy Heart")
                st.text("Vitamins: Calcium,Copper,Iron")
            elif(index_max==7):
                st.text("Identified Plant: salvia rosmarinus")
                st.text("Uses: Improve Brain Function,Stimulates Hair Growth,Joint Inflammation")
                st.text("Vitamins: Zinc, A, C")
            elif(index_max==8):
                st.text("Identified Plant: lavandula")
                st.text("Uses: Boost Sleep,Skin Health,Reduce Depression")
                st.text("Vitamins: C,E,A")
            elif(index_max==9):
                st.text("Identified Plant: hibiscus")
                st.text("Uses: Stop Hair Loss,Delay Gray Hair, Prevent Kidney Stones")
                st.text("Vitamins: C, Carotene, Iron")


if (__name__=="__main__"):
    main()
