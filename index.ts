//Adaptation of ./python/main.py in typescript

//import helpers
import {preprocess, saveImage, loadModel, predictFromImage} from './utils'

//path to image
const image = "./images/test1.png"
//model extracted from https://tfhub.dev/captain-pool/esrgan-tf2/1, using TFJS_Convertor Python Package
const model_sketch = "./models/model.json"

;(async(IMAGE_PATH:string, MODEL_PATH:string, EXPORT_PATH?)=>{
    //preprocess image
    const image = preprocess(IMAGE_PATH)
    //load model
    const model = await loadModel(MODEL_PATH)
    //predict upscaled image
    const fake = predictFromImage(model, image)
    //save epscaled image
    saveImage(fake, EXPORT_PATH)

})(image, model_sketch)