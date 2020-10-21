//Adaptation of ./scripts/GAN.py in typescript

//import libs
import {io, loadGraphModel, Tensor3D} from '@tensorflow/tfjs-node'
import {preprocess, saveImage} from './utils'

//path to image
const IMAGE_PATH = "./images/test2.jpg"
//model extracted from https://tfhub.dev/captain-pool/esrgan-tf2/1, using TFJS_Convertor Python Package
const modelPath = "./models/model.json"


loadGraphModel(io.fileSystem(modelPath)).then(model=>{
    const image = preprocess(IMAGE_PATH);
    const fake = <Tensor3D>model.predict(image)
    saveImage(fake.squeeze(), './SuperResolution.png')
})