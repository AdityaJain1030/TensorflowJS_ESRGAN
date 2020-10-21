## To reuse this code
A guide on how to reuse this code for your own projects!

### Typescript/Javascript
Start by going to [index.ts](https://github.com/FirstPotatoMan/TensorflowJS_GAN/blob/master/src/index.ts). You should see 
```ts
import {preprocess, saveImage, loadModel, predictFromImage} from './utils'
const image = "./images/test1.png"
const model_sketch = "./models/model.json"
;(async(IMAGE_PATH:string, MODEL_PATH:string, EXPORT_PATH?)=>{
    const image = preprocess(IMAGE_PATH)
    const model = await loadModel(MODEL_PATH)
    const fake = predictFromImage(model, image)
    saveImage(fake, EXPORT_PATH)
})(image, model_sketch)
```
The first 3 lines import the helper functions, and set the paths to the image and model. The path to the image and model can be changed according to where they are in your directory. The next snippet creates an [IIFE](https://developer.mozilla.org/en-US/docs/Glossary/IIFE), that loads the model, converts the image to something the model can use, predicts the enlarged image, and saves the new image. 

`preprocess` takes in a `string`, as the path to the image. It returns a `Tensor4D` with shape `(1, R, G, B)`

`loadModel` also takes in a `string`, this time with the path to the model. It returns a `GraphModel`

`predictFromImage` takes in a `GraphModel` as its first arguement, and a `Tensor4D` with shape `(1, R, G, B)` as its second arguement. It returns an enlarged `Tensor4D` with shape `(1, R, G, B)`. 

`saveImage` takes in a `Tensor4D` with shape `(1, R, G, B)`, and a `string` as the path to export the image. It returns nothing, but saves the image to the path specified

These 4 methods can be imported from [utils.ts](https://github.com/FirstPotatoMan/TensorflowJS_GAN/blob/master/src/utils.ts)


### Python
See this [Jupyter Notebook](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/image_enhancing.ipynb)
