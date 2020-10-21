import { Tensor3D, cast, node, expandDims, clipByValue, Tensor4D } from '@tensorflow/tfjs-node'
import * as fs from 'fs'

//Image Preproccesser Util (get it ready for NN)
export const preprocess = (path: string) => {
    //get image
    const image = fs.readFileSync(path)
    //load it with the new decodeImage Function
    const hr_image = <Tensor3D>node.decodeImage(new Uint8Array(image), 3)

    //change the type to 'float32' from 'int32'
    const castImage = cast(hr_image, 'float32')

    // expand the dims for multiple batch cpabilities
    return <Tensor4D>expandDims(castImage, 0)
}

//Image Saver Util (writing NN output to image)
export const saveImage = async (image: Tensor3D, filePath: string) => {
    //clip all outputs to range [0, 255] (color)
    const clipped: Tensor3D = clipByValue(image,0, 255)
    //encode PNG and write to 
    const encodedPng = await node.encodePng(clipped)
    fs.writeFileSync(filePath, encodedPng)

    console.log(`Wrote Super Resolution Image to ${filePath}!`)
}

