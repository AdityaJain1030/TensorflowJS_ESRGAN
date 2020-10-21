# To Install the pretrained ESRGAN models for this repo
0) Make sure you have `python` installed. You can check this with `python -v`. If you dont have python installed, follow [this guide](https://datatofish.com/install-python/)

1) Create a Virtual Environment, as some Tensorflow packages may collide with each other. Guide on how to create python virtual environments [here](https://docs.python.org/3/library/venv.html). You can also use docker.

2) In your virtual environment, install tensorflowjs with `pip install tensorflowjs`. This will be our model converter

3) Once you have tensorflowjs installed, run `tensorflowjs_converter --input_format=tf_hub --signature_name serving_default --quantization_bytes 1 https://tfhub.dev/captain-pool/esrgan-tf2/1 ./models/`. This will copy the model and convert it into JSON. 

4) Copy the whole './models' folder to this directory. Your dir tree should look like this
```
├───docs
│   ├───images
│   │   ├───CNN.png
│   │   ├───gan_diagram.svg
│   │   ├───SuperResolution1.png
│   │   ├───SuperResolution2png
│   │
│   ├───examples.md
│   ├───how_it_works.md
│   ├───Model_Instalation.md
│   ├───readme.md
│
├───models
│   ├───group1-shard1of2.bin
│   ├───group1-shard2of2.bin
│   ├───model.json
│
├───node_modules
├───scripts
│   ├───GAN.py
│
├───GAN.ts
├───package.json
├───package-lock.json
```

**The script should now be able to access the model**