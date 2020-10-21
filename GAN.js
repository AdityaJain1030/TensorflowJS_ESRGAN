"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_node_1 = require("@tensorflow/tfjs-node");
var utils_1 = require("./utils");
var IMAGE_PATH = "./images/test2.jpg";
var modelPath = "./models/model.json";
tfjs_node_1.loadGraphModel(tfjs_node_1.io.fileSystem(modelPath)).then(function (model) {
    var image = utils_1.preprocess(IMAGE_PATH);
    var fake = model.predict(image);
    utils_1.saveImage(fake.squeeze(), './SuperResolution.png');
});
