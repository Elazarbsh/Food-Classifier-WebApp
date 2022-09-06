const express = require('express');
const path = require('path');
const fileupload = require("express-fileupload");
const resizeImg = require('resize-image-buffer');
const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const tfnode = require('@tensorflow/tfjs-node');
const fs = require('fs');
const app = express();

const port = 3000;

app.use(express.static('public'))
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(fileupload());

const labels = ['Hotdog', 'Hamburger', 'Pizza'];

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
})

app.post('/upload', (req, res) => {
    const file = req.files.file;
    resizeImg(file.data, {
        width: 128,
        height: 128,
    }).then((img) => {
        const tfimage = tfnode.node.decodeImage(img);
        tf.loadLayersModel("file://model.json").then((model) => {
            var normalizedImage = tfimage.div(tf.scalar(255));
            normalizedImage = normalizedImage.reshape([-1, 128, 128, 3]);
            const prediction = model.predict(normalizedImage);
            const predictionIndex = prediction.as1D().argMax().dataSync();
            result = {
                label: labels[predictionIndex],
                accuracy: parseFloat((prediction.dataSync()[predictionIndex])).toFixed(3) * 100
            };
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(result));
        });
    });
});

app.listen(process.env.PORT || port);