let model;

// Inisialisasi model
async function init() {
    model = await tf.loadLayersModel('http://127.0.0.1:8887/model/model.json');
    console.log('Model loaded successfully');
}

// Load the TensorFlow JS model
async function loadModel() {
    if (!model) {
        model = await tf.loadLayersModel('http://127.0.0.1:8887/model/model.json');
    }
    return model;
}

// Read the input image and perform image classification
async function classifyImage() {
    const imageInput = document.getElementById('imageInput').files[0];
    if (!imageInput) {
        alert("Please upload an image first.");
        return;
    }
    
    const image = await readImage(imageInput);
    
    // Resize and normalize the image
    const tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([255, 255])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();
    
    const model = await loadModel();
    const output = model.predict(tensor);
    const result = await output.data();
    const className = getClassNames(result);
    
    document.getElementById('result').innerHTML = `Spesies Ikan: ${className}`;
}

// Read an image file and return an image element
function readImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const img = new Image();
            img.src = reader.result;
            img.onload = () => resolve(img);
            img.onerror = reject;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Get the class names from the output tensor
function getClassNames(output) {
    const classNames = ['Polyprion americanus', 'Trigloporus lastoviza', 'Anthias anthias'];
    const maxIndex = output.indexOf(Math.max(...output));
    return classNames[maxIndex];
}

// Add event listener to the classify button
document.getElementById('classifyButton').addEventListener('click', classifyImage);

// Inisialisasi model saat aplikasi dimulai
init();
