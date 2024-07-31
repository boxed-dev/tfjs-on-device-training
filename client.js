// async function trainAndSendModel(clientId) {
//     try {
//         console.log("Starting training process for client " + clientId + "...");

//         // Load and preprocess MNIST data for client1.csv or client2.csv
//         const response = await fetch(`mnist_data/${clientId}.csv`);
//         if (!response.ok) {
//             throw new Error(`Failed to load CSV file: ${response.statusText}`);
//         }

//         const csvData = await response.text();
//         console.log("CSV data loaded for client " + clientId);

//         // Parse CSV data
//         const data = csvData.trim().split('\n').slice(1).map(row => row.split(',').map(Number));
//         console.log("CSV data parsed for client " + clientId);

//         // Separate features and labels
//         const xs = data.map(row => row.slice(0, 784));
//         const ys = data.map(row => row[784]);

//         // Convert to tensors with correct types
//         const xTensor = tf.tensor2d(xs, [xs.length, 784]).reshape([-1, 28, 28, 1]);
//         const yTensor = tf.tensor1d(ys, 'float32');  // Ensure labels are float32

//         console.log("Data converted to tensors for client " + clientId);

//         // Create a simple model
//         const model = tf.sequential();
//         model.add(tf.layers.flatten({inputShape: [28, 28, 1]}));
//         model.add(tf.layers.dense({units: 128, activation: 'relu'}));
//         model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
//         model.compile({
//             optimizer: 'adam',
//             loss: 'sparseCategoricalCrossentropy',
//             metrics: ['accuracy']
//         });

//         console.log("Model created for client " + clientId);

//         // Train the model locally with an onEpochEnd callback
//         await model.fit(xTensor, yTensor, {
//             epochs: 10,
//             batchSize: 32,
//             callbacks: {
//                 onEpochEnd: async (epoch, logs) => {
//                     console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}, Accuracy = ${logs.acc}`);
//                 }
//             }
//         });

//         console.log("Model training completed for client " + clientId);

//         // Get model weights
//         const weights = model.getWeights().map(w => w.arraySync());

//         console.log("Model weights extracted for client " + clientId);

//         // Send model weights to the server
//         const updateResponse = await fetch('http://localhost:5000/update_model', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             body: JSON.stringify({weights: weights})
//         });

//         console.log("Model weights sent to the server for client " + clientId);

//         // Check the server response
//         if (!updateResponse.ok) {
//             throw new Error("Failed to send weights to the server");
//         }

//         console.log("Server response OK for client " + clientId);

//         // Optionally, get the updated global model
//         const response2 = await fetch('http://localhost:5000/get_model');
//         const globalWeights = await response2.json();
//         model.setWeights(globalWeights.weights.map(w => tf.tensor(w)));

//         console.log("Global model weights updated for client " + clientId);

//         // Evaluate the model
//         const result = await model.evaluate(xTensor, yTensor);
//         const [loss, accuracy] = await Promise.all(result); // Convert tensors to JavaScript values
//         console.log(`Client ${clientId} - Final Loss: ${loss}, Final Accuracy: ${accuracy}`);

//     } catch (error) {
//         console.error("Error during training process for client " + clientId + ":", error);
//     }
// }
async function trainAndSendModel(clientId) {
    try {
        console.log("Starting training process for client " + clientId + "...");

        // Load and preprocess MNIST data for client1.csv or client2.csv
        const response = await fetch(`mnist_data/${clientId}.csv`);
        if (!response.ok) {
            throw new Error(`Failed to load CSV file: ${response.statusText}`);
        }

        const csvData = await response.text();
        console.log("CSV data loaded for client " + clientId);

        // Parse CSV data
        const data = csvData.trim().split('\n').slice(1).map(row => row.split(',').map(Number));
        console.log("CSV data parsed for client " + clientId);

        // Separate features and labels
        const xs = data.map(row => row.slice(0, 784));
        const ys = data.map(row => row[784]);

        // Convert to tensors with correct types
        const xTensor = tf.tensor2d(xs, [xs.length, 784]).reshape([-1, 28, 28, 1]);
        const yTensor = tf.tensor1d(ys, 'float32');  // Ensure labels are float32

        console.log("Data converted to tensors for client " + clientId);

        // Create a simple model
        const model = tf.sequential();
        model.add(tf.layers.flatten({inputShape: [28, 28, 1]}));
        model.add(tf.layers.dense({units: 128, activation: 'relu'}));
        model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
        model.compile({
            optimizer: 'adam',
            loss: 'sparseCategoricalCrossentropy',
            metrics: ['accuracy']
        });

        console.log("Model created for client " + clientId);

        // Train the model locally with an onEpochEnd callback
        await model.fit(xTensor, yTensor, {
            epochs: 1,
            batchSize: 32,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}, Accuracy = ${logs.acc}`);
                }
            }
        });

        console.log("Model training completed for client " + clientId);

        // Get model weights
        const weights = model.getWeights().map(w => w.arraySync());

        console.log("Model weights extracted for client " + clientId);

        // Send model weights to the server
        const updateResponse = await fetch('http://localhost:5000/update_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({weights: weights})
        });

        console.log("Model weights sent to the server for client " + clientId);

        // Check the server response
        if (!updateResponse.ok) {
            throw new Error("Failed to send weights to the server");
        }

        console.log("Server response OK for client " + clientId);

        // Optionally, get the updated global model
        const response2 = await fetch('http://localhost:5000/get_model');
        const globalWeights = await response2.json();
        model.setWeights(globalWeights.weights.map(w => tf.tensor(w)));

        console.log("Global model weights updated for client " + clientId);

        // Evaluate the model with test data
        const testResponse = await fetch('mnist_data/test.csv');
        if (!testResponse.ok) {
            throw new Error(`Failed to load test CSV file: ${testResponse.statusText}`);
        }

        const testCsvData = await testResponse.text();
        console.log("Test CSV data loaded");

        // Parse test CSV data
        const testData = testCsvData.trim().split('\n').slice(1).map(row => row.split(',').map(Number));
        console.log("Test CSV data parsed");

        // Separate test features and labels
        const testXs = testData.map(row => row.slice(0, 784));
        const testYs = testData.map(row => row[784]);

        // Convert test data to tensors
        const testXTensor = tf.tensor2d(testXs, [testXs.length, 784]).reshape([-1, 28, 28, 1]);
        const testYTensor = tf.tensor1d(testYs, 'float32');

        console.log("Test data converted to tensors");

        // Evaluate the final global model on the test dataset
        const result = await model.evaluate(testXTensor, testYTensor);
        const [loss, accuracy] = await Promise.all(result); // Convert tensors to JavaScript values
        console.log(`Final Model - Loss: ${loss}, Accuracy: ${accuracy}`);

    } catch (error) {
        console.error("Error during training process for client " + clientId + ":", error);
    }
}

async function evaluateFinalModel() {
    try {
        console.log("Evaluating final model...");

        const response = await fetch('http://localhost:5000/evaluate_model', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Final model evaluation:", result);

        document.getElementById('result').textContent = `Final Model - Loss: ${result.loss}, Accuracy: ${result.accuracy}`;

    } catch (error) {
        console.error("Error during final model evaluation:", error);
        document.getElementById('result').textContent = `Error: ${error.message}`;
    }
}