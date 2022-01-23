import * as fs from 'fs-extra';

import * as tf from '@tensorflow/tfjs-node';

export const createModel = (name: string, units = 3) => {
  const model = tf.sequential({
    name,
    layers: [
      tf.layers.dense({
        inputShape: [512],
        activation: 'sigmoid',
        units,
      }),
      tf.layers.dense({
        inputShape: [2],
        activation: 'sigmoid',
        units,
      }),
      tf.layers.dense({
        inputShape: [2],
        activation: 'sigmoid',
        units,
      }),
    ],
  });

  model.compile({
    optimizer: tf.train.adam(0.06), // This is a standard compile config
    loss: 'meanSquaredError',
    metrics: ['accuracy'],
  });

  return model;
};

export const trainModel = async (
  model: tf.Sequential,
  trainingData: tf.Tensor,
  testingData: tf.Tensor,
  outputData: tf.Tensor
) => {
  await model.fit(trainingData, outputData, { epochs: 200 });
  const result = model.predict(testingData);
  if (Array.isArray(result)) {
    result.forEach((r) => r.print());
  } else {
    result.print();
  }
};

export const saveModel = (model: tf.Sequential, path: string) => {
  fs.ensureDirSync(path);
  model.save(`file://${path}`);
};
