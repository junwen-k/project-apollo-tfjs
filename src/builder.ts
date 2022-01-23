import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs-node';

import feedbacksInsightful from './dataset/feedback-category-insightful-training.json';
import feedbacksLowQuality from './dataset/feedback-category-low-quality-training.json';
import feedbacksNonInsightful from './dataset/feedback-category-non-insightful-training.json';
import feedbackTests from './dataset/feedback-category-test.json';
import { createModel, saveModel, trainModel } from './model';

const build = async () => {
  console.log('Creating model');
  const model = createModel('feedback-category-classifier-model');

  const encodeData = async (data: typeof feedbacks) => {
    const sentences = data.map((comment) => comment.text.toLowerCase());
    const embeddings = await (await use.load()).embed(sentences);
    return embeddings;
  };

  const feedbacks = [
    // Feedbacks that are easily understandable, with a clear objective on what to be improved.
    ...feedbacksInsightful,
    // Feedbacks that are understandable, but does not provide additional information to make it very useful.
    ...feedbacksNonInsightful,
    // Feedbacks that are either too short or does not have any meaningful message.
    ...feedbacksLowQuality,
  ];

  const [trainingData, testingData] = await Promise.all([
    encodeData(feedbacks),
    encodeData(feedbackTests),
  ]);

  const outputData = tf.tensor(
    feedbacks.map((feedback) => [
      feedback.category === 'insightful' ? 1 : 0,
      feedback.category === 'non-insightful' ? 1 : 0,
      feedback.category === 'low-quality' ? 1 : 0,
    ])
  );
  console.log('Training model');
  await trainModel(model, trainingData, testingData, outputData);

  console.log('Saving model');
  saveModel(model, 'dist/models/tfjs-model-feedback-category-classifier');
  model.summary();
};

export default build;
