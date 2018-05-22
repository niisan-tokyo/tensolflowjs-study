import * as tf from '@tensorflow/tfjs'

const model = tf.sequential()
model.add(tf.layers.dense({units: 32, inputShape: [1]}));
model.add(tf.layers.dense({units: 64}));
model.add(tf.layers.dense({units: 1}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);// 2n + 1

async function main() {
  await model.fit(xs, ys, {epochs: 10});

  // Run inference with predict().
  model.predict(tf.tensor2d([[5]], [1, 1])).print();
  model.save('localstorage://test')
}

main()
