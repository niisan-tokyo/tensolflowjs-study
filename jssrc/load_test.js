import * as tf from '@tensorflow/tfjs'

const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

async function main() {
  console.log('model load start')
  const model = await tf.loadModel('localstorage://test')
  console.log('model load complete')
  model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
  console.log('model train start')
  await model.fit(xs, ys, {epochs: 100});
  console.log('model train complete')
  // Run inference with predict().
  model.predict(tf.tensor2d([[5]], [1, 1])).print();
  model.save('localstorage://test')
}

main()
