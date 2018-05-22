import * as tf from '@tensorflow/tfjs'

const save_model = 'downloads://conv1d_test'
const load_model = 'http://localhost/models/conv1d_test.json'

// モデルの作成
const model = tf.sequential()
model.add(tf.layers.conv1d({filters: 64, kernelSize: 8, padding: 'same', inputShape: [64, 1], activation: 'relu'}))
model.add(tf.layers.maxPooling1d({poolSize: 2, padding: 'same'}))
model.add(tf.layers.conv1d({filters: 64, kernelSize: 8, padding: 'same', activation: 'relu'}))
model.add(tf.layers.maxPooling1d({poolSize: 2, padding: 'same'}))
model.add(tf.layers.conv1d({filters: 32, kernelSize: 8, padding: 'same', activation: 'relu'}))
model.add(tf.layers.conv1d({filters: 1, kernelSize: 8, padding: 'same', activation: 'tanh'}))

model.compile({optimizer: 'adam', loss: 'meanSquaredError'})

console.log(model)

// データの作成
let raw_data = []
for (let i = 0; i < 10000; i++) {
  raw_data.push((Math.sin(i) + Math.sin(3 * i) + Math.sin(10 * i) + Math.cos(5 * i) + Math.cos(7 * i)) / 5)
}

let xs = []
let ys = []

for (let j = 0; j < 1920; j++) {
  xs.push(raw_data.slice(j, j+64))
  ys.push(raw_data.slice(j+64, j+80))
}

const train_X = tf.tensor2d(xs, [1920, 64]).reshape([1920, 64, 1])
const train_Y = tf.tensor2d(ys, [1920, 16]).reshape([1920, 16, 1])

// 訓練
async function train() {
  const start = new Date()
  await model.fit(train_X, train_Y, {epochs: 10, validationSplit: 0.1})
  const saveResult = await model.save(save_model)
  model.predict(tf.tensor3d(raw_data.slice(9600, 9664), [1, 64, 1])).print()
  console.log(raw_data.slice(9664, 9680))
  const end = new Date()
  console.log((end - start) / 1000)
}

train()
