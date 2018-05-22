import * as tf from '@tensorflow/tfjs'

const save_model = 'downloads://conv1d_test'
const load_model = 'http://localhost:8888/model/conv1d_test.json'

// データの作成
let raw_data = []
for (let i = 0; i < 10000; i++) {
  raw_data.push((Math.sin(i) + Math.sin(3 * i) + Math.sin(10 * i) + Math.cos(5 * i) + Math.cos(7 * i)) / 5)
}

let xs = []
let ys = []

for (let j = 0; j < 9920; j++) {
  xs.push(raw_data.slice(j, j+64))
  ys.push(raw_data.slice(j+64, j+80))
}

const train_X = tf.tensor2d(xs, [9920, 64]).reshape([9920, 64, 1])
const train_Y = tf.tensor2d(ys, [9920, 16]).reshape([9920, 16, 1])

// 訓練
async function train() {
  const start = new Date()
  const model = await tf.loadModel(load_model)
  console.log(model)
  model.compile({optimizer: 'adam', loss: 'meanSquaredError'})
  await model.fit(train_X, train_Y, {epochs: 10, validationSplit: 0.1})

  console.log(raw_data.slice(9664, 9680))
  const end = new Date()
  console.log((end - start) / 1000)
  const saveResult = await model.save(save_model)
}

train()
