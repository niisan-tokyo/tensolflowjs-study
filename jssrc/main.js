import * as tf from '@tensorflow/tfjs'

async function main() {
  const model = await tf.loadModel('http://localhost:8888/model/model.json')
  console.log(model)
}

main()
