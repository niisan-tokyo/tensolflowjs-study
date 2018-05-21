import numpy as np
import json

# 時系列データの作成
timeline = np.arange(10000)
def sinnp(n, line):
    return np.sin(line * n / 100)

def cosnp(n, line):
    return np.cos(line * n / 100)

raw_data = (sinnp(1, timeline) + sinnp(3, timeline) + sinnp(10, timeline) + cosnp(5, timeline) + cosnp(7, timeline)) / 5
raw_data = raw_data + (np.random.rand(len(timeline)) * 0.1)# ノイズ項

# 時系列の元データの作成
input_data = []
output_data = []
for n in range(10000-80):
    input_data.append(raw_data[n:n+64])
    output_data.append(raw_data[n+64:n+80])

input_data = np.array(input_data)
output_data = np.array(output_data)

train_X = np.reshape(input_data, (-1, 64, 1))
train_Y = np.reshape(output_data, (-1, 16, 1))

f = open('/data/input/train_X.json', 'w')
f.write(json.dumps(train_X.tolist()))
f.close()

f = open('/data/input/train_Y.json', 'w')
f.write(json.dumps(train_Y.tolist()))
f.close()
