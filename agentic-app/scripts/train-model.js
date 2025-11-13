/* eslint-disable @typescript-eslint/no-require-imports */

const fs = require('fs');
const path = require('path');
const util = require('util');

if (typeof util.isNullOrUndefined !== 'function') {
  util.isNullOrUndefined = (value) => value === null || value === undefined;
}

const tf = require('@tensorflow/tfjs-node');

const SAMPLE_COUNT = 4000;
const SIGNAL_LENGTH = 256;
const TRAIN_EPOCHS = 30;
const BATCH_SIZE = 64;

function gaussian(x, mu, sigma, amplitude) {
  const exponent = -Math.pow(x - mu, 2) / (2 * Math.pow(sigma, 2));
  return amplitude * Math.exp(exponent);
}

function generateHeartbeatSample(isStemi) {
  const baselineSlope = (Math.random() - 0.5) * 0.02;
  const baselineNoise = () => (Math.random() - 0.5) * 0.05;
  const waveform = [];

  const pAmplitude = 0.15 + Math.random() * 0.05;
  const qAmplitude = -(0.12 + Math.random() * 0.04);
  const rAmplitude = 1 + Math.random() * 0.2;
  const sAmplitude = -(0.25 + Math.random() * 0.05);
  const tAmplitude = 0.3 + Math.random() * 0.1;

  const pCenter = 0.18 + Math.random() * 0.02;
  const qCenter = 0.25 + Math.random() * 0.01;
  const rCenter = 0.28 + Math.random() * 0.01;
  const sCenter = 0.30 + Math.random() * 0.015;
  const tCenter = 0.55 + Math.random() * 0.05;

  const stElevation = isStemi ? 0.15 + Math.random() * 0.1 : 0;
  const stDecay = isStemi ? 0.05 + Math.random() * 0.03 : 0;

  for (let i = 0; i < SIGNAL_LENGTH; i++) {
    const x = i / SIGNAL_LENGTH;
    const baseline = baselineSlope * x + baselineNoise();

    const pWave = gaussian(x, pCenter, 0.015, pAmplitude);
    const qWave = gaussian(x, qCenter, 0.008, qAmplitude);
    const rWave = gaussian(x, rCenter, 0.01, rAmplitude);
    const sWave = gaussian(x, sCenter, 0.01, sAmplitude);
    const tWave = gaussian(x, tCenter, 0.04, tAmplitude);

    let stSegment = 0;
    if (x > sCenter && x < tCenter) {
      const progress = (x - sCenter) / (tCenter - sCenter);
      stSegment = stElevation * Math.exp(-stDecay * progress);
    }

    const noise = (Math.random() - 0.5) * 0.04;
    waveform.push(
      baseline + pWave + qWave + rWave + sWave + stSegment + tWave + noise,
    );
  }

  return waveform;
}

function generateDataset() {
  const samples = [];

  for (let i = 0; i < SAMPLE_COUNT; i++) {
    const isStemi = i < SAMPLE_COUNT / 2;
    const sample = generateHeartbeatSample(isStemi);
    samples.push({ sample, label: isStemi ? 1 : 0 });
  }

  for (let i = samples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = samples[i];
    samples[i] = samples[j];
    samples[j] = tmp;
  }

  const xs = samples.map((item) => item.sample);
  const ys = samples.map((item) => item.label);

  const xTensor = tf.tensor(xs, [SAMPLE_COUNT, SIGNAL_LENGTH, 1]);
  const yTensor = tf.tensor(ys, [SAMPLE_COUNT, 1]);

  return { xs: xTensor, ys: yTensor };
}

async function train() {
  const { xs, ys } = generateDataset();

  const model = tf.sequential();
  model.add(
    tf.layers.conv1d({
      inputShape: [SIGNAL_LENGTH, 1],
      filters: 16,
      kernelSize: 5,
      strides: 1,
      activation: 'relu',
      padding: 'same',
    }),
  );
  model.add(tf.layers.maxPooling1d({ poolSize: 2 }));
  model.add(
    tf.layers.conv1d({
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    }),
  );
  model.add(tf.layers.maxPooling1d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  await model.fit(xs, ys, {
    epochs: TRAIN_EPOCHS,
    batchSize: BATCH_SIZE,
    validationSplit: 0.2,
    verbose: 1,
  });

  const modelDir = path.join(__dirname, '..', 'public', 'model');
  if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir, { recursive: true });
  }

  await model.save(`file://${modelDir}`);

  xs.dispose();
  ys.dispose();
  model.dispose();

  console.log(`Model saved to ${modelDir}`);
}

train().catch((err) => {
  console.error(err);
  process.exit(1);
});
