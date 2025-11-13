'use client';

import NextImage from 'next/image';
import { ChangeEvent, useCallback, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import styles from './page.module.css';

const SIGNAL_LENGTH = 256;

type AnalysisResult = {
  probability: number;
  classification: 'STEMI Detected' | 'No STEMI Detected';
  stElevation: number;
  rPeakIndex: number;
  tPeakIndex: number;
  qualityScore: number;
};

const formatPercent = (value: number) =>
  `${(Math.min(Math.max(value, 0), 1) * 100).toFixed(1)}%`;

function movingAverage(values: number[], windowSize: number) {
  if (values.length === 0) {
    return [];
  }
  const half = Math.floor(windowSize / 2);
  const smoothed: number[] = new Array(values.length);
  for (let i = 0; i < values.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = i - half; j <= i + half; j++) {
      if (j >= 0 && j < values.length) {
        sum += values[j];
        count += 1;
      }
    }
    smoothed[i] = count > 0 ? sum / count : values[i];
  }
  return smoothed;
}

function resample(values: number[], length: number) {
  if (values.length === length) {
    return values.slice();
  }
  const result = new Array(length).fill(0);
  const factor = (values.length - 1) / (length - 1);
  for (let i = 0; i < length; i++) {
    const index = i * factor;
    const low = Math.floor(index);
    const high = Math.min(Math.ceil(index), values.length - 1);
    const weight = index - low;
    result[i] = values[low] * (1 - weight) + values[high] * weight;
  }
  return result;
}

function normalize(values: number[]) {
  const mean = values.reduce((acc, value) => acc + value, 0) / values.length;
  const centered = values.map((value) => value - mean);
  const maxAbs = centered.reduce(
    (max, value) => Math.max(max, Math.abs(value)),
    1e-6,
  );
  return centered.map((value) => value / maxAbs);
}

function buildSvgPath(values: number[], width: number, height: number) {
  if (values.length === 0) {
    return '';
  }
  let path = `M 0 ${height / 2}`;
  values.forEach((value, index) => {
    const x = (index / (values.length - 1)) * width;
    const y = height / 2 - value * (height * 0.4);
    path += ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
  });
  return path;
}

function computeMetrics(signal: number[]): Omit<AnalysisResult, 'probability' | 'classification'> {
  const smoothed = movingAverage(signal, 5);
  let rPeakIndex = 0;
  let rPeakValue = -Infinity;
  for (let i = 0; i < smoothed.length; i++) {
    if (smoothed[i] > rPeakValue) {
      rPeakValue = smoothed[i];
      rPeakIndex = i;
    }
  }

  let tPeakIndex = Math.min(rPeakIndex + 20, smoothed.length - 1);
  let tPeakValue = -Infinity;
  for (let i = rPeakIndex + 15; i < smoothed.length; i++) {
    if (smoothed[i] > tPeakValue) {
      tPeakValue = smoothed[i];
      tPeakIndex = i;
    }
  }

  const baselineWindow = smoothed.slice(
    Math.max(rPeakIndex - 60, 0),
    Math.max(rPeakIndex - 20, 1),
  );
  const baseline =
    baselineWindow.length > 0
      ? baselineWindow.reduce((acc, value) => acc + value, 0) /
        baselineWindow.length
      : 0;

  const stWindow = smoothed.slice(
    Math.min(rPeakIndex + 8, smoothed.length - 1),
    Math.min(rPeakIndex + 35, smoothed.length),
  );
  const stElevation =
    stWindow.length > 0
      ? stWindow.reduce((acc, value) => acc + value, 0) / stWindow.length -
        baseline
      : 0;

  const variance =
    smoothed.reduce((acc, value) => acc + Math.pow(value - baseline, 2), 0) /
    smoothed.length;
  const qualityScore = 1 / (1 + Math.exp(-10 * (1 - variance)));

  return {
    stElevation,
    rPeakIndex,
    tPeakIndex,
    qualityScore,
  };
}

async function extractSignalFromImage(
  dataUrl: string,
): Promise<{ raw: number[]; normalized: number[] }> {
  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Unable to read image'));
    img.src = dataUrl;
  });

  const maxWidth = 1024;
  const scale = Math.min(1, maxWidth / image.width);
  const width = Math.max(64, Math.round(image.width * scale));
  const height = Math.max(64, Math.round(image.height * scale));

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Canvas 2D context unavailable');
  }
  context.drawImage(image, 0, 0, width, height);

  const { data } = context.getImageData(0, 0, width, height);
  const columns: number[] = [];
  for (let x = 0; x < width; x++) {
    let weightedSum = 0;
    let weightTotal = 1e-6;
    for (let y = 0; y < height; y++) {
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const brightness = (r + g + b) / (3 * 255);
      const signalStrength = Math.max(0, 1 - brightness - 0.1);
      if (signalStrength > 0) {
        weightedSum += y * signalStrength;
        weightTotal += signalStrength;
      }
    }
    const centerY = weightedSum / weightTotal;
    if (Number.isNaN(centerY)) {
      columns.push(height / 2);
    } else {
      columns.push(centerY);
    }
  }

  const median =
    columns.slice().sort((a, b) => a - b)[Math.floor(columns.length / 2)] ||
    height / 2;
  const inverted = columns.map((value) => median - value);
  const resampled = resample(inverted, SIGNAL_LENGTH);
  const smoothed = movingAverage(resampled, 7);
  const normalizedSignal = normalize(smoothed);

  return { raw: smoothed, normalized: normalizedSignal };
}

export default function Home() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [extractedSignal, setExtractedSignal] = useState<number[] | null>(null);
  const [rawSignal, setRawSignal] = useState<number[] | null>(null);
  const [analyzing, setAnalyzing] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const ensureModel = useCallback(async () => {
    if (model) {
      return model;
    }
    setLoadingMessage('Loading STEMI detection model...');
    await tf.ready();
    const loadedModel = await tf.loadLayersModel('/model/model.json');
    setModel(loadedModel);
    setLoadingMessage('');
    return loadedModel;
  }, [model]);

  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const selectedFile = event.target.files?.[0];
      if (!selectedFile) {
        return;
      }
      if (!selectedFile.type.startsWith('image/')) {
        setError('Please upload a valid ECG image.');
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result;
        if (typeof result === 'string') {
          setImageSrc(result);
          setFileName(selectedFile.name);
          setAnalysis(null);
          setError(null);
          setExtractedSignal(null);
          setRawSignal(null);
        }
      };
      reader.readAsDataURL(selectedFile);
    },
    [],
  );

  const runAnalysis = useCallback(async () => {
    if (!imageSrc) {
      setError('Select an ECG image first.');
      return;
    }
    try {
      setAnalyzing(true);
      setError(null);
      setAnalysis(null);
      setLoadingMessage('Extracting ECG waveform...');
      const { raw, normalized } = await extractSignalFromImage(imageSrc);
      setRawSignal(raw);
      setExtractedSignal(normalized);

      const loadedModel = await ensureModel();
      setLoadingMessage('Running STEMI detection...');

      const probabilities = tf.tidy(() => {
        const tensor = tf.tensor(normalized, [1, SIGNAL_LENGTH, 1]);
        const prediction = loadedModel.predict(tensor) as tf.Tensor;
        return prediction.dataSync()[0];
      });

      const metrics = computeMetrics(normalized);
      const classification =
        probabilities >= 0.5 ? 'STEMI Detected' : 'No STEMI Detected';
      setAnalysis({
        ...metrics,
        probability: probabilities,
        classification,
      });
      setLoadingMessage('');
    } catch (analysisError) {
      const message =
        analysisError instanceof Error
          ? analysisError.message
          : 'Unable to analyze ECG image.';
      setError(message);
      setLoadingMessage('');
    } finally {
      setAnalyzing(false);
    }
  }, [ensureModel, imageSrc]);

  const dropZoneText = useMemo(() => {
    if (fileName) {
      return fileName;
    }
    return 'Click to upload or drop an ECG image (PNG, JPG, BMP)';
  }, [fileName]);

  const signalPath = useMemo(() => {
    if (!extractedSignal) {
      return '';
    }
    return buildSvgPath(extractedSignal, 600, 220);
  }, [extractedSignal]);

  const rawSignalPath = useMemo(() => {
    if (!rawSignal) {
      return '';
    }
    return buildSvgPath(rawSignal, 600, 220);
  }, [rawSignal]);

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        <header className={styles.hero}>
          <div>
            <h1>STEMI Detection From ECG Images</h1>
            <p>
              Upload a 12-lead ECG snapshot to extract the waveform and run a
              machine-learned classifier that screens for ST-elevation
              myocardial infarction.
            </p>
          </div>
          <button
            className={styles.secondaryButton}
            onClick={() => fileInputRef.current?.click()}
            type="button"
          >
            Upload ECG
          </button>
        </header>

        <section className={styles.card}>
          <div
            className={styles.dropZone}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              className={styles.hiddenInput}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />
            <p>{dropZoneText}</p>
          </div>
          {imageSrc && (
            <div className={styles.previewGrid}>
              <div className={styles.previewPane}>
                <h2>ECG Preview</h2>
                <NextImage
                  src={imageSrc}
                  alt="Uploaded ECG"
                  width={600}
                  height={400}
                  priority={false}
                  unoptimized
                  className={styles.previewImage}
                />
              </div>
              <div className={styles.previewPane}>
                <h2>Extracted Waveform</h2>
                {extractedSignal ? (
                  <svg
                    className={styles.waveform}
                    viewBox="0 0 600 220"
                    preserveAspectRatio="none"
                  >
                    <rect
                      x="0"
                      y="0"
                      width="600"
                      height="220"
                      className={styles.grid}
                    />
                    <path d={signalPath} className={styles.waveformPath} />
                  </svg>
                ) : (
                  <div className={styles.placeholder}>
                    <span>Run analysis to view waveform</span>
                  </div>
                )}
              </div>
              {rawSignal && (
                <div className={styles.previewPane}>
                  <h2>Pre-normalized Signal</h2>
                  <svg
                    className={styles.waveform}
                    viewBox="0 0 600 220"
                    preserveAspectRatio="none"
                  >
                    <rect
                      x="0"
                      y="0"
                      width="600"
                      height="220"
                      className={styles.grid}
                    />
                    <path d={rawSignalPath} className={styles.rawPath} />
                  </svg>
                </div>
              )}
            </div>
          )}
          <div className={styles.actions}>
            <button
              type="button"
              className={styles.primaryButton}
              onClick={runAnalysis}
              disabled={!imageSrc || analyzing}
            >
              {analyzing ? 'Analyzing…' : 'Analyze ECG'}
            </button>
            {loadingMessage && <span>{loadingMessage}</span>}
          </div>
        </section>

        {analysis && (
          <section className={styles.resultsCard}>
            <div className={styles.resultHeader}>
              <h2>{analysis.classification}</h2>
              <span className={styles.probabilityBadge}>
                {formatPercent(analysis.probability)}
              </span>
            </div>
            <p>
              The classifier estimates a probability of STEMI at{' '}
              {formatPercent(analysis.probability)} based on waveform morphology
              and ST-segment elevation.
            </p>
            <div className={styles.metricsGrid}>
              <div>
                <span className={styles.metricLabel}>ST Elevation</span>
                <span className={styles.metricValue}>
                  {(analysis.stElevation * 4).toFixed(2)} mm (relative)
                </span>
              </div>
              <div>
                <span className={styles.metricLabel}>R-Peak Index</span>
                <span className={styles.metricValue}>
                  {analysis.rPeakIndex}
                </span>
              </div>
              <div>
                <span className={styles.metricLabel}>T-Peak Index</span>
                <span className={styles.metricValue}>
                  {analysis.tPeakIndex}
                </span>
              </div>
              <div>
                <span className={styles.metricLabel}>Signal Quality</span>
                <span className={styles.metricValue}>
                  {(analysis.qualityScore * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            <footer className={styles.resultFooter}>
              <p>
                Disclaimer: This tool is for research prototyping only and is
                not intended for direct clinical decision making.
              </p>
            </footer>
          </section>
        )}

        {error && (
          <section className={styles.errorCard}>
            <h2>Analysis Error</h2>
            <p>{error}</p>
          </section>
        )}
      </div>
    </div>
  );
}
