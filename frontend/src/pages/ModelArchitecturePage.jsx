import React from 'react';
import { 
  Cpu, 
  Layers, 
  FileAudio, 
  Sparkles, 
  BarChart3, 
  CheckCircle2, 
  GitBranch,
  Settings2
} from 'lucide-react';

export default function ModelArchitecturePage() {
  const stages = [
    {
      number: "01",
      title: "Audio Input & Acoustic Preprocessing",
      subtitle: "Signal Standardization & Dynamic Energy Scaling",
      description: "Voice signals recorded across diverse microphones and environments exhibit varying sampling rates, stereo channels, and recording levels. Our preprocessing pipeline standardizes every recording to 16,000 Hz single-channel (mono) PCM audio, removes DC offsets, eliminates leading/trailing silence, and applies peak amplitude normalization to prevent clipping and balance vocal energy.",
      details: [
        { label: "Target Frequency", val: "16,000 Hz (16 kHz)" },
        { label: "Channel Format", val: "Mono (1-channel)" },
        { label: "Normalization", val: "Peak amplitude normalization" },
        { label: "Acceptable Formats", val: "WAV, MP3, FLAC, M4A" }
      ]
    },
    {
      number: "02",
      title: "Wav2Vec 2.0 Deep Speech Transformer",
      subtitle: "Self-Supervised Acoustic Representation Learning",
      description: "Unlike handcrafted acoustic features (MFCC, Jitter, Shimmer) that capture only shallow spectral envelopes, Wav2Vec 2.0 utilizes a multi-layer Convolutional Feature Encoder coupled with 12 bidirectional Transformer encoder blocks. Pretrained on thousands of hours of speech, the model captures deep, contextualized temporal and harmonic dynamics of human phonation.",
      details: [
        { label: "Base Architecture", val: "facebook/wav2vec2-base" },
        { label: "Feature Encoder", val: "7-layer temporal CNN" },
        { label: "Context Network", val: "12 Transformer blocks (768 hidden units, 8 heads)" },
        { label: "Raw Output Tensor", val: "Shape (Batch, Time Frames, 768)" }
      ]
    },
    {
      number: "03",
      title: "Temporal Global Mean Pooling",
      subtitle: "Dimension Reduction & Fixed-Length Vectorization",
      description: "Audio recordings naturally vary in duration, producing variable-length hidden state sequences across time frames. To bridge variable-length deep speech representations with tabular machine learning classifiers, we apply global temporal mean pooling along the sequence length dimension. This condenses the sequential representations into a robust, translation-invariant 768-dimensional feature embedding vector.",
      details: [
        { label: "Pooling Function", val: "z = (1/T) * Σ h_t" },
        { label: "Input Shape", val: "T × 768 variable frames" },
        { label: "Output Embedding", val: "Fixed 768-dimensional vector" },
        { label: "Invariance Property", val: "Temporal shift & length invariance" }
      ]
    },
    {
      number: "04",
      title: "Random Forest Multi-Class Classifier",
      subtitle: "Ensemble Decision Trees with Balanced Class Weighting",
      description: "The 768-dimensional speech representations are fed into a Random Forest ensemble consisting of 100 decorrelated decision trees constructed via bootstrap aggregating (bagging). Random feature subspacing prevents individual acoustic dimensions from dominating splits. Balanced class weighting automatically scales tree loss inversely proportional to class frequencies, directly mitigating medical class imbalance without synthetic data corruption.",
      details: [
        { label: "Estimators", val: "100 Decision Trees" },
        { label: "Split Criterion", val: "Gini Impurity" },
        { label: "Class Weighting", val: "Balanced (w_c = N / (C * n_c))" },
        { label: "Probability Mode", val: "Normalized Tree Vote Aggregation" }
      ]
    },
    {
      number: "05",
      title: "Multi-Class Laryngeal Disorder Output",
      subtitle: "Confidence Scores & Posterior Probability Distribution",
      description: "The forest generates probabilistic outputs across all configured target classes. The class with the highest ensemble vote is designated as the primary classification result, accompanied by a percentage confidence rating and a complete distribution of posterior probabilities for clinical inspection.",
      details: [
        { label: "Target Categories", val: "6 laryngeal diagnostic classes" },
        { label: "Output Format", val: "Predicted Class, Confidence %, Class Probabilities" },
        { label: "Safety Classification", val: "Screening support (Non-diagnostic)" }
      ]
    }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
      <div>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc', marginBottom: '0.5rem' }}>
          Machine Learning System Architecture
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '0.925rem' }}>
          Theoretical principles and architectural components of the Wav2Vec 2.0 + Random Forest pipeline.
        </p>
      </div>

      {/* Visual Pipeline Block */}
      <div className="glass-card" style={{
        background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.9) 0%, rgba(10, 15, 29, 0.95) 100%)',
        padding: '2.5rem 1.5rem'
      }}>
        <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc', textAlign: 'center', marginBottom: '2rem' }}>
          End-to-End Information Processing Flow
        </h3>

        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1rem',
          maxWidth: '520px',
          margin: '0 auto'
        }}>
          {[
            { name: "Audio Recording", sub: "User voice input (.wav, .mp3, .flac)" },
            { name: "Preprocessing", sub: "16 kHz resampling, mono conversion, peak normalization" },
            { name: "Wav2Vec 2.0 Encoder", sub: "facebook/wav2vec2-base self-supervised speech transformer" },
            { name: "Deep Speech Representation", sub: "Contextualized hidden states (Batch, T, 768)" },
            { name: "Mean / Temporal Pooling", sub: "Collapse temporal dimension to fixed 768-D vector" },
            { name: "Random Forest Classifier", sub: "100-tree ensemble with balanced class weighting" },
            { name: "Multi-Class Prediction", sub: "Predicted class, confidence %, posterior probabilities" }
          ].map((node, i, arr) => (
            <React.Fragment key={i}>
              <div style={{
                width: '100%',
                backgroundColor: i === arr.length - 1 ? 'rgba(2, 132, 199, 0.2)' : '#1e293b',
                border: i === arr.length - 1 ? '1px solid #38bdf8' : '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: '10px',
                padding: '1rem 1.25rem',
                textAlign: 'center'
              }}>
                <div style={{ fontWeight: 700, fontSize: '0.95rem', color: i === arr.length - 1 ? '#38bdf8' : '#f8fafc' }}>
                  {node.name}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.2rem' }}>
                  {node.sub}
                </div>
              </div>
              {i < arr.length - 1 && (
                <div style={{ color: '#38bdf8', fontSize: '1.25rem', fontWeight: 800 }}>
                  &darr;
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Deep-Dive Stage Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <h2 style={{ fontSize: '1.35rem', fontWeight: 700, color: '#f8fafc' }}>
          Detailed Architectural Breakdown
        </h2>

        {stages.map((stg, idx) => (
          <div key={idx} className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span className="mono" style={{
                fontSize: '0.85rem',
                fontWeight: 700,
                color: '#38bdf8',
                backgroundColor: 'rgba(2, 132, 199, 0.15)',
                padding: '0.25rem 0.6rem',
                borderRadius: '6px'
              }}>
                STAGE {stg.number}
              </span>
              <div>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#ffffff' }}>
                  {stg.title}
                </h3>
                <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                  {stg.subtitle}
                </div>
              </div>
            </div>

            <p style={{ fontSize: '0.875rem', color: '#cbd5e1', lineHeight: '1.6' }}>
              {stg.description}
            </p>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '0.75rem',
              marginTop: '0.5rem',
              paddingTop: '0.75rem',
              borderTop: '1px solid rgba(255, 255, 255, 0.05)'
            }}>
              {stg.details.map((item, dIdx) => (
                <div key={dIdx} style={{ backgroundColor: '#0a0f1d', padding: '0.65rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <span style={{ color: '#64748b', fontSize: '0.75rem', display: 'block' }}>{item.label}</span>
                  <span className="mono" style={{ color: '#f8fafc', fontSize: '0.825rem', fontWeight: 600 }}>{item.val}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
