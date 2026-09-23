import React from 'react';
import { 
  CheckCircle, 
  AlertTriangle, 
  BarChart2, 
  Layers, 
  Cpu, 
  FileAudio, 
  RotateCcw,
  ArrowRight
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Cell 
} from 'recharts';

export default function PredictionPage({ predictionResult, setActivePage }) {
  if (!predictionResult) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        <div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc', marginBottom: '0.5rem' }}>
            Multi-Class Voice Disorder Prediction
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '0.925rem' }}>
            Diagnostic prediction output derived from pretrained Wav2Vec 2.0 speech representations and Random Forest classifier.
          </p>
        </div>

        <div className="glass-card" style={{
          textAlign: 'center',
          padding: '4rem 2rem',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.25rem'
        }}>
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <BarChart2 size={32} color="#64748b" />
          </div>

          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.5rem' }}>
              No Voice Recording Evaluated Yet
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.875rem', maxWidth: '500px', margin: '0 auto' }}>
              To view classification predictions and class-wise probability distributions, please upload an audio recording in the Voice Analysis section.
            </p>
          </div>

          <button 
            className="btn-primary"
            onClick={() => setActivePage('analysis')}
            style={{ marginTop: '0.5rem' }}
          >
            <span>Go to Voice Analysis</span>
            <ArrowRight size={16} />
          </button>
        </div>
      </div>
    );
  }

  const {
    predicted_class,
    display_name,
    confidence,
    class_probabilities = {},
    category,
    description,
    audio_metadata,
    filename,
    feature_dim
  } = predictionResult;

  // Convert class probabilities to chart data
  const chartData = Object.entries(class_probabilities).map(([cls, prob]) => ({
    name: cls.replace(/_/g, ' '),
    rawName: cls,
    probability: Math.round(prob * 1000) / 10, // percentage e.g. 87.2%
    isPredicted: cls === predicted_class
  })).sort((a, b) => b.probability - a.probability);

  const confidencePercentage = Math.round(confidence * 100);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc', marginBottom: '0.5rem' }}>
          Classification Results & Posterior Probabilities
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '0.925rem' }}>
          Acoustic classification for audio file: <strong style={{ color: '#38bdf8' }}>{filename || 'recording.wav'}</strong>
        </p>
      </div>

      {/* Experimental Research Disclaimer */}
      <div style={{
        backgroundColor: 'rgba(245, 158, 11, 0.08)',
        border: '1px solid rgba(245, 158, 11, 0.3)',
        borderRadius: '10px',
        padding: '1rem 1.25rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem'
      }}>
        <AlertTriangle size={20} color="#f59e0b" style={{ flexShrink: 0 }} />
        <span style={{ fontSize: '0.825rem', color: '#cbd5e1' }}>
          <strong>Notice:</strong> Experimental research prototype with a small dataset (18 speakers) — strictly for academic evaluation, not medical diagnosis.
        </span>
      </div>

      {/* Result Card Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
        gap: '1.5rem'
      }}>
        {/* Primary Classification Result */}
        <div className="glass-card" style={{
          background: 'linear-gradient(145deg, rgba(19, 29, 54, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%)',
          borderColor: predicted_class === 'Normal' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(2, 132, 199, 0.4)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
            <span className={predicted_class === 'Normal' ? 'badge badge-green' : 'badge badge-blue'}>
              {category || 'Classification Outcome'}
            </span>
            <span className="mono" style={{ fontSize: '0.75rem', color: '#64748b' }}>
              Inference Mode
            </span>
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <span style={{ fontSize: '0.85rem', color: '#94a3b8', display: 'block', marginBottom: '0.25rem' }}>
              Predicted Class:
            </span>
            <h2 style={{
              fontSize: '2rem',
              fontWeight: 800,
              color: predicted_class === 'Normal' ? '#34d399' : '#38bdf8',
              letterSpacing: '-0.02em',
              lineHeight: '1.1'
            }}>
              {display_name || predicted_class}
            </h2>
          </div>

          {/* Confidence Meter */}
          <div style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Model Confidence:</span>
              <span className="mono" style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>
                {confidencePercentage}%
              </span>
            </div>

            <div style={{
              width: '100%',
              height: '10px',
              backgroundColor: '#0a0f1d',
              borderRadius: '9999px',
              overflow: 'hidden',
              border: '1px solid rgba(255,255,255,0.08)'
            }}>
              <div style={{
                height: '100%',
                width: `${confidencePercentage}%`,
                background: predicted_class === 'Normal' 
                  ? 'linear-gradient(90deg, #10b981 0%, #34d399 100%)' 
                  : 'linear-gradient(90deg, #0284c7 0%, #38bdf8 100%)',
                borderRadius: '9999px',
                transition: 'width 0.8s ease'
              }} />
            </div>
          </div>

          <p style={{ fontSize: '0.825rem', color: '#94a3b8', lineHeight: '1.5' }}>
            {description}
          </p>
        </div>

        {/* Feature & Technical Vector Info */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc' }}>
            Acoustic Representation Parameters
          </h3>

          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
            fontSize: '0.85rem'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <span style={{ color: '#94a3b8' }}>Feature Extractor:</span>
              <span className="mono" style={{ color: '#38bdf8' }}>facebook/wav2vec2-base</span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <span style={{ color: '#94a3b8' }}>Embedding Dimensions:</span>
              <span className="mono" style={{ color: '#f8fafc' }}>{feature_dim || 768} features</span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <span style={{ color: '#94a3b8' }}>Pooling Operation:</span>
              <span className="mono" style={{ color: '#f8fafc' }}>Temporal Mean Pooling</span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <span style={{ color: '#94a3b8' }}>Audio Duration:</span>
              <span className="mono" style={{ color: '#f8fafc' }}>{audio_metadata?.duration_seconds || 0}s</span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0' }}>
              <span style={{ color: '#94a3b8' }}>Sampling Frequency:</span>
              <span className="mono" style={{ color: '#34d399' }}>16,000 Hz</span>
            </div>
          </div>

          <div style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
            <button 
              className="btn-secondary"
              onClick={() => setActivePage('analysis')}
              style={{ width: '100%', fontSize: '0.85rem' }}
            >
              <RotateCcw size={16} />
              <span>Analyze Another Recording</span>
            </button>
          </div>
        </div>
      </div>

      {/* Class Probability Distribution Chart */}
      <div className="glass-card">
        <div style={{ marginBottom: '1.25rem' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc' }}>
            Class Probability Distribution
          </h3>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
            Posterior class probabilities generated by the Random Forest ensemble across all evaluated classes
          </p>
        </div>

        <div style={{ height: '280px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 100, bottom: 10 }}
            >
              <XAxis 
                type="number" 
                domain={[0, 100]} 
                unit="%" 
                stroke="#64748b" 
                fontSize={12} 
              />
              <YAxis 
                type="category" 
                dataKey="name" 
                stroke="#94a3b8" 
                fontSize={12} 
                width={120}
              />
              <Tooltip 
                formatter={(value) => [`${value}%`, 'Probability']}
                contentStyle={{ 
                  backgroundColor: '#0f172a', 
                  borderColor: 'rgba(255,255,255,0.1)', 
                  borderRadius: '8px', 
                  color: '#f8fafc' 
                }}
              />
              <Bar dataKey="probability" radius={[0, 6, 6, 0]}>
                {chartData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.isPredicted ? '#38bdf8' : '#334155'} 
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
