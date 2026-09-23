import React from 'react';
import { 
  ArrowRight, 
  Mic, 
  Layers, 
  Cpu, 
  BarChart3, 
  Sparkles, 
  ShieldAlert,
  ChevronRight,
  Database,
  FileCheck2
} from 'lucide-react';

export default function HomePage({ setActivePage, classesInfo, serverHealth }) {
  const targetClasses = classesInfo?.target_classes || [];

  const pipelineSteps = [
    { title: "Voice Recording", desc: "User uploads audio (.wav, .mp3, .flac)", icon: Mic, color: "#38bdf8" },
    { title: "Audio Preprocessing", desc: "Mono conversion, 16 kHz resample & normalize", icon: Layers, color: "#06b6d4" },
    { title: "Wav2Vec 2.0", desc: "Pretrained deep transformer speech encoder", icon: Cpu, color: "#3b82f6" },
    { title: "Speech Features", desc: "Temporal mean pooling to 768-D representation", icon: Sparkles, color: "#8b5cf6" },
    { title: "Random Forest", desc: "Balanced ensemble classification trees", icon: BarChart3, color: "#10b981" },
    { title: "Multi-Class Prediction", desc: "Class diagnosis, confidence & probabilities", icon: FileCheck2, color: "#0284c7" }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
      {/* Hero Section */}
      <section style={{
        background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.9) 0%, rgba(10, 15, 29, 1) 100%)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '16px',
        padding: '3.5rem 2rem',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
          backgroundColor: 'rgba(2, 132, 199, 0.12)',
          border: '1px solid rgba(2, 132, 199, 0.3)',
          borderRadius: '9999px',
          padding: '0.35rem 1rem',
          fontSize: '0.8rem',
          fontWeight: 600,
          color: '#38bdf8',
          marginBottom: '1.5rem'
        }}>
          <span>Final-Year Academic Research Platform</span>
        </div>

        <h1 style={{
          fontSize: 'clamp(2rem, 4vw, 2.75rem)',
          fontWeight: 800,
          letterSpacing: '-0.025em',
          color: '#ffffff',
          lineHeight: '1.2',
          maxWidth: '900px',
          margin: '0 auto 1.25rem'
        }}>
          AI-Based Multi-Class Laryngeal Voice Disorder Classification System
        </h1>

        <p style={{
          fontSize: '1.125rem',
          color: '#94a3b8',
          maxWidth: '740px',
          margin: '0 auto 2.25rem',
          lineHeight: '1.6'
        }}>
          An AI-based research system that analyzes voice recordings using 
          <strong> Wav2Vec 2.0 </strong> speech representations and 
          <strong> Random Forest </strong> classification.
        </p>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '1rem',
          flexWrap: 'wrap'
        }}>
          <button 
            className="btn-primary"
            onClick={() => setActivePage('analysis')}
            style={{ fontSize: '1rem', padding: '0.875rem 2rem' }}
          >
            <Mic size={20} />
            <span>Start Voice Analysis</span>
            <ArrowRight size={18} />
          </button>

          <button 
            className="btn-secondary"
            onClick={() => setActivePage('architecture')}
            style={{ fontSize: '1rem', padding: '0.875rem 1.75rem' }}
          >
            <Cpu size={20} />
            <span>Examine ML Architecture</span>
          </button>
        </div>
      </section>

      {/* Visual Pipeline Section */}
      <section>
        <div style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f8fafc' }}>
            Acoustic Feature Extraction & Classification Pipeline
          </h2>
          <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
            End-to-end representation learning and non-linear classification flow
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '1rem',
          position: 'relative'
        }}>
          {pipelineSteps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <div 
                key={idx}
                className="glass-card"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem',
                  position: 'relative'
                }}
              >
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}>
                  <div style={{
                    width: '38px',
                    height: '38px',
                    borderRadius: '8px',
                    backgroundColor: `${step.color}20`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <Icon size={20} color={step.color} />
                  </div>
                  <span className="mono" style={{ fontSize: '0.75rem', color: '#64748b' }}>
                    0{idx + 1}
                  </span>
                </div>

                <div style={{ fontWeight: 600, fontSize: '0.95rem', color: '#f8fafc' }}>
                  {step.title}
                </div>
                <div style={{ fontSize: '0.8rem', color: '#94a3b8', lineHeight: '1.4' }}>
                  {step.desc}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Target Classes Section */}
      <section>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '1.5rem' }}>
          <div>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f8fafc' }}>
              Supported Laryngeal Pathologies (6 Target Classes)
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
              The system architecture is designed to classify voice samples across six diagnostic categories
            </p>
          </div>
          <button 
            className="btn-secondary"
            onClick={() => setActivePage('dataset')}
            style={{ padding: '0.5rem 1rem', fontSize: '0.825rem' }}
          >
            <Database size={16} />
            <span>Dataset Audit</span>
          </button>
        </div>

        <div className="grid-responsive">
          {targetClasses.map((cls, idx) => (
            <div key={cls.id || idx} className="glass-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                <span className="badge badge-blue">{cls.category || "Laryngeal"}</span>
                <span className="mono" style={{ fontSize: '0.75rem', color: '#64748b' }}>Class {idx + 1}</span>
              </div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#ffffff', marginBottom: '0.4rem' }}>
                {cls.displayName}
              </h3>
              <p style={{ fontSize: '0.825rem', color: '#94a3b8', lineHeight: '1.5' }}>
                {cls.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Methodology Highlight */}
      <section className="glass-card" style={{
        backgroundColor: '#0f172a',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '2rem',
        alignItems: 'center'
      }}>
        <div>
          <span className="badge badge-green" style={{ marginBottom: '0.75rem' }}>
            Methodological Advancement
          </span>
          <h2 style={{ fontSize: '1.35rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.75rem' }}>
            Why Wav2Vec 2.0 + Random Forest?
          </h2>
          <p style={{ fontSize: '0.875rem', color: '#94a3b8', lineHeight: '1.6', marginBottom: '1rem' }}>
            Traditional voice pathology screening relies on handcrafted acoustic features (MFCCs, jitter, shimmer) or rigid 1D CNNs. Our approach leverages <strong>self-supervised deep speech representations</strong> learned from thousands of hours of speech via Wav2Vec 2.0, coupled with an interpretable, non-linear <strong>Random Forest ensemble</strong> that naturally resists overfitting and handles multi-class imbalance.
          </p>
          <button 
            className="btn-secondary"
            onClick={() => setActivePage('architecture')}
            style={{ fontSize: '0.85rem' }}
          >
            <span>Read Architectural Theory</span>
            <ChevronRight size={16} />
          </button>
        </div>

        <div style={{
          backgroundColor: '#0a0f1d',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '10px',
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.875rem'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
            <span style={{ color: '#94a3b8' }}>Feature Extractor:</span>
            <span className="mono" style={{ color: '#38bdf8' }}>facebook/wav2vec2-base</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
            <span style={{ color: '#94a3b8' }}>Representation Dimension:</span>
            <span className="mono" style={{ color: '#f8fafc' }}>768 features (Mean Pooled)</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
            <span style={{ color: '#94a3b8' }}>Classifier Architecture:</span>
            <span className="mono" style={{ color: '#f8fafc' }}>Random Forest (100 Trees)</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
            <span style={{ color: '#94a3b8' }}>Class Balancing:</span>
            <span className="mono" style={{ color: '#34d399' }}>Balanced Class Weighting</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
            <span style={{ color: '#94a3b8' }}>Input Standardization:</span>
            <span className="mono" style={{ color: '#f8fafc' }}>16,000 Hz, Single-Channel</span>
          </div>
        </div>
      </section>
    </div>
  );
}
