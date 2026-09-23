import React from 'react';
import { AlertTriangle, ShieldCheck } from 'lucide-react';

export default function Footer() {
  return (
    <footer style={{
      backgroundColor: '#0a0f1d',
      borderTop: '1px solid rgba(255, 255, 255, 0.08)',
      padding: '2.5rem 1.5rem',
      marginTop: 'auto'
    }}>
      <div style={{ maxWidth: '1280px', margin: '0 auto' }}>
        {/* Medical Disclaimer Banner */}
        <div style={{
          backgroundColor: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          borderRadius: '10px',
          padding: '1rem 1.25rem',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '1rem',
          marginBottom: '2rem'
        }}>
          <AlertTriangle size={22} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <div style={{ fontWeight: 700, fontSize: '0.875rem', color: '#fbbf24', marginBottom: '0.25rem' }}>
              ACADEMIC RESEARCH & CLINICAL SCREENING DISCLAIMER
            </div>
            <p style={{ fontSize: '0.825rem', color: '#cbd5e1', lineHeight: '1.5' }}>
              <strong>This system is developed for academic and research purposes only. It is not a medical diagnostic tool and should not replace professional medical evaluation.</strong> Acoustic voice classification provides non-invasive computer-assisted screening support and must be evaluated alongside clinical laryngoscopy, stroboscopy, and formal laryngological assessment.
            </p>
          </div>
        </div>

        {/* Footer Info Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1.5rem',
          paddingTop: '1rem',
          borderTop: '1px solid rgba(255, 255, 255, 0.05)',
          fontSize: '0.825rem',
          color: '#64748b'
        }}>
          <div>
            <div style={{ color: '#f8fafc', fontWeight: 600, marginBottom: '0.5rem' }}>
              Project Overview
            </div>
            <p>
              AI-Based Multi-Class Laryngeal Voice Disorder Classification System using self-supervised Wav2Vec 2.0 speech representations and Random Forest ensembling.
            </p>
          </div>

          <div>
            <div style={{ color: '#f8fafc', fontWeight: 600, marginBottom: '0.5rem' }}>
              Technology Stack
            </div>
            <p>
              PyTorch &middot; Hugging Face Transformers (<span className="mono">facebook/wav2vec2-base</span>) &middot; Scikit-Learn &middot; Librosa &middot; FastAPI &middot; React + Vite
            </p>
          </div>

          <div>
            <div style={{ color: '#f8fafc', fontWeight: 600, marginBottom: '0.5rem' }}>
              Target Pathologies
            </div>
            <p>
              Normal Phonation, Laryngozele, Vox Senilis (Presbyphonia), Vocal Fold Polyp, Vocal Fold Paralysis, and Laryngitis.
            </p>
          </div>
        </div>

        <div style={{
          marginTop: '2rem',
          textAlign: 'center',
          fontSize: '0.75rem',
          color: '#475569',
          borderTop: '1px solid rgba(255, 255, 255, 0.05)',
          paddingTop: '1.5rem'
        }}>
          Final-Year Academic Project &middot; Academic & Research Evaluation Platform
        </div>
      </div>
    </footer>
  );
}
