import React from 'react';
import { 
  ShieldCheck, 
  BookOpen, 
  GraduationCap, 
  Award, 
  Cpu, 
  AlertTriangle,
  FolderPlus,
  Stethoscope
} from 'lucide-react';

export default function AboutProjectPage() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
      <div>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc', marginBottom: '0.5rem' }}>
          About the Academic Project
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '0.925rem' }}>
          Final-year engineering and biomedical computing research project on AI-based multi-class laryngeal voice disorder classification.
        </p>
      </div>

      {/* Mandatory Medical Disclaimer Banner */}
      <div style={{
        backgroundColor: 'rgba(245, 158, 11, 0.08)',
        border: '1px solid rgba(245, 158, 11, 0.3)',
        borderRadius: '12px',
        padding: '1.5rem',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '1.25rem'
      }}>
        <AlertTriangle size={28} color="#f59e0b" style={{ flexShrink: 0, marginTop: '4px' }} />
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fbbf24', marginBottom: '0.5rem' }}>
            Academic & Healthcare Disclaimer
          </h2>
          <p style={{ fontSize: '0.9rem', color: '#cbd5e1', lineHeight: '1.6' }}>
            <strong>This system is developed for academic and research purposes only. It is not a medical diagnostic tool and should not replace professional medical evaluation.</strong> 
            This application is constructed strictly as a non-invasive acoustic decision-support and screening demonstration. In clinical otolaryngology practice, definitive diagnosis of laryngeal conditions requires indirect laryngoscopy, flexible fiberoptic laryngoscopy, videostroboscopy, and comprehensive physical examination by an ENT specialist.
          </p>
        </div>
      </div>

      {/* Core Project Objectives */}
      <div className="glass-card">
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc', marginBottom: '1rem' }}>
          Academic Research Objectives
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
          <div style={{ backgroundColor: '#0a0f1d', padding: '1.25rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ fontWeight: 600, color: '#38bdf8', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Cpu size={18} />
              <span>Self-Supervised Representation</span>
            </div>
            <p style={{ fontSize: '0.825rem', color: '#94a3b8', lineHeight: '1.5' }}>
              Investigate the efficacy of deep contextualized speech representations extracted from self-supervised transformer encoders (Wav2Vec 2.0) over classical handcrafted acoustic parameters (MFCCs, jitter, shimmer).
            </p>
          </div>

          <div style={{ backgroundColor: '#0a0f1d', padding: '1.25rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ fontWeight: 600, color: '#34d399', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Award size={18} />
              <span>Interpretable Ensemble Learning</span>
            </div>
            <p style={{ fontSize: '0.825rem', color: '#94a3b8', lineHeight: '1.5' }}>
              Deploy balanced Random Forest ensembles that handle high-dimensional speech features without overfitting, offering robust multi-class probability outputs for clinical transparency.
            </p>
          </div>

          <div style={{ backgroundColor: '#0a0f1d', padding: '1.25rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ fontWeight: 600, color: '#fbbf24', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <ShieldCheck size={18} />
              <span>Strict Empirical Honesty</span>
            </div>
            <p style={{ fontSize: '0.825rem', color: '#94a3b8', lineHeight: '1.5' }}>
              Adhere to strict scientific integrity: never fabricate synthetic patient data, invent accuracy metrics, or claim unwarranted clinical diagnostic authority.
            </p>
          </div>
        </div>
      </div>

      {/* Target Classes Extensibility Guide */}
      <div className="glass-card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <FolderPlus size={22} color="#38bdf8" />
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
            System Extensibility: Adding New Pathology Classes
          </h2>
        </div>

        <p style={{ fontSize: '0.875rem', color: '#94a3b8', lineHeight: '1.6', marginBottom: '1.25rem' }}>
          The architecture is fully modular. While currently configured for 6 primary laryngeal categories, researchers and clinical partners can introduce new verified pathology categories dynamically without altering code:
        </p>

        <div style={{
          backgroundColor: '#0a0f1d',
          borderRadius: '8px',
          padding: '1.25rem',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem',
          fontSize: '0.85rem'
        }}>
          <div>
            <span style={{ color: '#38bdf8', fontWeight: 600 }}>1. Create Category Folder:</span>
            <div className="mono" style={{ color: '#cbd5e1', fontSize: '0.8rem', marginTop: '0.2rem' }}>
              dataset/New_Condition_Name/
            </div>
          </div>
          <div>
            <span style={{ color: '#38bdf8', fontWeight: 600 }}>2. Populate Audio Recordings:</span>
            <div style={{ color: '#cbd5e1', fontSize: '0.8rem', marginTop: '0.2rem' }}>
              Place sustained vowel recordings (.wav, .mp3, .flac) from clinically confirmed cases into the new folder.
            </div>
          </div>
          <div>
            <span style={{ color: '#38bdf8', fontWeight: 600 }}>3. Automatic Discovery & Retraining:</span>
            <div style={{ color: '#cbd5e1', fontSize: '0.8rem', marginTop: '0.2rem' }}>
              The dataset scanner dynamically detects the new class, the feature extractor pulls 768-D representations, and the Random Forest retrains the multi-class decision boundaries automatically.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
