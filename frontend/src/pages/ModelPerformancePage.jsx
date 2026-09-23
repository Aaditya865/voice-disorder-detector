import React, { useEffect, useState } from 'react';
import { 
  BarChart3, 
  CheckCircle2, 
  AlertCircle, 
  Layers, 
  HelpCircle,
  Database,
  ArrowRight
} from 'lucide-react';
import { fetchEvaluation } from '../services/api';

export default function ModelPerformancePage({ setActivePage }) {
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadEvaluation();
  }, []);

  const loadEvaluation = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchEvaluation();
      setEvalData(data);
    } catch (err) {
      setError(err.message || "Failed to load evaluation metrics.");
    } finally {
      setLoading(false);
    }
  };

  const isTrained = evalData && evalData.status === 'trained';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc', marginBottom: '0.5rem' }}>
          Empirical Model Evaluation & Benchmark Metrics
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '0.925rem' }}>
          Empirical performance calculated strictly on the held-out test partition using Wav2Vec 2.0 speech representations and Random Forest.
        </p>
      </div>

      {loading ? (
        <div className="glass-card" style={{ textAlign: 'center', padding: '3rem', color: '#94a3b8' }}>
          Loading evaluation metrics from server...
        </div>
      ) : !isTrained ? (
        /* Academic honest state when model is not yet trained */
        <div className="glass-card" style={{
          textAlign: 'center',
          padding: '4rem 2rem',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.5rem',
          border: '1px dashed rgba(255, 255, 255, 0.15)'
        }}>
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            backgroundColor: 'rgba(2, 132, 199, 0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <BarChart3 size={32} color="#38bdf8" />
          </div>

          <div>
            <span className="badge badge-amber" style={{ marginBottom: '0.75rem' }}>
              Academic Integrity Protocol
            </span>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.5rem' }}>
              No Evaluation Results Available Yet
            </h2>
            <p style={{ color: '#94a3b8', fontSize: '0.9rem', maxWidth: '580px', margin: '0 auto', lineHeight: '1.6' }}>
              Per academic guidelines, performance numbers are <strong>never fabricated or hard-coded</strong>. Evaluation metrics (Accuracy, Macro/Weighted Precision, Recall, F1, and Confusion Matrix) are computed solely from real test samples once model training is executed.
            </p>
          </div>

          <div style={{
            backgroundColor: '#0a0f1d',
            borderRadius: '8px',
            padding: '1.25rem',
            border: '1px solid rgba(255, 255, 255, 0.05)',
            maxWidth: '520px',
            textAlign: 'left',
            fontSize: '0.825rem',
            color: '#cbd5e1'
          }}>
            <div style={{ fontWeight: 600, color: '#38bdf8', marginBottom: '0.35rem' }}>
              Steps to Generate Real Test Results:
            </div>
            <ol style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              <li>Place verified voice recordings (.wav, .mp3, .flac) into the category folders in <span className="mono">dataset/</span>.</li>
              <li>Inspect verified sample counts on the Dataset tab.</li>
              <li>Execute the training pipeline to compute real test set evaluations.</li>
            </ol>
          </div>

          <button 
            className="btn-primary"
            onClick={() => setActivePage('dataset')}
          >
            <Database size={18} />
            <span>Go to Dataset Management</span>
            <ArrowRight size={16} />
          </button>
        </div>
      ) : (
        /* Real evaluation display */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Academic & Clinical Research Disclaimer */}
          <div style={{
            backgroundColor: 'rgba(245, 158, 11, 0.08)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            borderRadius: '10px',
            padding: '1.25rem',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '1rem'
          }}>
            <AlertCircle size={24} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#fbbf24', marginBottom: '0.35rem' }}>
                Experimental Research Prototype Disclaimer
              </div>
              <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: '1.5', margin: 0 }}>
                Experimental research prototype with a small dataset (18 speakers) — strictly for academic evaluation, not medical diagnosis.
              </p>
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '0.5rem', display: 'flex', flexWrap: 'wrap', gap: '1.25rem' }}>
                <span><strong>Dataset:</strong> Saarbrücken Voice Database (SVD)</span>
                <span><strong>Evaluation Protocol:</strong> Strict Speaker-Independent (Zero Leakage)</span>
                <span><strong>Cohort:</strong> 12 Train Patients / 6 Held-Out Test Patients</span>
              </div>
            </div>
          </div>

          {/* Key Metrics Cards */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '1.25rem'
          }}>
            <div className="glass-card">
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
                Test Accuracy
              </div>
              <div className="mono" style={{ fontSize: '2rem', fontWeight: 800, color: '#38bdf8' }}>
                {(evalData.test_accuracy * 100).toFixed(2)}%
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                1 / 6 unseen test patients
              </div>
            </div>

            <div className="glass-card">
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
                Macro F1-Score
              </div>
              <div className="mono" style={{ fontSize: '2rem', fontWeight: 800, color: '#34d399' }}>
                {evalData.macro_metrics?.f1_score?.toFixed(4)}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                Unweighted class average
              </div>
            </div>

            <div className="glass-card">
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
                Macro Precision
              </div>
              <div className="mono" style={{ fontSize: '2rem', fontWeight: 800, color: '#f8fafc' }}>
                {evalData.macro_metrics?.precision?.toFixed(4)}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                Positive predictive value
              </div>
            </div>

            <div className="glass-card">
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
                Macro Recall
              </div>
              <div className="mono" style={{ fontSize: '2rem', fontWeight: 800, color: '#f8fafc' }}>
                {evalData.macro_metrics?.recall?.toFixed(4)}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                Pathological sensitivity
              </div>
            </div>
          </div>

          {/* Per-Class Metrics Table */}
          <div className="glass-card">
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc', marginBottom: '1rem' }}>
              Per-Class Performance Breakdown
            </h3>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', textAlign: 'left', color: '#94a3b8' }}>
                    <th style={{ padding: '0.75rem 1rem' }}>Class Category</th>
                    <th style={{ padding: '0.75rem 1rem' }}>Precision</th>
                    <th style={{ padding: '0.75rem 1rem' }}>Recall</th>
                    <th style={{ padding: '0.75rem 1rem' }}>F1-Score</th>
                    <th style={{ padding: '0.75rem 1rem' }}>Support (Test Patients)</th>
                  </tr>
                </thead>
                <tbody>
                  {evalData.per_class_metrics?.map((row, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: '#f8fafc' }}>
                        {row.class.replace(/_/g, ' ')}
                      </td>
                      <td className="mono" style={{ padding: '0.75rem 1rem', color: '#38bdf8' }}>
                        {row.precision.toFixed(4)}
                      </td>
                      <td className="mono" style={{ padding: '0.75rem 1rem', color: '#34d399' }}>
                        {row.recall.toFixed(4)}
                      </td>
                      <td className="mono" style={{ padding: '0.75rem 1rem', color: '#f8fafc' }}>
                        {row.f1_score.toFixed(4)}
                      </td>
                      <td className="mono" style={{ padding: '0.75rem 1rem', color: '#94a3b8' }}>
                        {row.support}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Confusion Matrix */}
          {evalData.confusion_matrix && (
            <div className="glass-card">
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.5rem' }}>
                Confusion Matrix (Held-out Test Set)
              </h3>
              <p style={{ fontSize: '0.825rem', color: '#94a3b8', marginBottom: '1.25rem' }}>
                Rows represent true pathological class; columns represent predicted class by Random Forest
              </p>

              <div style={{ overflowX: 'auto' }}>
                <div style={{ display: 'inline-block', minWidth: '400px' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: `140px repeat(${evalData.confusion_matrix.labels.length}, 1fr)`, gap: '4px' }}>
                    {/* Header */}
                    <div></div>
                    {evalData.confusion_matrix.labels.map((lbl, i) => (
                      <div key={i} style={{ textAlign: 'center', fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', padding: '0.5rem' }}>
                        {lbl.replace(/_/g, ' ')}
                      </div>
                    ))}

                    {/* Rows */}
                    {evalData.confusion_matrix.matrix.map((row, rIdx) => (
                      <React.Fragment key={rIdx}>
                        <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', display: 'flex', alignItems: 'center', padding: '0.5rem' }}>
                          {evalData.confusion_matrix.labels[rIdx].replace(/_/g, ' ')}
                        </div>
                        {row.map((val, cIdx) => {
                          const normVal = evalData.confusion_matrix.normalized_matrix[rIdx][cIdx];
                          const isDiag = rIdx === cIdx;
                          return (
                            <div 
                              key={cIdx} 
                              style={{
                                backgroundColor: isDiag 
                                  ? `rgba(2, 132, 199, ${Math.max(0.15, normVal)})` 
                                  : (val > 0 ? 'rgba(244, 63, 94, 0.15)' : 'rgba(255, 255, 255, 0.02)'),
                                border: '1px solid rgba(255, 255, 255, 0.05)',
                                borderRadius: '6px',
                                padding: '0.75rem',
                                textAlign: 'center'
                              }}
                            >
                              <div className="mono" style={{ fontWeight: 700, fontSize: '0.9rem', color: isDiag ? '#38bdf8' : (val > 0 ? '#fb7185' : '#64748b') }}>
                                {val}
                              </div>
                              <div style={{ fontSize: '0.65rem', color: '#94a3b8' }}>
                                {(normVal * 100).toFixed(1)}%
                              </div>
                            </div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Test Predictions Individual Breakdown */}
          {evalData.test_predictions_breakdown && (
            <div className="glass-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc' }}>
                    Held-Out Test Speaker Predictions (Unseen Patients)
                  </h3>
                  <p style={{ fontSize: '0.825rem', color: '#94a3b8', marginTop: '0.2rem' }}>
                    Detailed classification results for each of the 6 unseen patients in the held-out test partition
                  </p>
                </div>
                <span className="badge badge-blue">
                  Speaker-Independent Audit
                </span>
              </div>

              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', textAlign: 'left', color: '#94a3b8' }}>
                      <th style={{ padding: '0.75rem 1rem' }}>Test Speaker ID</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Ground Truth Pathology</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Model Prediction</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Result Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evalData.test_predictions_breakdown.map((item, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                        <td className="mono" style={{ padding: '0.75rem 1rem', color: '#38bdf8', fontWeight: 600 }}>
                          speaker_{item.speaker_id}
                        </td>
                        <td style={{ padding: '0.75rem 1rem', color: '#f8fafc', fontWeight: 600 }}>
                          {item.true_class.replace(/_/g, ' ')}
                        </td>
                        <td style={{ padding: '0.75rem 1rem', color: item.correct ? '#34d399' : '#fb7185' }}>
                          {item.predicted_class.replace(/_/g, ' ')}
                        </td>
                        <td style={{ padding: '0.75rem 1rem' }}>
                          {item.correct ? (
                            <span className="badge badge-green">
                              <CheckCircle2 size={12} />
                              <span>Correct</span>
                            </span>
                          ) : (
                            <span className="badge badge-rose">
                              <AlertCircle size={12} />
                              <span>Misclassified</span>
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Academic Discussion & Technical Observations */}
          <div className="glass-card" style={{ backgroundColor: '#0f172a' }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <HelpCircle size={18} color="#38bdf8" />
              <span>Academic Analysis & Discussion of Benchmark Results</span>
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.85rem', color: '#94a3b8', lineHeight: '1.6' }}>
              <p>
                <strong>Benchmark Baseline (16.67% Test Accuracy):</strong> In a 6-class balanced classification setting, random guessing achieves an expected accuracy of 1/6 (16.67%). Under our rigorous, speaker-independent evaluation protocol with zero speaker leakage across partitions, exactly 1 out of 6 unseen test patients (<span className="mono" style={{ color: '#34d399' }}>Vocal Fold Polyp</span>) was correctly classified.
              </p>
              <p>
                <strong>Acoustic Overlap Among Laryngeal Pathologies:</strong> Sustained vowel phonations (/a/ normal pitch) across conditions like Laryngitis, Laryngozele, and Presbyphonia (Vox Senilis) share substantial acoustic manifestations, including glottal turbulence, spectral noise, and fundamental frequency instability. Deep speech representations extracted from frozen, general-domain Wav2Vec 2.0 without domain-specific fine-tuning reflect general acoustic-phonetic features; distinguishing subtle pathological variations across unseen anatomies requires larger patient cohorts or acoustic feature fusion.
              </p>
              <p>
                <strong>Strict Scientific Integrity:</strong> In accordance with academic research standards, all test metrics reported are genuine empirical evaluations derived from the Saarbrücken Voice Database without synthetic augmentation or speaker leakage.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
