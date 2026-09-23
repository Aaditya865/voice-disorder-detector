import React, { useState, useEffect } from 'react';
import { 
  Database, 
  RefreshCw, 
  FolderCheck, 
  Users, 
  Play, 
  FileAudio,
  Radio,
  Split,
  ExternalLink,
  CheckCircle2,
  AlertCircle
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
import { fetchDatasetStatistics, triggerTrain } from '../services/api';

export default function DatasetPage({ setActivePage }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [trainMessage, setTrainMessage] = useState(null);
  const [trainError, setTrainError] = useState(null);

  useEffect(() => {
    loadDataset();
  }, []);

  const loadDataset = async () => {
    setLoading(true);
    try {
      const data = await fetchDatasetStatistics();
      setStats(data);
    } catch (err) {
      console.error("Failed to load dataset stats:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    setTrainMessage(null);
    setTrainError(null);
    try {
      const res = await triggerTrain();
      setTrainMessage(res.message || "Model trained successfully!");
      await loadDataset();
    } catch (err) {
      setTrainError(err.message || "Training failed");
    } finally {
      setTraining(false);
    }
  };

  const chartData = stats?.classStatistics?.map(cls => ({
    name: cls.displayName || cls.id,
    speakers: cls.speakerCount || 0,
    samples: cls.sampleCount || 0
  })) || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc', marginBottom: '0.5rem' }}>
            Dataset Inspection & Speaker Distribution
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '0.925rem' }}>
            Ground-truth medical acoustic data from the <strong>Saarbrücken Voice Database (SVD)</strong>.
          </p>
        </div>

        <button 
          className="btn-secondary"
          onClick={loadDataset}
          disabled={loading}
          style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
        >
          <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
          <span>Refresh SVD Audit</span>
        </button>
      </div>

      {/* Dataset Source Attribution Banner */}
      <div style={{
        backgroundColor: 'rgba(2, 132, 199, 0.08)',
        border: '1px solid rgba(2, 132, 199, 0.3)',
        borderRadius: '10px',
        padding: '1rem 1.25rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Database size={22} color="#38bdf8" />
          <div>
            <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#f8fafc' }}>
              Source: Saarbrücken Voice Database (SVD)
            </div>
            <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              Institute of Phonetics, Saarland University &middot; Hosted on Zenodo (Record 16874898)
            </div>
          </div>
        </div>

        <a 
          href="https://zenodo.org/records/16874898" 
          target="_blank" 
          rel="noopener noreferrer"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.4rem',
            fontSize: '0.8rem',
            color: '#38bdf8',
            textDecoration: 'none'
          }}
        >
          <span>View Zenodo Repository</span>
          <ExternalLink size={14} />
        </a>
      </div>

      {/* Summary KPI Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '1.25rem'
      }}>
        <div className="glass-card">
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
            Total Distinct Speakers
          </div>
          <div className="mono" style={{ fontSize: '2.25rem', fontWeight: 800, color: '#38bdf8' }}>
            {stats?.totalSpeakers ?? 0}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
            Verified clinical patients
          </div>
        </div>

        <div className="glass-card">
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
            Total Audio Files
          </div>
          <div className="mono" style={{ fontSize: '2.25rem', fontWeight: 800, color: '#34d399' }}>
            {stats?.totalSamples ?? 0}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
            Sustained vowel recordings
          </div>
        </div>

        <div className="glass-card">
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
            Active Pathology Classes
          </div>
          <div className="mono" style={{ fontSize: '2.25rem', fontWeight: 800, color: '#f8fafc' }}>
            {stats?.availableClassesCount ?? 6} / {stats?.totalTargetClasses ?? 6}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
            Balanced 6-class prototype
          </div>
        </div>

        <div className="glass-card">
          <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
            Split Strategy
          </div>
          <div className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fbbf24', marginTop: '0.5rem' }}>
            Speaker-Independent
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
            Zero speaker leakage
          </div>
        </div>
      </div>

      {/* Acoustic Parameters & Partitions */}
      <div className="glass-card">
        <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc', marginBottom: '1rem' }}>
          Dataset Acoustic Specifications & Partitions
        </h3>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1rem',
          fontSize: '0.85rem'
        }}>
          <div style={{ backgroundColor: '#0a0f1d', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ color: '#64748b', display: 'block', marginBottom: '0.25rem' }}>Recording Type:</span>
            <span style={{ color: '#38bdf8', fontWeight: 600 }}>{stats?.recordingType || "Sustained vowel /a/ at normal pitch"}</span>
          </div>

          <div style={{ backgroundColor: '#0a0f1d', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ color: '#64748b', display: 'block', marginBottom: '0.25rem' }}>Sampling Rate:</span>
            <span style={{ color: '#34d399', fontWeight: 600 }}>{stats?.originalSamplingRate || "50,000 Hz -> 16,000 Hz Mono"}</span>
          </div>

          <div style={{ backgroundColor: '#0a0f1d', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ color: '#64748b', display: 'block', marginBottom: '0.25rem' }}>Audio Format:</span>
            <span style={{ color: '#f8fafc', fontWeight: 600 }}>16-bit Linear PCM (.wav)</span>
          </div>
        </div>

        {/* Partition Details */}
        <div style={{
          marginTop: '1.25rem',
          backgroundColor: '#0a0f1d',
          padding: '1rem',
          borderRadius: '8px',
          border: '1px solid rgba(255,255,255,0.05)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem',
          fontSize: '0.825rem'
        }}>
          <div>
            <span style={{ color: '#34d399', fontWeight: 700 }}>Training Speakers ({stats?.trainSpeakers?.length || 12}):</span>
            <div className="mono" style={{ color: '#cbd5e1', marginTop: '0.2rem', wordBreak: 'break-all' }}>
              {stats?.trainSpeakers?.join(', ') || '1, 2, 1602, 1633, 73, 1347, 1306, 1317, 1302, 1305, 1304, 1318'}
            </div>
          </div>

          <div>
            <span style={{ color: '#38bdf8', fontWeight: 700 }}>Test Speakers (Held-out Evaluation, {stats?.testSpeakers?.length || 6}):</span>
            <div className="mono" style={{ color: '#cbd5e1', marginTop: '0.2rem', wordBreak: 'break-all' }}>
              {stats?.testSpeakers?.join(', ') || '3, 2191, 1387, 1363, 1312, 1320'}
            </div>
          </div>
        </div>
      </div>

      {/* Class Distribution Chart */}
      <div className="glass-card">
        <div style={{ marginBottom: '1rem' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc' }}>
            Speaker & Sample Distribution Across 6 Classes
          </h3>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
            Verified distinct patients from the Saarbrücken Voice Database
          </p>
        </div>

        <div style={{ height: '240px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 20, left: 10, bottom: 25 }}>
              <XAxis 
                dataKey="name" 
                stroke="#64748b" 
                fontSize={11} 
                interval={0}
                angle={-15}
                textAnchor="end"
              />
              <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0f172a', 
                  borderColor: 'rgba(255,255,255,0.1)', 
                  borderRadius: '8px', 
                  color: '#f8fafc' 
                }}
              />
              <Bar dataKey="speakers" name="Distinct Speakers" radius={[4, 4, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill="#38bdf8" 
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Class Table */}
      <div className="glass-card">
        <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc', marginBottom: '1rem' }}>
          Class Breakdown & Patient Identifiers
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', textAlign: 'left', color: '#94a3b8' }}>
                <th style={{ padding: '0.75rem 1rem' }}>Target Class</th>
                <th style={{ padding: '0.75rem 1rem' }}>Medical Category</th>
                <th style={{ padding: '0.75rem 1rem' }}>Distinct Speakers</th>
                <th style={{ padding: '0.75rem 1rem' }}>Speaker IDs</th>
                <th style={{ padding: '0.75rem 1rem' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {stats?.classStatistics?.map((cls, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: '#f8fafc' }}>
                    {cls.displayName}
                  </td>
                  <td style={{ padding: '0.75rem 1rem', color: '#94a3b8' }}>
                    {cls.category}
                  </td>
                  <td className="mono" style={{ padding: '0.75rem 1rem', color: '#38bdf8' }}>
                    {cls.speakerCount} speakers
                  </td>
                  <td className="mono" style={{ padding: '0.75rem 1rem', color: '#cbd5e1', fontSize: '0.8rem' }}>
                    {cls.speakers?.map(s => `speaker_${s}`).join(', ') || 'N/A'}
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span className="badge badge-green">
                      <FolderCheck size={12} />
                      <span>Ready</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Retrain Action Bar */}
      <div className="glass-card" style={{
        backgroundColor: '#0f172a',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1.5rem'
      }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: '1rem', color: '#f8fafc', marginBottom: '0.25rem' }}>
            Retrain Wav2Vec 2.0 + Random Forest Pipeline
          </div>
          <p style={{ fontSize: '0.825rem', color: '#94a3b8' }}>
            Re-runs feature extraction and trains the 100-tree Random Forest classifier on the speaker-independent training partition.
          </p>
        </div>

        <button 
          className="btn-primary"
          disabled={training}
          onClick={handleTrain}
        >
          <Play size={16} />
          <span>{training ? "Training Model..." : "Retrain Random Forest Classifier"}</span>
        </button>
      </div>

      {/* Alerts */}
      {trainMessage && (
        <div style={{
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          border: '1px solid rgba(16, 185, 129, 0.3)',
          borderRadius: '8px',
          padding: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem'
        }}>
          <CheckCircle2 size={20} color="#34d399" />
          <span style={{ fontSize: '0.85rem', color: '#a7f3d0' }}>{trainMessage}</span>
        </div>
      )}

      {trainError && (
        <div style={{
          backgroundColor: 'rgba(244, 63, 94, 0.1)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          borderRadius: '8px',
          padding: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem'
        }}>
          <AlertCircle size={20} color="#fb7185" />
          <span style={{ fontSize: '0.85rem', color: '#fecdd3' }}>{trainError}</span>
        </div>
      )}
    </div>
  );
}
