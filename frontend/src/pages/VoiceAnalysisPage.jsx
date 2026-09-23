import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileAudio, 
  Play, 
  Pause, 
  RotateCcw, 
  Zap, 
  CheckCircle2, 
  AlertCircle, 
  Loader2, 
  ArrowRight,
  Info
} from 'lucide-react';
import WaveformVisualizer from '../components/WaveformVisualizer';
import { extractWaveform, predictVoice } from '../services/api';

export default function VoiceAnalysisPage({ onPredictionResult, setActivePage, serverHealth }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileMeta, setFileMeta] = useState(null);
  const [envelope, setEnvelope] = useState([]);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const [analyzing, setAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(0); // 1: Preprocessing, 2: Wav2Vec2, 3: RF
  const [errorMsg, setErrorMsg] = useState(null);
  const [warningMsg, setWarningMsg] = useState(null);

  const fileInputRef = useRef(null);
  const audioRef = useRef(null);

  const handleFile = async (file) => {
    if (!file) return;
    setErrorMsg(null);
    setWarningMsg(null);

    const validExtensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a'];
    const hasValidExt = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

    if (!hasValidExt) {
      setErrorMsg(`Unsupported file type. Please upload an audio file in ${validExtensions.join(', ')} format.`);
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);

    // Call backend to extract true audio envelope and metadata
    try {
      const data = await extractWaveform(file);
      setEnvelope(data.waveform_envelope || []);
      setFileMeta(data.metadata);
      setDuration(data.metadata?.duration_seconds || 0);
    } catch (err) {
      console.warn("Waveform extraction warning:", err);
      // Fallback: estimate from basic file properties
      setFileMeta({
        original_sample_rate: 16000,
        target_sample_rate: 16000,
        duration_seconds: 0,
        processed_samples: 0
      });
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      if (!duration && audioRef.current.duration) {
        setDuration(audioRef.current.duration);
      }
    }
  };

  const handleReset = () => {
    if (audioRef.current) {
      audioRef.current.pause();
    }
    setSelectedFile(null);
    setFileMeta(null);
    setEnvelope([]);
    setAudioUrl(null);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setErrorMsg(null);
    setWarningMsg(null);
    setAnalyzing(false);
    setAnalysisStep(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setErrorMsg(null);
    setWarningMsg(null);
    setAnalyzing(true);

    try {
      // Step 1: Preprocessing
      setAnalysisStep(1);
      await new Promise(r => setTimeout(r, 400));

      // Step 2: Wav2Vec 2.0 Feature Extraction
      setAnalysisStep(2);

      // Perform actual prediction API call
      const result = await predictVoice(selectedFile);

      // Step 3: Random Forest Classification
      setAnalysisStep(3);
      await new Promise(r => setTimeout(r, 300));

      // Pass result to parent state and navigate to Prediction tab
      onPredictionResult(result);
      setActivePage('prediction');
    } catch (err) {
      setAnalyzing(false);
      setAnalysisStep(0);
      const msg = err.message || "Failed to analyze voice recording.";
      if (msg.includes("not trained") || msg.includes("model_not_trained")) {
        setWarningMsg(msg);
      } else {
        setErrorMsg(msg);
      }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc', marginBottom: '0.5rem' }}>
          Voice Recording Acoustic Analysis
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '0.925rem' }}>
          Upload a sustained vowel phonation or voice sample (.wav, .mp3, .flac) to extract 768-dimensional Wav2Vec 2.0 representations.
        </p>
      </div>

      {/* Upload Dropzone */}
      <div 
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        style={{
          border: '2px dashed rgba(2, 132, 199, 0.35)',
          borderRadius: '12px',
          backgroundColor: selectedFile ? 'rgba(15, 23, 42, 0.6)' : 'rgba(19, 29, 54, 0.5)',
          padding: '2.5rem 1.5rem',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'all 0.2s ease'
        }}
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
      >
        <input 
          type="file" 
          ref={fileInputRef} 
          accept=".wav,.mp3,.flac,.ogg,.m4a,audio/*"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files && handleFile(e.target.files[0])}
        />

        <div style={{
          width: '56px',
          height: '56px',
          borderRadius: '50%',
          backgroundColor: 'rgba(2, 132, 199, 0.12)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 1rem'
        }}>
          <UploadCloud size={28} color="#38bdf8" />
        </div>

        <div style={{ fontWeight: 600, fontSize: '1.05rem', color: '#f8fafc', marginBottom: '0.35rem' }}>
          {selectedFile ? selectedFile.name : "Drag & drop audio file here, or click to browse"}
        </div>
        <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
          Standardized to 16,000 Hz Mono for Wav2Vec 2.0 (Supports WAV, MP3, FLAC)
        </div>
      </div>

      {/* Audio & File Info Section */}
      {selectedFile && (
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{
                width: '40px',
                height: '40px',
                borderRadius: '8px',
                backgroundColor: 'rgba(2, 132, 199, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <FileAudio size={22} color="#38bdf8" />
              </div>
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.95rem', color: '#f8fafc' }}>
                  {selectedFile.name}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                  {(selectedFile.size / 1024).toFixed(1)} KB &middot; {selectedFile.type || 'audio/waveform'}
                </div>
              </div>
            </div>

            {/* Hidden native audio tag */}
            {audioUrl && (
              <audio 
                ref={audioRef} 
                src={audioUrl} 
                onTimeUpdate={handleTimeUpdate}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => { setIsPlaying(false); setCurrentTime(0); }}
                onLoadedMetadata={(e) => setDuration(e.target.duration)}
                style={{ display: 'none' }}
              />
            )}

            {/* Audio Controls */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <button 
                className="btn-secondary" 
                onClick={togglePlay}
                style={{ padding: '0.5rem 1rem', fontSize: '0.825rem' }}
              >
                {isPlaying ? <Pause size={16} /> : <Play size={16} />}
                <span>{isPlaying ? 'Pause' : 'Play Audio'}</span>
              </button>

              <button 
                className="btn-secondary" 
                onClick={handleReset}
                style={{ padding: '0.5rem 1rem', fontSize: '0.825rem', color: '#94a3b8' }}
              >
                <RotateCcw size={16} />
                <span>Reset</span>
              </button>
            </div>
          </div>

          {/* Genuine Waveform Visualizer */}
          <WaveformVisualizer 
            envelope={envelope}
            duration={duration || fileMeta?.duration_seconds || 0}
            currentTime={currentTime}
            isPlaying={isPlaying}
          />

          {/* File Metadata Badges */}
          {fileMeta && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
              gap: '0.75rem',
              fontSize: '0.775rem'
            }}>
              <div style={{ backgroundColor: '#0a0f1d', padding: '0.6rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <span style={{ color: '#64748b', display: 'block' }}>Original Rate</span>
                <span className="mono" style={{ color: '#f8fafc', fontWeight: 600 }}>{fileMeta.original_sample_rate} Hz</span>
              </div>
              <div style={{ backgroundColor: '#0a0f1d', padding: '0.6rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <span style={{ color: '#64748b', display: 'block' }}>Standardized Rate</span>
                <span className="mono" style={{ color: '#38bdf8', fontWeight: 600 }}>{fileMeta.target_sample_rate} Hz</span>
              </div>
              <div style={{ backgroundColor: '#0a0f1d', padding: '0.6rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <span style={{ color: '#64748b', display: 'block' }}>Duration</span>
                <span className="mono" style={{ color: '#f8fafc', fontWeight: 600 }}>{fileMeta.duration_seconds}s</span>
              </div>
              <div style={{ backgroundColor: '#0a0f1d', padding: '0.6rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
                <span style={{ color: '#64748b', display: 'block' }}>Acoustic Channel</span>
                <span className="mono" style={{ color: '#f8fafc', fontWeight: 600 }}>Mono (1-ch)</span>
              </div>
            </div>
          )}

          {/* Analyze Action Bar */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            paddingTop: '0.75rem',
            borderTop: '1px solid rgba(255, 255, 255, 0.05)'
          }}>
            <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
              Feeds audio into pretrained <strong style={{ color: '#94a3b8' }}>facebook/wav2vec2-base</strong> model.
            </div>

            <button 
              className="btn-primary"
              disabled={analyzing}
              onClick={handleAnalyze}
            >
              {analyzing ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  <span>Processing Audio...</span>
                </>
              ) : (
                <>
                  <Zap size={18} />
                  <span>Execute Analysis</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Progress Stepper during Analysis */}
      {analyzing && (
        <div className="glass-card" style={{ backgroundColor: '#0f172a' }}>
          <div style={{ fontWeight: 600, fontSize: '0.9rem', color: '#38bdf8', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Loader2 size={16} className="animate-spin" />
            <span>Multi-Stage Inference Execution</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div style={{
              padding: '0.75rem',
              borderRadius: '8px',
              backgroundColor: analysisStep >= 1 ? 'rgba(2, 132, 199, 0.15)' : 'rgba(255,255,255,0.02)',
              border: analysisStep >= 1 ? '1px solid rgba(2, 132, 199, 0.4)' : '1px solid transparent'
            }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Step 1</div>
              <div style={{ fontWeight: 600, fontSize: '0.85rem', color: analysisStep >= 1 ? '#38bdf8' : '#64748b' }}>
                Resampling & Normalization
              </div>
            </div>

            <div style={{
              padding: '0.75rem',
              borderRadius: '8px',
              backgroundColor: analysisStep >= 2 ? 'rgba(2, 132, 199, 0.15)' : 'rgba(255,255,255,0.02)',
              border: analysisStep >= 2 ? '1px solid rgba(2, 132, 199, 0.4)' : '1px solid transparent'
            }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Step 2</div>
              <div style={{ fontWeight: 600, fontSize: '0.85rem', color: analysisStep >= 2 ? '#38bdf8' : '#64748b' }}>
                Wav2Vec 2.0 Feature Extraction (768-D)
              </div>
            </div>

            <div style={{
              padding: '0.75rem',
              borderRadius: '8px',
              backgroundColor: analysisStep >= 3 ? 'rgba(2, 132, 199, 0.15)' : 'rgba(255,255,255,0.02)',
              border: analysisStep >= 3 ? '1px solid rgba(2, 132, 199, 0.4)' : '1px solid transparent'
            }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Step 3</div>
              <div style={{ fontWeight: 600, fontSize: '0.85rem', color: analysisStep >= 3 ? '#38bdf8' : '#64748b' }}>
                Random Forest Inference
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Not Trained Warning Banner */}
      {warningMsg && (
        <div style={{
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: '10px',
          padding: '1.25rem',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '1rem'
        }}>
          <AlertCircle size={24} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#fbbf24' }}>
              Model Not Trained Yet
            </div>
            <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: '1.5' }}>
              {warningMsg}
            </p>
            <div style={{ marginTop: '0.5rem' }}>
              <button 
                className="btn-secondary"
                onClick={() => setActivePage('dataset')}
                style={{ fontSize: '0.8rem', padding: '0.4rem 0.85rem' }}
              >
                <span>Navigate to Dataset Page to Inspect or Train</span>
                <ArrowRight size={14} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error Banner */}
      {errorMsg && (
        <div style={{
          backgroundColor: 'rgba(244, 63, 94, 0.1)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          borderRadius: '10px',
          padding: '1.25rem',
          display: 'flex',
          alignItems: 'center',
          gap: '1rem'
        }}>
          <AlertCircle size={22} color="#f43f5e" style={{ flexShrink: 0 }} />
          <div style={{ fontSize: '0.875rem', color: '#fecdd3' }}>
            {errorMsg}
          </div>
        </div>
      )}
    </div>
  );
}
