import React, { useEffect, useRef } from 'react';

export default function WaveformVisualizer({ 
  envelope = [], 
  duration = 0, 
  isPlaying = false,
  currentTime = 0 
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Draw background grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    // Center line
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    if (!envelope || envelope.length === 0) {
      // Empty placeholder state
      ctx.fillStyle = '#64748b';
      ctx.font = '12px Plus Jakarta Sans, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Waveform preview will appear here upon loading valid audio', width / 2, height / 2 + 4);
      return;
    }

    const n = envelope.length;
    const barWidth = Math.max(2, (width / n) - 1.5);
    const halfHeight = height / 2;

    // Draw waveform bars symmetrically around center line
    envelope.forEach((val, i) => {
      const x = (i / n) * width;
      // Clamp amplitude
      const amp = Math.min(1.0, Math.max(0.02, val));
      const barH = amp * (halfHeight - 8);

      // Gradient color based on playback progress
      const progress = duration > 0 ? (currentTime / duration) : 0;
      const isPlayed = (i / n) <= progress;

      if (isPlayed) {
        ctx.fillStyle = '#38bdf8'; // Played highlight
      } else {
        ctx.fillStyle = '#0284c7'; // Unplayed base
      }

      // Upper bar
      ctx.fillRect(x, halfHeight - barH, barWidth, barH);
      // Lower symmetric reflection with lower opacity
      ctx.fillStyle = isPlayed ? 'rgba(56, 189, 248, 0.4)' : 'rgba(2, 132, 199, 0.35)';
      ctx.fillRect(x, halfHeight, barWidth, barH * 0.7);
    });

    // Draw current playhead line if duration > 0
    if (duration > 0 && currentTime > 0) {
      const playheadX = (currentTime / duration) * width;
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
    }
  }, [envelope, duration, currentTime, isPlaying]);

  return (
    <div style={{
      backgroundColor: '#0a0f1d',
      borderRadius: '8px',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      padding: '0.75rem',
      position: 'relative'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '0.5rem',
        fontSize: '0.75rem',
        color: '#94a3b8'
      }}>
        <span>Acoustic Waveform Envelope (Standardized 16 kHz Mono)</span>
        <span className="mono">{duration > 0 ? `${duration.toFixed(2)}s duration` : '0.00s'}</span>
      </div>

      <canvas 
        ref={canvasRef} 
        width={720} 
        height={130} 
        style={{
          width: '100%',
          height: '130px',
          display: 'block'
        }}
      />

      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '0.7rem',
        color: '#64748b',
        marginTop: '0.35rem'
      }}>
        <span>0.00s</span>
        <span>{duration > 0 ? `${(duration / 2).toFixed(2)}s` : '0.50x'}</span>
        <span>{duration > 0 ? `${duration.toFixed(2)}s` : 'Max'}</span>
      </div>
    </div>
  );
}
