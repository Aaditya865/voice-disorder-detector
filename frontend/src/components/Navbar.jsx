import React from 'react';
import { 
  Activity, 
  Mic, 
  BarChart3, 
  Database, 
  Cpu, 
  Info, 
  CheckCircle2, 
  XCircle,
  Stethoscope
} from 'lucide-react';

export default function Navbar({ activePage, setActivePage, serverHealth }) {
  const navItems = [
    { id: 'home', label: 'Home', icon: Activity },
    { id: 'analysis', label: 'Voice Analysis', icon: Mic },
    { id: 'prediction', label: 'Prediction', icon: Stethoscope },
    { id: 'performance', label: 'Model Performance', icon: BarChart3 },
    { id: 'dataset', label: 'Dataset', icon: Database },
    { id: 'architecture', label: 'Model Architecture', icon: Cpu },
    { id: 'about', label: 'About Project', icon: Info },
  ];

  return (
    <header style={{
      backgroundColor: '#0f172a',
      borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      backdropFilter: 'blur(8px)'
    }}>
      <div style={{
        maxWidth: '1280px',
        margin: '0 auto',
        padding: '0.875rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        {/* Brand */}
        <div 
          onClick={() => setActivePage('home')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            cursor: 'pointer'
          }}
        >
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #0284c7 0%, #0d9488 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(2, 132, 199, 0.3)'
          }}>
            <Activity size={22} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '1.05rem', letterSpacing: '-0.01em', color: '#f8fafc' }}>
              Laryngeal Voice AI
            </div>
            <div style={{ fontSize: '0.725rem', color: '#38bdf8', fontWeight: 500 }}>
              Wav2Vec 2.0 + Random Forest
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.25rem',
          flexWrap: 'wrap'
        }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  padding: '0.5rem 0.875rem',
                  borderRadius: '6px',
                  fontSize: '0.85rem',
                  fontWeight: isActive ? 600 : 500,
                  color: isActive ? '#38bdf8' : '#94a3b8',
                  backgroundColor: isActive ? 'rgba(2, 132, 199, 0.12)' : 'transparent',
                  border: isActive ? '1px solid rgba(2, 132, 199, 0.3)' : '1px solid transparent',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <Icon size={16} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Server Status Pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.35rem 0.75rem',
          borderRadius: '9999px',
          backgroundColor: 'rgba(15, 23, 42, 0.8)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          fontSize: '0.75rem'
        }}>
          {serverHealth?.status === 'online' ? (
            <>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10b981' }}></span>
              <span style={{ color: '#34d399', fontWeight: 500 }}>API Connected</span>
              <span style={{ color: '#64748b' }}>({serverHealth.device?.toUpperCase()})</span>
            </>
          ) : (
            <>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#f43f5e' }}></span>
              <span style={{ color: '#fb7185', fontWeight: 500 }}>API Offline</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
