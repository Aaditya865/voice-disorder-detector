import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import VoiceAnalysisPage from './pages/VoiceAnalysisPage';
import PredictionPage from './pages/PredictionPage';
import ModelPerformancePage from './pages/ModelPerformancePage';
import DatasetPage from './pages/DatasetPage';
import ModelArchitecturePage from './pages/ModelArchitecturePage';
import AboutProjectPage from './pages/AboutProjectPage';
import { fetchHealth, fetchClasses } from './services/api';

export default function App() {
  const [activePage, setActivePage] = useState('home');
  const [serverHealth, setServerHealth] = useState(null);
  const [classesInfo, setClassesInfo] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    loadInitialData();
    const interval = setInterval(loadInitialData, 15000);
    return () => clearInterval(interval);
  }, []);

  const loadInitialData = async () => {
    try {
      const health = await fetchHealth();
      setServerHealth(health);
    } catch (err) {
      setServerHealth({ status: 'offline' });
    }

    try {
      const cls = await fetchClasses();
      setClassesInfo(cls);
    } catch (err) {
      // Offline fallback
    }
  };

  const handlePredictionResult = (result) => {
    setPredictionResult(result);
  };

  return (
    <div className="app-container">
      <Navbar 
        activePage={activePage} 
        setActivePage={setActivePage} 
        serverHealth={serverHealth}
      />

      <main className="main-content">
        {activePage === 'home' && (
          <HomePage 
            setActivePage={setActivePage} 
            classesInfo={classesInfo}
            serverHealth={serverHealth}
          />
        )}

        {activePage === 'analysis' && (
          <VoiceAnalysisPage 
            onPredictionResult={handlePredictionResult}
            setActivePage={setActivePage}
            serverHealth={serverHealth}
          />
        )}

        {activePage === 'prediction' && (
          <PredictionPage 
            predictionResult={predictionResult}
            setActivePage={setActivePage}
          />
        )}

        {activePage === 'performance' && (
          <ModelPerformancePage 
            setActivePage={setActivePage}
          />
        )}

        {activePage === 'dataset' && (
          <DatasetPage 
            setActivePage={setActivePage}
          />
        )}

        {activePage === 'architecture' && (
          <ModelArchitecturePage />
        )}

        {activePage === 'about' && (
          <AboutProjectPage />
        )}
      </main>

      <Footer />
    </div>
  );
}
