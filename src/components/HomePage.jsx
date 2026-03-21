import React from 'react';
import UrlInput from './UrlInput';
import LightRays from '../LightRays';

const HomePage = ({ onAnalyze }) => {
  return (
    <div className="min-h-screen bg-gray-900 relative">
      {/* Background Light Rays */}
      <div style={{ width: '100%', height: '100vh', position: 'fixed', top: 0, left: 0, zIndex: 0 }}>
        <LightRays
          raysOrigin="top-center"
          raysColor="#FFFFFF"
          raysSpeed={0.3}
          lightSpread={0.5}
          rayLength={1.0}
          followMouse={true}
          mouseInfluence={0.1}
          noiseAmount={0.05}
          distortion={0.02}
          className="custom-rays"
        />
      </div>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8 relative z-10">
        <UrlInput onAnalyze={onAnalyze} />
      </main>
    </div>
  );
};

export default HomePage;