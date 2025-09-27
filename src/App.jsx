import React, { useState } from 'react';
import UrlInput from './components/UrlInput';
import Dashboard from './components/Dashboard';
import LightRays from './LightRays';

function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasResults, setHasResults] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);

  const handleAnalyze = async (url) => {
    setIsAnalyzing(true);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock response data
      const mockData = {
        url: url,
        sentimentBreakdown: [
          { name: 'Positive', value: 65, color: '#22c55e' },
          { name: 'Negative', value: 20, color: '#ef4444' },
          { name: 'Neutral', value: 15, color: '#6b7280' }
        ],
        positiveKeywords: [
          'excellent', 'great quality', 'fast shipping', 'love it', 'perfect',
          'amazing', 'highly recommend', 'good value', 'satisfied', 'awesome'
        ],
        negativeKeywords: [
          'poor quality', 'too expensive', 'broke quickly', 'disappointed',
          'not worth it', 'terrible', 'waste of money', 'defective'
        ],
        overallSentiment: {
          score: 85,
          label: 'Very Positive'
        }
      };
      
      setAnalysisData(mockData);
      setHasResults(true);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setHasResults(false);
    setAnalysisData(null);
  };

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
        {!hasResults ? (
          <UrlInput onAnalyze={handleAnalyze} isLoading={isAnalyzing} />
        ) : (
          <Dashboard data={analysisData} onReset={handleReset} />
        )}
      </main>
    </div>
  );
}

export default App;