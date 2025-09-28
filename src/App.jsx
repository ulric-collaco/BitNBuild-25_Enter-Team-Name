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
      console.log('🚀 Starting analysis for URL:', url);
      console.log('📡 Sending request to web scraper API...');
      
      // Call the web scraper API
      const response = await fetch('http://localhost:8001/scrape-url', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url })
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const result = await response.json();
      console.log('✅ Web scraper response received');
      console.log('📊 Response status:', result.status);
      console.log('💬 Response message:', result.message);
      console.log('🎯 Check the Web Scraper terminal for detailed AI analysis output!');

      // Parse the Gemini response which should be the analysis result
      let analysisResult;
      try {
        analysisResult = JSON.parse(result.gemini_response);
      } catch (parseError) {
        console.error('Failed to parse Gemini response:', parseError);
        throw new Error('Invalid response format from analysis service');
      }

      console.log('📊 Analysis result:', analysisResult);
      
      // Transform the API response to match our Dashboard component expectations
      const transformedData = {
        url: url,
        status: result.status,
        message: result.message,
        rawResponse: analysisResult,
        sentimentBreakdown: analysisResult.sentiment_distribution ? 
          analysisResult.sentiment_distribution.map(item => ({
            name: item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1),
            value: Math.round(item.percentage),
            color: item.sentiment === 'positive' ? '#22c55e' : 
                   item.sentiment === 'negative' ? '#ef4444' : '#6b7280'
          })) : [
            { name: 'Positive', value: 0, color: '#22c55e' },
            { name: 'Negative', value: 0, color: '#ef4444' },
            { name: 'Neutral', value: 0, color: '#6b7280' }
          ],
        positiveKeywords: analysisResult.keywords?.positive || [],
        negativeKeywords: analysisResult.keywords?.negative || [],
        overallSentiment: {
          score: analysisResult.overall_sentiment?.confidence || 0,
          label: analysisResult.overall_sentiment?.sentiment || 'Unknown'
        },
        totalReviews: analysisResult.summary?.total_reviews || 0,
        emotions: analysisResult.emotion_distribution || []
      };
      
      setAnalysisData(transformedData);
      setHasResults(true);
      
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      alert(`Analysis failed: ${error.message}`);
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