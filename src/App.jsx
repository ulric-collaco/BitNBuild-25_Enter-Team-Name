import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import HomePage from './components/HomePage';
import LoadingPage from './components/LoadingPage';
import DashboardPage from './components/DashboardPage';

// Main App Logic Component
function AppLogic() {
  const [analysisData, setAnalysisData] = useState(null);
  const [currentUrl, setCurrentUrl] = useState('');
  const navigate = useNavigate();
  const location = useLocation();

  // Use environment variable for production, fallback to Railway URL
  const WEB_SCRAPER_URL = process.env.REACT_APP_WEB_SCRAPER_URL || 'https://webscrapemaybe-production.up.railway.app';

  const handleAnalyze = async (url) => {
    setCurrentUrl(url);
    
    // Navigate to loading page immediately
    navigate('/loading');
    
    try {
      console.log('🚀 Starting analysis for URL:', url);
      console.log('📡 Sending request to web scraper API:', WEB_SCRAPER_URL);
      
      // Call the web scraper API
      const response = await fetch(`${WEB_SCRAPER_URL}/scrape-url`, {
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

      // Parse the analysis response
      let analysisResult;
      try {
        if (typeof result.gemini_response === 'string') {
          analysisResult = JSON.parse(result.gemini_response);
        } else {
          analysisResult = result.gemini_response;
        }
      } catch (parseError) {
        console.error('Failed to parse analysis response:', parseError);
        throw new Error('Invalid response format from analysis service');
      }

      console.log('📊 Analysis result:', analysisResult);
      
      // Transform the API response to match Dashboard expectations
      const transformedData = {
        url: url,
        status: result.status,
        message: result.message,
        rawResponse: analysisResult,
        sentimentBreakdown: analysisResult.checklist?.sentiment ? [
          { 
            name: 'Positive', 
            value: analysisResult.checklist.sentiment.positive, 
            color: '#22c55e' 
          },
          { 
            name: 'Negative', 
            value: analysisResult.checklist.sentiment.negative, 
            color: '#ef4444' 
          },
          { 
            name: 'Neutral', 
            value: analysisResult.checklist.sentiment.neutral, 
            color: '#6b7280' 
          }
        ] : [
          { name: 'Positive', value: 0, color: '#22c55e' },
          { name: 'Negative', value: 0, color: '#ef4444' },
          { name: 'Neutral', value: 0, color: '#6b7280' }
        ],
        positiveKeywords: analysisResult.keywords?.positive_keywords || [],
        negativeKeywords: analysisResult.keywords?.negative_keywords || [],
        overallSentiment: {
          score: analysisResult.average_rating || 0,
          label: analysisResult.checklist?.sentiment?.positive > analysisResult.checklist?.sentiment?.negative ? 'Positive' : 
                 analysisResult.checklist?.sentiment?.negative > analysisResult.checklist?.sentiment?.positive ? 'Negative' : 'Neutral'
        },
        totalReviews: analysisResult.total_reviews || 0,
        emotions: [] // Removed as per requirements
      };
      
      setAnalysisData(transformedData);
      
      // Navigate to dashboard after processing is complete
      navigate('/dashboard');
      
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      alert(`Analysis failed: ${error.message}`);
      
      // Navigate back to home on error
      navigate('/');
    }
  };

  const handleReset = () => {
    setAnalysisData(null);
    setCurrentUrl('');
    navigate('/');
  };

  return (
    <Routes>
      <Route path="/" element={<HomePage onAnalyze={handleAnalyze} />} />
      <Route path="/loading" element={<LoadingPage url={currentUrl} />} />
      <Route 
        path="/dashboard" 
        element={
          analysisData ? 
            <DashboardPage analysisData={analysisData} onReset={handleReset} /> :
            <HomePage onAnalyze={handleAnalyze} />
        } 
      />
    </Routes>
  );
}

// Main App Component with Router
function App() {
  return (
    <Router>
      <AppLogic />
    </Router>
  );
}

export default App;