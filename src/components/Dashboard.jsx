import React from 'react';
import SummaryCard from './SummaryCard';

const Dashboard = ({ data, onReset }) => {
  const { sentimentBreakdown, positiveKeywords, negativeKeywords, url } = data;

  // Calculate sentiment meter angle (0° = all positive, 90° = neutral, 180° = all negative)
  const calculateMeterAngle = () => {
    if (!sentimentBreakdown || sentimentBreakdown.length === 0) return 90;
    
    const positive = sentimentBreakdown.find(s => s.name === 'Positive')?.value || 0;
    const negative = sentimentBreakdown.find(s => s.name === 'Negative')?.value || 0;
    const neutral = sentimentBreakdown.find(s => s.name === 'Neutral')?.value || 0;
    const total = positive + negative + neutral;
    
    if (total === 0) return 90;
    
    // Calculate weighted angle: positive pulls toward 0°, negative toward 180°
    const sentimentScore = (positive - negative) / total;
    const angle = 90 - (sentimentScore * 90); // 0° to 180° range
    return Math.max(0, Math.min(180, angle));
  };

  const SentimentMeter = () => {
    const angle = calculateMeterAngle();
    const rotation = angle - 90; // Convert to rotation for needle
    
    const getColorFromAngle = (angle) => {
      if (angle <= 60) return '#22c55e'; // Green (positive)
      if (angle <= 120) return '#f59e0b'; // Yellow (neutral)
      return '#ef4444'; // Red (negative)
    };

    const getSentimentLabel = (angle) => {
      if (angle <= 30) return 'Very Positive';
      if (angle <= 60) return 'Positive';
      if (angle <= 90) return 'Mostly Positive';
      if (angle <= 120) return 'Mixed/Neutral';
      if (angle <= 150) return 'Mostly Negative';
      return 'Negative';
    };

    return (
      <div className="flex flex-col items-center">
        <div className="relative w-64 h-32 mb-4">
          {/* Semicircle background */}
          <svg width="256" height="128" className="overflow-visible">
            {/* Background arc */}
            <path
              d="M 32 96 A 64 64 0 0 1 224 96"
              fill="none"
              stroke="#e5e7eb"
              strokeWidth="16"
              strokeLinecap="round"
            />
            
            {/* Colored segments */}
            <path
              d="M 32 96 A 64 64 0 0 1 128 32"
              fill="none"
              stroke="#22c55e"
              strokeWidth="12"
              strokeLinecap="round"
              opacity="0.3"
            />
            <path
              d="M 128 32 A 64 64 0 0 1 224 96"
              fill="none"
              stroke="#ef4444"
              strokeWidth="12"
              strokeLinecap="round"
              opacity="0.3"
            />
            
            {/* Needle */}
            <g transform={`translate(128,96) rotate(${rotation})`}>
              <line
                x1="0"
                y1="0"
                x2="0"
                y2="-50"
                stroke={getColorFromAngle(angle)}
                strokeWidth="4"
                strokeLinecap="round"
              />
              <circle
                cx="0"
                cy="0"
                r="6"
                fill={getColorFromAngle(angle)}
              />
            </g>
            
            {/* Labels */}
            <text x="32" y="110" textAnchor="middle" className="text-xs fill-green-600 font-medium">
              Positive
            </text>
            <text x="128" y="20" textAnchor="middle" className="text-xs fill-yellow-600 font-medium">
              Neutral
            </text>
            <text x="224" y="110" textAnchor="middle" className="text-xs fill-red-600 font-medium">
              Negative
            </text>
          </svg>
        </div>
        
        <div className="text-center">
          <div className="text-2xl font-bold" style={{ color: getColorFromAngle(angle) }}>
            {getSentimentLabel(angle)}
          </div>
          <div className="text-sm text-gray-600 mt-1">
            Overall Sentiment Score
          </div>
        </div>
      </div>
    );
  };

  const KeywordCloud = ({ keywords, type }) => {
    if (!keywords || keywords.length === 0) {
      return (
        <div className="text-center py-8 text-gray-500">
          No {type} keywords found
        </div>
      );
    }

    const maxScore = Math.max(...keywords.map(k => k.score || 1));
    
    return (
      <div className="flex flex-wrap gap-3 justify-center">
        {keywords.map((keywordObj, index) => {
          const keyword = typeof keywordObj === 'string' ? keywordObj : keywordObj.keyword;
          const score = typeof keywordObj === 'object' ? keywordObj.score || 1 : 1;
          const normalizedScore = score / maxScore;
          const fontSize = Math.max(0.8, normalizedScore * 1.5);
          const opacity = Math.max(0.6, normalizedScore);
          
          return (
            <div
              key={index}
              className={`px-4 py-2 rounded-full font-medium transition-all duration-200 hover:scale-110 cursor-pointer ${
                type === 'positive'
                  ? 'bg-gradient-to-r from-green-100 to-green-200 text-green-800 border border-green-300 hover:from-green-200 hover:to-green-300'
                  : 'bg-gradient-to-r from-red-100 to-red-200 text-red-800 border border-red-300 hover:from-red-200 hover:to-red-300'
              }`}
              style={{
                fontSize: `${fontSize}rem`,
                opacity: opacity,
                boxShadow: `0 2px 8px ${type === 'positive' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)'}`
              }}
              title={`Score: ${score.toFixed(3)}`}
            >
              {keyword}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="space-y-8">
      {/* Header with URL and Reset Button */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex-1">
            <h2 className="text-xl font-bold text-gray-900 mb-2">Analysis Results</h2>
            <p className="text-gray-600 break-all">{url}</p>
          </div>
          <button
            onClick={onReset}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-medium transition-colors"
          >
            Analyze Another Product
          </button>
        </div>
      </div>

      {/* Summary Card */}
      <SummaryCard data={data} />

      {/* Sentiment Meter */}
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h3 className="text-xl font-semibold text-gray-900 mb-6 text-center">Sentiment Analysis</h3>
        <SentimentMeter />
        
        {/* Sentiment Breakdown Numbers */}
        <div className="grid grid-cols-3 gap-4 mt-8 pt-6 border-t border-gray-200">
          {sentimentBreakdown.map((sentiment, index) => (
            <div key={index} className="text-center">
              <div className="text-2xl font-bold" style={{ color: sentiment.color }}>
                {sentiment.value}
              </div>
              <div className="text-sm text-gray-600 capitalize">
                {sentiment.name} Reviews
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Keywords Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Positive Keywords */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-center mb-6">
            <div className="flex items-center">
              <div className="w-4 h-4 bg-green-500 rounded-full mr-3"></div>
              <h3 className="text-lg font-semibold text-gray-900">Positive Keywords</h3>
              <div className="ml-2 bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">
                {positiveKeywords?.length || 0}
              </div>
            </div>
          </div>
          <KeywordCloud keywords={positiveKeywords} type="positive" />
        </div>

        {/* Negative Keywords */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-center mb-6">
            <div className="flex items-center">
              <div className="w-4 h-4 bg-red-500 rounded-full mr-3"></div>
              <h3 className="text-lg font-semibold text-gray-900">Negative Keywords</h3>
              <div className="ml-2 bg-red-100 text-red-800 text-xs px-2 py-1 rounded-full">
                {negativeKeywords?.length || 0}
              </div>
            </div>
          </div>
          <KeywordCloud keywords={negativeKeywords} type="negative" />
        </div>
      </div>

      {/* KeyBERT Analysis Info */}
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg p-4">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-purple-500 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h4 className="text-sm font-medium text-purple-800">KeyBERT Analysis</h4>
            <p className="text-sm text-purple-600 mt-1">
              Keywords extracted using KeyBERT with RoBERTa sentiment analysis. Size and opacity indicate relevance scores from the ML model.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;