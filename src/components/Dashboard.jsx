import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import SummaryCard from './SummaryCard';

const Dashboard = ({ data, onReset }) => {
  const { sentimentBreakdown, positiveKeywords, negativeKeywords, url } = data;

  const KeywordTag = ({ keyword, type }) => (
    <span
      className={`inline-block px-3 py-1 rounded-full text-sm font-medium mr-2 mb-2 ${
        type === 'positive'
          ? 'bg-green-100 text-green-800 border border-green-200'
          : 'bg-red-100 text-red-800 border border-red-200'
      }`}
    >
      {keyword}
    </span>
  );

  const renderCustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium">{data.name}</p>
          <p className="text-sm text-gray-600">{data.value}% of reviews</p>
        </div>
      );
    }
    return null;
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

      {/* Charts and Keywords Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Sentiment Breakdown Chart */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Breakdown</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={sentimentBreakdown}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}%`}
                >
                  {sentimentBreakdown.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={renderCustomTooltip} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Keywords Section */}
        <div className="space-y-6">
          {/* Positive Keywords */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <span className="text-green-500 mr-2">✓</span>
              <h3 className="text-lg font-semibold text-gray-900">Top Positive Keywords</h3>
            </div>
            <div className="flex flex-wrap">
              {positiveKeywords && positiveKeywords.length > 0 ? (
                positiveKeywords.slice(0, 8).map((keywordObj, index) => (
                  <KeywordTag 
                    key={index} 
                    keyword={typeof keywordObj === 'string' ? keywordObj : keywordObj.keyword} 
                    type="positive" 
                  />
                ))
              ) : (
                <span className="text-gray-500 italic">No positive keywords found</span>
              )}
            </div>
          </div>

          {/* Negative Keywords */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <span className="text-red-500 mr-2">✗</span>
              <h3 className="text-lg font-semibold text-gray-900">Top Negative Keywords</h3>
            </div>
            <div className="flex flex-wrap">
              {negativeKeywords && negativeKeywords.length > 0 ? (
                negativeKeywords.slice(0, 8).map((keywordObj, index) => (
                  <KeywordTag 
                    key={index} 
                    keyword={typeof keywordObj === 'string' ? keywordObj : keywordObj.keyword} 
                    type="negative" 
                  />
                ))
              ) : (
                <span className="text-gray-500 italic">No negative keywords found</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* API Integration Note */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-blue-400 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h4 className="text-sm font-medium text-blue-800">Demo Mode</h4>
            <p className="text-sm text-blue-600 mt-1">
              This is showing mock data. Connect to your backend API at <code className="bg-blue-100 px-1 rounded">POST /analyze</code> to analyze real reviews.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;