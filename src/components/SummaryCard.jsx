import React from 'react';

const SummaryCard = ({ data }) => {
  const { overallSentiment } = data;

  const getSentimentColor = (score) => {
    if (score >= 80) return 'text-green-600 bg-green-50 border-green-200';
    if (score >= 60) return 'text-blue-600 bg-blue-50 border-blue-200';
    if (score >= 40) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  const getSentimentIcon = (score) => {
    if (score >= 80) return '😊';
    if (score >= 60) return '🙂';
    if (score >= 40) return '😐';
    return '😞';
  };

  return (
    <div className={`bg-white rounded-lg shadow-lg p-6 border-2 ${getSentimentColor(overallSentiment.score)}`}>
      <div className="text-center">
        <div className="text-4xl mb-2">{getSentimentIcon(overallSentiment.score)}</div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Overall Sentiment</h3>
        <div className="flex items-center justify-center space-x-2">
          <span className="text-3xl font-bold">{overallSentiment.score}%</span>
          <span className="text-lg font-medium">{overallSentiment.label}</span>
        </div>
        <p className="text-sm text-gray-600 mt-2">
          Based on sentiment analysis of customer reviews
        </p>
      </div>
    </div>
  );
};

export default SummaryCard;