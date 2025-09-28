import React, { useState } from 'react';

const UrlInput = ({ onAnalyze }) => {
  const [url, setUrl] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (url.trim()) {
      onAnalyze(url.trim());
    }
  };

  const isValidUrl = (string) => {
    try {
      new URL(string);
      return true;
    } catch (_) {
      return false;
    }
  };

  const canSubmit = url.trim() && isValidUrl(url.trim());

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold text-white mb-6">
          Review Radar
        </h2>
        <p className="text-xl text-gray-300 max-w-2xl mx-auto leading-relaxed">
          Paste any product URL to instantly analyze customer sentiment and discover key insights
        </p>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-8 shadow-2xl">
        <form onSubmit={handleSubmit} className="space-y-8">
          <div>
            <input
              type="url"
              id="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://amazon.com/product-example"
              className="w-full px-6 py-4 bg-white/90 backdrop-blur border-0 rounded-xl text-gray-900 placeholder-gray-500 text-lg focus:ring-2 focus:ring-white/30 focus:bg-white outline-none transition-all duration-200 shadow-lg"
            />
            {url.trim() && !isValidUrl(url.trim()) && (
              <p className="mt-3 text-sm text-red-300 bg-red-500/20 rounded-lg px-4 py-2">
                Please enter a valid URL starting with https://
              </p>
            )}
          </div>

          <button
            type="submit"
            disabled={!canSubmit}
            className={`w-full py-4 px-8 rounded-xl font-semibold text-lg transition-all duration-200 ${
              canSubmit
                ? 'bg-[#517985] hover:bg-[#3e5c66] text-white shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
                : 'bg-white/20 text-gray-400 cursor-not-allowed'
            }`}
          >
            Analyze Reviews
          </button>
        </form>


      </div>
    </div>
  );
};

export default UrlInput;