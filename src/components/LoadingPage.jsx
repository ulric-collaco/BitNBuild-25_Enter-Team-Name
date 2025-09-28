import React from 'react';
import LightRays from '../LightRays';

const LoadingPage = ({ url }) => {
  return (
    <div className="min-h-screen bg-gray-900 relative flex items-center justify-center">
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

      {/* Loading Content */}
      <div className="relative z-10 max-w-2xl mx-auto px-4 text-center">
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 p-12 shadow-2xl">
          
          {/* Animated Loading Spinner */}
          <div className="mb-8">
            <div className="relative mx-auto w-24 h-24">
              <div className="absolute inset-0 border-4 border-white/20 rounded-full"></div>
              <div className="absolute inset-0 border-4 border-transparent border-t-white rounded-full animate-spin"></div>
              <div className="absolute inset-2 border-2 border-transparent border-t-blue-400 rounded-full animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}></div>
            </div>
          </div>

          {/* Loading Text */}
          <h2 className="text-3xl font-bold text-white mb-4">
            Analyzing Reviews
          </h2>
          
          <p className="text-lg text-gray-300 mb-6">
            Our AI is processing customer reviews and extracting insights...
          </p>

          {/* URL Display */}
          <div className="bg-white/5 rounded-lg p-4 mb-6">
            <p className="text-sm text-gray-400 mb-2">Analyzing URL:</p>
            <p className="text-white font-mono text-sm break-all">
              {url}
            </p>
          </div>

          {/* Progress Steps */}
          <div className="space-y-3">
            <div className="flex items-center text-sm text-gray-300">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-3 animate-pulse"></div>
              <span>Fetching customer reviews...</span>
            </div>
            <div className="flex items-center text-sm text-gray-300">
              <div className="w-2 h-2 bg-yellow-500 rounded-full mr-3 animate-pulse"></div>
              <span>Analyzing sentiment patterns...</span>
            </div>
            <div className="flex items-center text-sm text-gray-300">
              <div className="w-2 h-2 bg-blue-500 rounded-full mr-3 animate-pulse"></div>
              <span>Extracting key insights...</span>
            </div>
          </div>

          {/* Estimated Time */}
          <div className="mt-8 text-sm text-gray-400">
            <p>This usually takes 30-60 seconds</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingPage;