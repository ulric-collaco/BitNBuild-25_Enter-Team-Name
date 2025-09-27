# Review Radar

A React frontend for analyzing product reviews with sentiment analysis.

## Features

- **Clean UI**: Simple, centered input form for product page URLs
- **Sentiment Analysis Dashboard**: 
  - Pie chart showing sentiment breakdown (Positive, Negative, Neutral)
  - Color-coded keyword tags for positive/negative terms
  - Overall sentiment score display
- **Responsive Design**: Works on desktop and mobile devices
- **Loading States**: Smooth loading animations during analysis
- **Mock Data**: Ready for backend integration with placeholder API calls

## Tech Stack

- React 18
- TailwindCSS for styling
- Recharts for data visualization
- Functional components with hooks

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Open [http://localhost:3000](http://localhost:3000) to view in the browser.

## Backend Integration

The app is ready for backend integration. Update the `handleAnalyze` function in `App.jsx` to call your actual API:

```javascript
// Replace the mock API call in App.jsx
const response = await fetch('/api/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ url }),
});
const data = await response.json();
```

Expected API response format:
```javascript
{
  url: string,
  sentimentBreakdown: [
    { name: 'Positive', value: 65, color: '#22c55e' },
    { name: 'Negative', value: 20, color: '#ef4444' },
    { name: 'Neutral', value: 15, color: '#6b7280' }
  ],
  positiveKeywords: string[],
  negativeKeywords: string[],
  overallSentiment: {
    score: number, // 0-100
    label: string   // e.g., "Very Positive"
  }
}
```

## File Structure

```
src/
├── App.jsx                 # Main app component with state management
├── index.js               # React app entry point
├── index.css              # TailwindCSS imports
└── components/
    ├── UrlInput.jsx       # URL input form with validation
    ├── Dashboard.jsx      # Results dashboard with chart and keywords
    └── SummaryCard.jsx    # Overall sentiment summary display
```

## Components

### UrlInput
- Validates URLs before submission
- Shows loading spinner during analysis
- Handles form submission and error states

### Dashboard
- Displays sentiment breakdown pie chart
- Shows positive/negative keyword tags
- Includes reset functionality to analyze another product

### SummaryCard
- Shows overall sentiment score and label
- Dynamic styling based on sentiment score
- Emoji indicators for different sentiment levels

## Customization

- Colors and styling can be modified in the TailwindCSS classes
- Chart appearance can be customized through Recharts props
- Mock data structure can be adjusted in `App.jsx`

## Production Build

```bash
npm run build
```

This creates a `build` folder with optimized production files ready for deployment.