// frontend/src/ForecastChart.js

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';

function ForecastChart({ symbol }) {
  const [forecast, setForecast] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Replace the port if your Flask server is hosted elsewhere
    axios.get(`http://localhost:5001/api/forecast?symbol=${symbol}&days=30`)
      .then(response => {
        setForecast(response.data.forecast);
        setLoading(false);
      })
      .catch(err => {
        console.error("Error fetching forecast data", err);
        setError("Error fetching forecast data");
        setLoading(false);
      });
  }, [symbol]);

  // Prepare data for the chart
  const chartData = {
    labels: forecast.map(item => item.date),
    datasets: [
      {
        label: 'Predicted Close Price',
        data: forecast.map(item => item.predicted_close),
        fill: false,
        borderColor: 'red'
      }
    ]
  };

  if (loading) return <p>Loading forecast...</p>;
  if (error) return <p>{error}</p>;

  return (
    <div>
      <h2>{symbol} Future Outlook</h2>
      <Line data={chartData} />
    </div>
  );
}

export default ForecastChart;
