// frontend/src/App.js

import React, { useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  TimeScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import 'chartjs-adapter-date-fns';

// Register Chart.js components, including the time scale.
ChartJS.register(TimeScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function App() {
  const [symbol, setSymbol] = useState('');
  const [error, setError] = useState(null);
  // dataSets will hold both historical and forecast data.
  const [dataSets, setDataSets] = useState(null);
  const [loading, setLoading] = useState(false);
  // timeSpan holds the selected time span for historical data.
  const [timeSpan, setTimeSpan] = useState('1Y'); // default to 1 Year

  const handleProcess = async () => {
    if (!symbol) {
      setError('Please enter a ticker symbol.');
      return;
    }
    setError(null);
    setDataSets(null);
    setLoading(true);

    try {
      // Fetch forecast (and historical) data from the Node backend.
      const response = await axios.get(`http://localhost:5001/api/stock?symbol=${symbol}`);
      // Expecting the response to be an object with a 'stdout' key containing the JSON string.
      if (response.data.stdout) {
        const parsedData = JSON.parse(response.data.stdout);
        // Ensure that both historical and forecast data exist.
        if (parsedData.historical && parsedData.forecast) {
          setDataSets(parsedData);
        } else {
          setError('Incomplete data in response.');
        }
      } else {
        setError('Response format not recognized.');
      }
    } catch (err) {
      console.error(err);
      setError('Error processing CSV through main.py');
    } finally {
      setLoading(false);
    }
  };

  // Function to filter historical data based on the selected time span.
  const getFilteredHistorical = () => {
    if (!dataSets || !dataSets.historical || dataSets.historical.length === 0) return [];
    // Assume historical data is sorted by date ascending.
    const historical = dataSets.historical;
    // Use the latest available historical date as a reference.
    const lastDate = new Date(historical[historical.length - 1].date);
    let startDate = new Date(lastDate);

    // Adjust the startDate based on the selected time span.
    switch (timeSpan) {
      case '1M':
        startDate.setMonth(startDate.getMonth() - 1);
        break;
      case '3M':
        startDate.setMonth(startDate.getMonth() - 3);
        break;
      case '6M':
        startDate.setMonth(startDate.getMonth() - 6);
        break;
      case '1Y':
        startDate.setFullYear(startDate.getFullYear() - 1);
        break;
      case '5Y':
        startDate.setFullYear(startDate.getFullYear() - 5);
        break;
      default:
        break;
    }
    // Filter historical data to include only data with date >= startDate.
    return historical.filter(item => new Date(item.date) >= startDate);
  };

  // Prepare chart data if we have both historical and forecast datasets.
  const chartData = dataSets
    ? {
        datasets: [
          {
            label: 'Historical Close Price',
            // Map filtered historical data into { x, y } format.
            data: getFilteredHistorical().map(item => ({
              x: item.date,
              y: item.close,
            })),
            borderColor: 'green',
            backgroundColor: 'rgba(0, 128, 0, 0.2)',
            fill: false,
          },
          {
            label: 'Forecasted Close Price',
            data: dataSets.forecast.map(item => ({
              x: item.date,
              y: item.predicted_close,
            })),
            borderColor: 'blue',
            backgroundColor: 'rgba(0, 0, 255, 0.2)',
            fill: false,
          },
        ],
      }
    : null;

  // Set chart options with a time scale on the x-axis.
  const options = {
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day',
        },
        title: {
          display: true,
          text: 'Date',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Price',
        },
      },
    },
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Stock Data Processor</h1>
      <p>
        Enter a Fortune 500 company's ticker symbol (e.g., <strong>AAPL</strong>):
      </p>
      <input
        type="text"
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
        placeholder="Enter ticker symbol"
        style={{ padding: '8px', fontSize: '16px' }}
      />
      <br />
      <br />
      {/* Drop down menu to select time span */}
      <label>
        Select Time Span:{' '}
        <select value={timeSpan} onChange={(e) => setTimeSpan(e.target.value)}>
          <option value="1M">1 Month</option>
          <option value="3M">3 Months</option>
          <option value="6M">6 Months</option>
          <option value="1Y">1 Year</option>
          <option value="5Y">5 Years</option>
        </select>
      </label>
      <br />
      <br />
      <button
        onClick={handleProcess}
        style={{ marginLeft: '10px', padding: '8px 12px', fontSize: '16px' }}
      >
        Process CSV with Python Model
      </button>
      {loading && <p style={{ color: 'blue' }}>Loading model, please wait...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {chartData && (
        <div>
          <h3>Historical and Forecast Chart:</h3>
          <Line data={chartData} options={options} />
        </div>
      )}
    </div>
  );
}

export default App;
