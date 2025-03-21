const express = require('express');
const yahooFinance = require('yahoo-finance2').default; // Updated library
const { Parser } = require('json2csv');
const cors = require('cors');
const fs = require('fs');
const { exec } = require('child_process');
const path = require('path'); // Import path module

const app = express();
app.use(cors());

const PORT = process.env.PORT || 5001;

const csvDirectory = '/Users/hashemalomar/Desktop/LSTM-1/backend/csvs';

app.get('/api/stock', async (req, res) => {
  const { symbol } = req.query;

  if (!symbol) {
    return res.status(400).json({ error: 'Ticker symbol is required' });
  }

  // Define date range: Last 5 years from today
  const endDate = new Date();
  const startDate = new Date();
  startDate.setFullYear(startDate.getFullYear() - 5);

  try {
    // Fetch historical data using yahoo-finance2
    const queryOptions = { period1: startDate, period2: endDate, interval: "1d" };
    const quotes = await yahooFinance.historical(symbol.toUpperCase(), queryOptions);

    if (!quotes || quotes.length === 0) {
      return res.status(404).json({ error: 'No data found for this symbol' });
    }

    // Convert the data to CSV format
    const fields = ['date', 'open', 'high', 'low', 'close', 'volume'];
    const parser = new Parser({ fields });
    const csv = parser.parse(quotes);

    // Define the file name so that main.py can read it.
    // For example, if symbol is NVDA, the file name becomes "NVDA_5_years.csv".
    const fileName = `${symbol.toUpperCase()}_5_years.csv`;
    const filePath = path.join(csvDirectory, fileName);
    // Write CSV to disk
    fs.writeFile(filePath, csv, (err) => {
      if (err) {
        console.error('Error writing CSV file:', err);
        return res.status(500).json({ error: 'Error writing CSV file' });
      }

      console.log(`CSV file saved as ${fileName}`);


      const command = `python /Users/hashemalomar/Desktop/LSTM-1/LSTModel/main.py ${fileName}`;
      exec(command, (error, stdout, stderr) => {
        if (error) {
          console.error(`Error executing Python script: ${error.message}`);
          return res.status(500).json({ error: error.message });
        }
        if (stderr) {
          console.error(`Python script stderr: ${stderr}`);
          // You can choose to return stderr as part of the response if needed.
        }
        // Return the output of the Python script to the client.
        return res.json({ stdout });
      });
    });
  } catch (error) {
    console.error('Error fetching stock data:', error);
    return res.status(500).json({ error: 'Error fetching stock data' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
