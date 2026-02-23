import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// 1. Define the shape of coordinate data
interface Point {
  x: number;
  y: number;
}


const App: React.FC = () => {
  const [k, setK] = useState<number>(1);
  const [backendData, setBackendData] = useState<Point[]>([]);
  /**
   * Make network call
   * Method: POST
   * URL: http://localhost:8000/calculate
   * Header: 'Content-Type': 'application/json'
   * Body JSON: { k: slope }
   * 
   * Wait for backend to send calculation back
   * update state
   * @param slope 
   */

  const fetchPrediction = async (slope: number) => {
    try {
      const response = await fetch('http://localhost:8000/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ k: slope }),
      });
      const result = await response.json();
      setBackendData(result.data);
    } catch (error) {
      console.error("Backend offline? Falling back to local calculation.", error);
    }
  };

  // 2. Trigger backend call whenever k changes
  // slider changes -> state k changes -> triggers re-render, useEffect: run the fetchPrediction effect whenever k changes -> fetchPrediction: API call
  useEffect(() => {
    fetchPrediction(k);
  }, [k]);


  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setK(isNaN(val) ? 0:val);
  };

  return (
    <div style={{ padding: '40px', fontFamily: 'system-ui, sans-serif' }}>
      <h2>Interactive Slope Demo</h2>
      <p>Equation: <strong>y = {k.toFixed(1)}x</strong></p>
      
      <div style={{ marginBottom: '20px', background: '#f4f4f4', padding: '15px', borderRadius: '8px' }}>
        <label style={{ fontWeight: 'bold' }}>Adjust Slope (k): </label>
        <input 
          type="range" //slider
          min="0" 
          max="5" 
          step="0.1"
          value={k} 
          onChange={handleInputChange}
          style={{ width: '200px', marginLeft: '10px' }}
        />
        <span style={{ marginLeft: '10px' }}>{k}</span>
      </div>
      
     <div style={{ width: '600px', height: '400px', backgroundColor: '#fff', border: '1px solid #ddd' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={backendData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={true} />
            <XAxis 
              dataKey="x" 
              type="number" 
              domain={[0, 20]} 
              ticks={[0, 5, 10, 15, 20]}
            />
            <YAxis 
              type="number" 
              domain={[0, 20]} 
              allowDataOverflow={true}
              ticks={[0, 5, 10, 15, 20]}
            />
            <Tooltip />
            <Line 
              type="linear" 
              dataKey="y" 
              stroke="#2563eb" 
              strokeWidth={3}
              dot={false}
              animationDuration={500} // Smooth transition for the 'rotation'
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default App;
