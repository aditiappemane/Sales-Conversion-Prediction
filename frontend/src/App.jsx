import { useState } from 'react'
import './App.css'

function App() {
  const [transcript, setTranscript] = useState('')
  const [probability, setProbability] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setProbability(null)
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript }),
      })
      const data = await response.json()
      if (data.probability !== undefined) {
        setProbability(data.probability)
      } else {
        setError(data.error || 'No probability returned')
      }
    } catch (err) {
      setError('Error connecting to backend')
    }
    setLoading(false)
  }

  return (
    <div className="container">
      <h1>Sales Conversion Prediction</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          rows={6}
          value={transcript}
          onChange={e => setTranscript(e.target.value)}
          placeholder="Paste sales call transcript here..."
          required
        />
        <br />
        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict Conversion Probability'}
        </button>
      </form>
      {probability !== null && (
        <div className="result">
          <h2>Conversion Probability:</h2>
          <p>{(probability * 100).toFixed(2)}%</p>
        </div>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  )
}

export default App
