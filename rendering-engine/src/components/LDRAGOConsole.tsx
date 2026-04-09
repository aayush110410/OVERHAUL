import React, { useState } from 'react';

export default function LDRAGOConsole() {
  const [messages, setMessages] = useState([
    { role: 'ldrago', text: 'Connecting to neural net...\nSystem ready. Awaiting command.' }
  ]);
  const [inputStr, setInputStr] = useState('');

  const handleSend = () => {
    if (!inputStr.trim()) return;
    setMessages(prev => [...prev, { role: 'user', text: inputStr }]);
    
    setTimeout(() => {
      setMessages(prev => [...prev, {
        role: 'ldrago', 
        text: 'Analyzing directive. Re-routing traffic grids to alleviate congestion.'
      }]);
    }, 1000);
    setInputStr('');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, overflowY: 'auto', paddingBottom: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {messages.map((m, i) => (
          <div key={i} style={{ 
            padding: '10px 15px', 
            borderRadius: '8px',
            background: m.role === 'ldrago' ? 'rgba(0, 150, 255, 0.1)' : 'rgba(255, 255, 255, 0.05)',
            borderLeft: m.role === 'ldrago' ? '2px solid rgba(0, 150, 255, 0.5)' : 'none',
            fontSize: '13px',
            color: 'rgba(255,255,255,0.85)',
            whiteSpace: 'pre-wrap'
          }}>
            {m.text}
          </div>
        ))}
      </div>
      
      <div style={{ marginTop: 'auto', display: 'flex', gap: '10px' }}>
        <input 
          type="text" 
          value={inputStr}
          onChange={(e) => setInputStr(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="SEND DIRECTIVE..."
          style={{
            flex: 1,
            background: 'rgba(0,0,0,0.5)',
            border: '1px solid rgba(255,255,255,0.2)',
            padding: '10px',
            color: 'white',
            borderRadius: '4px',
            outline: 'none',
            fontFamily: 'monospace'
          }}
        />
        <button 
          onClick={handleSend}
          style={{
            background: 'rgba(0, 150, 255, 0.2)',
            border: '1px solid rgba(0, 150, 255, 0.5)',
            color: '#a3d8ff',
            padding: '0 15px',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          EXECUTE
        </button>
      </div>
    </div>
  );
}
