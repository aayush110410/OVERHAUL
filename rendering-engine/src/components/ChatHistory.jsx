import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'

export default function ChatHistory({ messages, onSendFollowUp, isAnalyzing }) {
  const [followUp, setFollowUp] = useState('')
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function handleSend() {
    const text = followUp.trim()
    if (!text || isAnalyzing) return
    onSendFollowUp(text)
    setFollowUp('')
  }

  function handleKeyDown(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      handleSend()
    }
  }

  if (!messages || messages.length === 0) return null

  return (
    <div className="ov-chat-history">
      <div className="ov-chat-header">
        <span className="demo-card-title">CONVERSATION</span>
        <span className="demo-card-tag">{messages.length} TURNS</span>
      </div>

      <div className="ov-chat-messages">
        <AnimatePresence>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              className={`ov-chat-msg ov-chat-msg-${msg.role}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <div className="ov-chat-msg-header">
                <span className="ov-chat-msg-role">
                  {msg.role === 'user' ? '→ YOU' : '◈ OVERHAUL'}
                </span>
                {msg.timestamp && (
                  <span className="ov-chat-msg-time">{msg.timestamp}</span>
                )}
              </div>
              <div className="ov-chat-msg-content">
                {msg.role === 'assistant' ? (
                  <div className="markdown-content">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <p>{msg.content}</p>
                )}
              </div>
              {msg.stats && (
                <div className="ov-chat-msg-stats">
                  {msg.stats.runtime && <span>⏱ {msg.stats.runtime}s</span>}
                  {msg.stats.engines && <span>🔧 {msg.stats.engines} engines</span>}
                  {msg.stats.model && <span>🤖 {msg.stats.model}</span>}
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={endRef} />
      </div>

      {/* Follow-up Input */}
      <div className="ov-chat-input-row">
        <input
          type="text"
          className="ov-chat-input"
          placeholder="Ask a follow-up question..."
          value={followUp}
          onChange={e => setFollowUp(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isAnalyzing}
        />
        <button
          className="ov-chat-send-btn"
          onClick={handleSend}
          disabled={!followUp.trim() || isAnalyzing}
        >
          {isAnalyzing ? <span className="demo-spinner" /> : '→'}
        </button>
      </div>
    </div>
  )
}
