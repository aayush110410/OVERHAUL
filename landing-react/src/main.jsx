import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import App from './App.jsx'
import Demo from './Demo.jsx'
import Contact from './Contact.jsx'
import Support from './Support.jsx'
import { PrivacyPolicy, TermsConditions, RefundsPolicy, ShippingPolicy } from './Policies.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/demo" element={<Demo />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/support" element={<Support />} />
        <Route path="/privacy" element={<PrivacyPolicy />} />
        <Route path="/terms" element={<TermsConditions />} />
        <Route path="/refunds" element={<RefundsPolicy />} />
        <Route path="/shipping" element={<ShippingPolicy />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
