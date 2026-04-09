import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import './index.css'
import App from './App.jsx'

const Demo = lazy(() => import('./Demo.jsx'))
const CustomerValidation = lazy(() => import('./CustomerValidation.jsx'))
const Contact = lazy(() => import('./Contact.jsx'))
const Support = lazy(() => import('./Support.jsx'))
const Features = lazy(() => import('./Features.jsx'))
const Docs = lazy(() => import('./Docs.jsx'))
const Founders = lazy(() => import('./Founders.jsx'))
const FlyoverSim = lazy(() => import('./FlyoverSim.jsx'))
const Demo2 = lazy(() => import('./Demo2.jsx'))

const PrivacyPolicy = lazy(() => import('./Policies.jsx').then(m => ({ default: m.PrivacyPolicy })))
const TermsConditions = lazy(() => import('./Policies.jsx').then(m => ({ default: m.TermsConditions })))
const RefundsPolicy = lazy(() => import('./Policies.jsx').then(m => ({ default: m.RefundsPolicy })))
const ShippingPolicy = lazy(() => import('./Policies.jsx').then(m => ({ default: m.ShippingPolicy })))

// Loading component for lazy-loaded pages
const PageLoader = () => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100vh',
    background: '#0a0a0a',
    color: '#c8ff00',
    fontFamily: 'JetBrains Mono, monospace',
    fontSize: '1.2rem'
  }}>
    <div style={{ textAlign: 'center' }}>
      <div style={{ marginBottom: '1rem', fontSize: '2rem' }}>⚡</div>
      <div>LOADING OVERHAUL...</div>
    </div>
  </div>
)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Suspense fallback={<PageLoader />}>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/demo" element={<Demo />} />
          <Route path="/demo2" element={<Demo2 />} />
          <Route path="/validation" element={<CustomerValidation />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/support" element={<Support />} />
          <Route path="/features" element={<Features />} />
          <Route path="/docs" element={<Docs />} />
          <Route path="/founders" element={<Founders />} />
          <Route path="/flyover" element={<FlyoverSim />} />
          <Route path="/privacy" element={<PrivacyPolicy />} />
          <Route path="/terms" element={<TermsConditions />} />
          <Route path="/refunds" element={<RefundsPolicy />} />
          <Route path="/shipping" element={<ShippingPolicy />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  </StrictMode>,
)
