import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import App from './App.jsx'

const Demo = lazy(() => import('./Demo.jsx'))
const CustomerValidation = lazy(() => import('./CustomerValidation.jsx'))
const Contact = lazy(() => import('./Contact.jsx'))
const Support = lazy(() => import('./Support.jsx'))
const Features = lazy(() => import('./Features.jsx'))
const Docs = lazy(() => import('./Docs.jsx'))
const Founders = lazy(() => import('./Founders.jsx'))

const PrivacyPolicy = lazy(() => import('./Policies.jsx').then(m => ({ default: m.PrivacyPolicy })))
const TermsConditions = lazy(() => import('./Policies.jsx').then(m => ({ default: m.TermsConditions })))
const RefundsPolicy = lazy(() => import('./Policies.jsx').then(m => ({ default: m.RefundsPolicy })))
const ShippingPolicy = lazy(() => import('./Policies.jsx').then(m => ({ default: m.ShippingPolicy })))

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Suspense fallback={null}>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/demo" element={<Demo />} />
          <Route path="/validation" element={<CustomerValidation />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/support" element={<Support />} />
          <Route path="/features" element={<Features />} />
          <Route path="/docs" element={<Docs />} />
          <Route path="/founders" element={<Founders />} />
          <Route path="/privacy" element={<PrivacyPolicy />} />
          <Route path="/terms" element={<TermsConditions />} />
          <Route path="/refunds" element={<RefundsPolicy />} />
          <Route path="/shipping" element={<ShippingPolicy />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  </StrictMode>,
)
