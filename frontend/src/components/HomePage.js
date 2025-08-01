import React from 'react';
import '../styles/App.css';

function HomePage({ onGetStarted }) {
  return (
    <div
      className="home-container"
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(120deg, #e0e7ff 0%, #f9fafc 100%)',
        position: 'relative',
        overflow: 'hidden',
        padding: '0 0 40px 0'
      }}
    >
      {/* Animated background blobs */}
      <div style={{
        position: 'absolute',
        top: '-100px',
        left: '-100px',
        width: 340,
        height: 340,
        background: 'radial-gradient(circle, #6366f1 0%, #e0e7ff 80%)',
        opacity: 0.18,
        borderRadius: '50%',
        zIndex: 0,
        animation: 'float 8s ease-in-out infinite alternate'
      }} />
      <div style={{
        position: 'absolute',
        bottom: '-120px',
        right: '-120px',
        width: 380,
        height: 380,
        background: 'radial-gradient(circle, #4f46e5 0%, #f9fafc 80%)',
        opacity: 0.13,
        borderRadius: '50%',
        zIndex: 0,
        animation: 'float 10s ease-in-out infinite alternate-reverse'
      }} />

      <div
        className="home-card"
        style={{
          background: '#fff',
          borderRadius: '18px',
          boxShadow: '0 4px 24px rgba(0,0,0,0.10)',
          padding: '2.5rem 3.5rem 2.5rem 3.5rem',
          maxWidth: 700,
          minWidth: 480,
          textAlign: 'center',
          zIndex: 1,
          position: 'relative',
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        {/* Animated logo */}
        <div
          style={{
            fontSize: 60,
            marginBottom: 12,
            animation: 'logoPop 1.2s cubic-bezier(.68,-0.55,.27,1.55)'
          }}
        >
          <span role="img" aria-label="EduQuery Logo">📘</span>
        </div>
        <h1
          style={{
            fontWeight: 800,
            fontSize: 38,
            marginBottom: 10,
            color: '#2d3a4a',
            letterSpacing: '-1px',
            animation: 'fadeInDown 1s'
          }}
        >
          Welcome to EduQuery
        </h1>
        <div style={{ color: '#6366f1', fontWeight: 600, fontSize: 20, marginBottom: 18, animation: 'fadeIn 1.2s' }}>
          <span style={{ fontStyle: 'italic' }}>“Your AI-powered PDF learning companion”</span>
        </div>
        <div style={{
          display: 'flex',
          flexDirection: 'row',
          gap: 28,
          marginBottom: 32,
          marginTop: 18,
          justifyContent: 'center',
          flexWrap: 'wrap',
        }}>
          {/* Animated feature cards */}
          <div className="feature-card" style={{
            background: '#f3f4f6',
            borderRadius: 12,
            padding: '18px 32px',
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            boxShadow: '0 2px 8px rgba(99,102,241,0.07)',
            fontSize: 18,
            fontWeight: 500,
            color: '#374151',
            animation: 'fadeInLeft 1.1s',
            minWidth: 220,
            marginBottom: 10
          }}>
            <span style={{ fontSize: 26, color: '#6366f1' }}>📤</span>
            Upload any PDF and start chatting instantly
          </div>
          <div className="feature-card" style={{
            background: '#f3f4f6',
            borderRadius: 12,
            padding: '18px 32px',
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            boxShadow: '0 2px 8px rgba(99,102,241,0.07)',
            fontSize: 18,
            fontWeight: 500,
            color: '#374151',
            animation: 'fadeInLeft 1.3s',
            minWidth: 220,
            marginBottom: 10
          }}>
            <span style={{ fontSize: 26, color: '#6366f1' }}>💡</span>
            Get answers, summaries, and explanations
          </div>
          <div className="feature-card" style={{
            background: '#f3f4f6',
            borderRadius: 12,
            padding: '18px 32px',
            display: 'flex',
            alignItems: 'center',
            gap: 14,
            boxShadow: '0 2px 8px rgba(99,102,241,0.07)',
            fontSize: 18,
            fontWeight: 500,
            color: '#374151',
            animation: 'fadeInLeft 1.5s',
            minWidth: 220,
            marginBottom: 10
          }}>
            <span style={{ fontSize: 26, color: '#6366f1' }}>🔒</span>
            No registration required – just start learning!
          </div>
        </div>
        {/* Animated Get Started button */}
        <button
          onClick={onGetStarted}
          style={{
            background: 'linear-gradient(90deg, #6366f1 0%, #4f46e5 100%)',
            color: '#fff',
            border: 'none',
            borderRadius: '10px',
            padding: '18px 60px',
            fontSize: 22,
            fontWeight: 700,
            cursor: 'pointer',
            boxShadow: '0 2px 12px rgba(99,102,241,0.13)',
            transition: 'transform 0.1s',
            marginTop: 18,
            animation: 'popIn 1.7s'
          }}
          onMouseDown={e => e.currentTarget.style.transform = 'scale(0.97)'}
          onMouseUp={e => e.currentTarget.style.transform = 'scale(1)'}
        >
          🚀 Get Started
        </button>
      </div>
      <footer style={{ marginTop: 40, color: '#6b7280', fontSize: 16, zIndex: 1 }}>
        EduQuery &copy; {new Date().getFullYear()}
      </footer>
      {/* Keyframes for floating animation and other effects */}
      <style>
        {`
          @keyframes float {
            0% { transform: translateY(0px);}
            100% { transform: translateY(30px);}
          }
          @keyframes logoPop {
            0% { transform: scale(0.5) rotate(-10deg); opacity: 0; }
            60% { transform: scale(1.15) rotate(5deg); opacity: 1; }
            100% { transform: scale(1) rotate(0deg); opacity: 1; }
          }
          @keyframes fadeInDown {
            0% { opacity: 0; transform: translateY(-30px);}
            100% { opacity: 1; transform: translateY(0);}
          }
          @keyframes fadeIn {
            0% { opacity: 0;}
            100% { opacity: 1;}
          }
          @keyframes fadeInLeft {
            0% { opacity: 0; transform: translateX(-40px);}
            100% { opacity: 1; transform: translateX(0);}
          }
          @keyframes popIn {
            0% { opacity: 0; transform: scale(0.7);}
            100% { opacity: 1; transform: scale(1);}
          }
        `}
      </style>
    </div>
  );
}

export default HomePage;