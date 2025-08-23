'use client';

import React, { useRef } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';

export default function HowItWorks() {
  const sectionRef = useRef<HTMLDivElement>(null);

  useGSAP(() => {
    gsap.fromTo('.step-item', 
      { y: 50, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.8, ease: "power3.out", stagger: 0.3 }
    );
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
      {/* How It Works */}
      <div ref={sectionRef} className="text-center">
        <h2 className="select-none pointer-events-none text-3xl md:text-4xl font-bold mb-12">How It Works</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="step-item space-y-4">
            <div className="select-none pointer-events-none w-12 h-12 mx-auto bg-blue-500 text-white rounded-full flex items-center justify-center font-bold text-xl">
              1
            </div>
            <h3 className="select-none pointer-events-none select-none pointer-events-none text-xl font-semibold">Upload Video</h3>
            <p className="select-none pointer-events-none text-muted-foreground">
              Upload a classroom activity video for analysis. Supports MP4, AVI, and MOV formats.
            </p>
          </div>
          
          <div className="step-item space-y-4">
            <div className="select-none pointer-events-none w-12 h-12 mx-auto bg-purple-500 text-white rounded-full flex items-center justify-center font-bold text-xl">
              2
            </div>
            <h3 className="text-xl font-semibold select-none pointer-events-none">AI Analysis</h3>
            <p className="select-none pointer-events-none text-muted-foreground">
              Our AI analyzes behavioral patterns, social interactions, and developmental indicators.
            </p>
          </div>
          
          <div className="step-item space-y-4">
            <div className="select-none pointer-events-none w-12 h-12 mx-auto bg-green-500 text-white rounded-full flex items-center justify-center font-bold text-xl">
              3
            </div>
            <h3 className="select-none pointer-events-none text-xl font-semibold">Get Results</h3>
            <p className="select-none pointer-events-none text-muted-foreground">
              Receive detailed analysis with confidence scores and actionable recommendations.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
