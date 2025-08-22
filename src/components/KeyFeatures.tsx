'use client';

import React, { useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Shield, Brain, CheckCircle } from 'lucide-react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';

export default function KeyFeatures() {
  const featuresRef = useRef<HTMLDivElement>(null);

  useGSAP(() => {
    gsap.fromTo('.feature-card', 
      { y: 50, opacity: 0, scale: 0.9 },
      { y: 0, opacity: 1, scale: 1, duration: 0.8, ease: "back.out(1.7)", stagger: 0.2 }
    );
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
      {/* Key Features */}
      <div ref={featuresRef} className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <Card className="feature-card border-2 hover:border-primary/50 transition-colors">
          <CardContent className="p-6 text-center space-y-4">
            <div className="w-16 h-16 mx-auto bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center">
              <Shield className="h-8 w-8 text-blue-600" />
            </div>
            <h3 className="text-xl font-semibold">Privacy First</h3>
            <p className="text-muted-foreground">
              Videos are processed locally and immediately deleted. No data storage, complete privacy protection.
            </p>
          </CardContent>
        </Card>

        <Card className="feature-card border-2 hover:border-primary/50 transition-colors">
          <CardContent className="p-6 text-center space-y-4">
            <div className="w-16 h-16 mx-auto bg-purple-100 dark:bg-purple-900/20 rounded-full flex items-center justify-center">
              <Brain className="h-8 w-8 text-purple-600" />
            </div>
            <h3 className="text-xl font-semibold">AI-Powered Analysis</h3>
            <p className="text-muted-foreground">
              Advanced machine learning models trained on behavioral patterns and developmental indicators.
            </p>
          </CardContent>
        </Card>

        <Card className="feature-card border-2 hover:border-primary/50 transition-colors">
          <CardContent className="p-6 text-center space-y-4">
            <div className="w-16 h-16 mx-auto bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center">
              <CheckCircle className="h-8 w-8 text-green-600" />
            </div>
            <h3 className="text-xl font-semibold">Clinical Support</h3>
            <p className="text-muted-foreground">
              Designed to assist healthcare professionals, not replace clinical judgment and evaluation.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
