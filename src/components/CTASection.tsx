'use client';

import React, { useRef } from 'react';
import { Button } from '@/components/ui/button';
import { ArrowRight } from 'lucide-react';
import Link from 'next/link';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';

export default function CTASection() {
  const ctaRef = useRef<HTMLDivElement>(null);

  useGSAP(() => {
    gsap.fromTo('.cta-content', 
      { y: 50, opacity: 0, scale: 0.95 },
      { y: 0, opacity: 1, scale: 1, duration: 1, ease: "power3.out" }
    );
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
      {/* CTA Section */}
      <div ref={ctaRef} className="text-center">
        <div className="cta-content bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 md:p-12">
          <h2 className="select-none pointer-events-none text-3xl md:text-4xl font-bold text-white mb-4">
            Ready to Get Started?
          </h2>
          <p className="select-none pointer-events-none text-blue-100 text-lg mb-8 max-w-2xl mx-auto">
            Join healthcare professionals who trust our AI-powered autism detection technology 
            for early intervention support.
          </p>
          <Button size="lg" variant="secondary" className="text-lg px-8 py-6" asChild>
            <Link href="/sign-up">
              Create Free Account
              <ArrowRight className="ml-2 h-5 w-5" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
