"use client";

import React from "react";
import { motion } from "motion/react";
import { LampContainer } from "@/components/ui/lamp";
import { Card, CardContent } from "@/components/ui/card";
import { Brain, Heart, Shield, Users, Award, Target } from "lucide-react";
import { GlowingEffect } from "./ui/glowing-effect";

export default function AboutContent() {
  return (
    <div className="min-h-screen">
      {/* Hero Section with Lamp Effect */}
      <LampContainer className="min-h-screen">
        <motion.div
          initial={{ opacity: 0.5, y: 100 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{
            delay: 0.3,
            duration: 0.8,
            ease: "easeInOut",
          }}
          className="mt-8 text-center space-y-6"
        >
          <h1 className="bg-gradient-to-br from-slate-300 to-slate-500 py-4 bg-clip-text text-4xl font-medium tracking-tight text-transparent md:text-7xl">
            About Our Mission
          </h1>
          <p className="text-slate-400 text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
            Pioneering early autism detection through advanced AI technology to support 
            healthcare professionals in providing timely interventions and better outcomes.
          </p>
        </motion.div>
      </LampContainer>

      {/* Content Sections */}
      <div className="bg-background py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-20">
          
          {/* Mission & Vision */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center space-y-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold">Our Purpose</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <Card className="border-2 hover:border-blue-500/50 transition-colors">
                <CardContent className="p-8 text-center space-y-4">
                  <Target className="h-12 w-12 mx-auto text-blue-600" />
                  <h3 className="text-2xl font-semibold">Mission</h3>
                  <p className="text-muted-foreground">
                    To democratize early autism detection by providing healthcare professionals 
                    with cutting-edge AI tools that analyze behavioral patterns in classroom 
                    environments, enabling earlier interventions and better life outcomes.
                  </p>
                </CardContent>
              </Card>

              <Card className="border-2 hover:border-purple-500/50 transition-colors">
                <CardContent className="p-8 text-center space-y-4">
                  <Award className="h-12 w-12 mx-auto text-purple-600" />
                  <h3 className="text-2xl font-semibold">Vision</h3>
                  <p className="text-muted-foreground">
                    A world where every child has access to early autism screening, 
                    where technology bridges the gap between detection and intervention, 
                    and where families receive support when they need it most.
                  </p>
                </CardContent>
              </Card>
            </div>
          </motion.div>

          {/* Core Values */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="space-y-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-center">Our Values</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <Card className="border-2 hover:border-green-500/50 transition-colors">
                <CardContent className="p-6 text-center space-y-4">
                  <Shield className="h-10 w-10 mx-auto text-green-600" />
                  <h3 className="text-xl font-semibold">Privacy First</h3>
                  <p className="text-muted-foreground text-sm">
                    Complete data privacy with local processing. No storage, 
                    no transmission - your data stays with you.
                  </p>
                </CardContent>
              </Card>

              <Card className="border-2 hover:border-blue-500/50 transition-colors">
                <CardContent className="p-6 text-center space-y-4">
                  <Brain className="h-10 w-10 mx-auto text-blue-600" />
                  <h3 className="text-xl font-semibold">Scientific Rigor</h3>
                  <p className="text-muted-foreground text-sm">
                    Evidence-based AI models trained on peer-reviewed research 
                    and validated datasets.
                  </p>
                </CardContent>
              </Card>

              <Card className="border-2 hover:border-red-500/50 transition-colors">
                <CardContent className="p-6 text-center space-y-4">
                  <Heart className="h-10 w-10 mx-auto text-red-600" />
                  <h3 className="text-xl font-semibold">Ethical AI</h3>
                  <p className="text-muted-foreground text-sm">
                    Supporting healthcare professionals, not replacing human 
                    judgment and clinical expertise.
                  </p>
                </CardContent>
              </Card>
            </div>
          </motion.div>

          {/* Technology Approach */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="space-y-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-center">Our Technology</h2>
            
            <div className="bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 rounded-2xl p-8 md:p-12">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                <div className="space-y-6">
                  <h3 className="text-2xl font-semibold">Advanced Computer Vision</h3>
                  <p className="text-muted-foreground">
                    Our AI system analyzes classroom videos using multi-modal deep learning 
                    to detect early autism indicators through:
                  </p>
                  <ul className="space-y-3 text-muted-foreground">
                    <li className="flex items-center gap-3">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      Facial expression pattern recognition
                    </li>
                    <li className="flex items-center gap-3">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      Social interaction behavior analysis
                    </li>
                    <li className="flex items-center gap-3">
                      <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                      Eye contact and gaze pattern tracking
                    </li>
                    <li className="flex items-center gap-3">
                      <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                      Repetitive movement detection
                    </li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-slate-700 rounded-xl p-6 shadow-lg">
                  <h4 className="font-semibold mb-4 text-center">Analysis Pipeline</h4>
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-semibold">1</div>
                      <span className="text-sm">Video Frame Extraction</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center text-sm font-semibold">2</div>
                      <span className="text-sm">Feature Detection & Analysis</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-semibold">3</div>
                      <span className="text-sm">Pattern Recognition</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center text-sm font-semibold">4</div>
                      <span className="text-sm">Confidence Scoring</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Team & Contact */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-center space-y-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold">Get Involved</h2>
            
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 md:p-12 text-white">
              <Users className="h-16 w-16 mx-auto mb-6" />
              <h3 className="text-2xl font-semibold mb-4">Join Our Mission</h3>
              <p className="text-blue-100 mb-8 max-w-2xl mx-auto">
                Whether you're a healthcare professional, researcher, or advocate, 
                we welcome collaboration to improve early autism detection and intervention.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Card className="bg-white/10 border-white/20">
                  <CardContent className="p-4 text-center">
                    <h4 className="font-semibold mb-2">Healthcare Professionals</h4>
                    <p className="text-sm text-blue-100">
                      Partner with us to validate and improve our detection algorithms
                    </p>
                  </CardContent>
                </Card>
                <Card className="bg-white/10 border-white/20">
                  <CardContent className="p-4 text-center">
                    <h4 className="font-semibold mb-2">Researchers</h4>
                    <p className="text-sm text-blue-100">
                      Collaborate on studies and contribute to our open research initiatives
                    </p>
                  </CardContent>
                </Card>
              </div>
            </div>
          </motion.div>

          {/* Disclaimer */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6"
          >
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-3">
              Important Medical Disclaimer
            </h3>
            <p className="text-yellow-700 dark:text-yellow-300 text-sm">
              This technology is designed as a screening tool to assist healthcare professionals 
              and is not intended for self-diagnosis or as a replacement for professional medical 
              evaluation. Always consult qualified healthcare providers for proper autism assessment 
              and diagnosis. Early detection tools should complement, not replace, clinical judgment.
            </p>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
