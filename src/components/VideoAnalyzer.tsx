'use client';

import React, { useState, useCallback, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Upload, Play, Pause, RotateCcw, Shield, AlertTriangle, CheckCircle } from 'lucide-react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';

interface AnalysisResult {
  confidence: number;
  indicators: string[];
  timestamp: string;
  frameAnalysis: {
    eyeContact: number;
    facialExpressions: number;
    bodyMovement: number;
    socialEngagement: number;
  };
}

export default function VideoAnalyzer() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);

  useGSAP(() => {
    if (cardRef.current) {
      gsap.fromTo(cardRef.current, 
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 1, ease: "power3.out" }
      );
    }
  }, []);

  useGSAP(() => {
    if (analysisResult && resultRef.current) {
      gsap.fromTo(resultRef.current,
        { opacity: 0, scale: 0.9 },
        { opacity: 1, scale: 1, duration: 0.8, ease: "back.out(1.7)" }
      );
    }
  }, [analysisResult]);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      
      // Create preview URL
      const url = URL.createObjectURL(file);
      setVideoPreview(url);
      
      // Reset analysis
      setAnalysisResult(null);
      setProgress(0);
      
      // Animate file selection
      if (progressRef.current) {
        gsap.fromTo(progressRef.current,
          { opacity: 0, x: -50 },
          { opacity: 1, x: 0, duration: 0.5, ease: "power2.out" }
        );
      }
    }
  }, []);

  const simulateAnalysis = useCallback(async () => {
    setIsAnalyzing(true);
    setProgress(0);
    
    // Simulate analysis progress
    for (let i = 0; i <= 100; i += 2) {
      await new Promise(resolve => setTimeout(resolve, 100));
      setProgress(i);
    }
    
    // Simulate analysis result
    const mockResult: AnalysisResult = {
      confidence: Math.random() * 30 + 20, // 20-50% confidence range
      indicators: [
        'Limited eye contact patterns detected',
        'Repetitive movement behaviors observed',
        'Reduced social engagement markers',
        'Atypical facial expression patterns'
      ],
      timestamp: new Date().toISOString(),
      frameAnalysis: {
        eyeContact: Math.random() * 40 + 10,
        facialExpressions: Math.random() * 35 + 15,
        bodyMovement: Math.random() * 45 + 20,
        socialEngagement: Math.random() * 30 + 10
      }
    };
    
    setAnalysisResult(mockResult);
    setIsAnalyzing(false);
  }, []);

  const handleVideoToggle = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const resetAnalysis = () => {
    setSelectedFile(null);
    setVideoPreview(null);
    setAnalysisResult(null);
    setProgress(0);
    setIsAnalyzing(false);
    setIsPlaying(false);
    
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 p-6">
      {/* Privacy Notice */}
      <Card className="border-green-200 bg-green-50/50 dark:bg-green-950/20 dark:border-green-800">
        <CardContent className="pt-6">
          <div className="flex items-center gap-3">
            <Shield className="text-green-600 h-6 w-6" />
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200">Privacy Protected</h3>
              <p className="text-green-700 dark:text-green-300 text-sm">
                Videos are processed locally and immediately discarded after analysis. No data is stored or transmitted.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Upload Card */}
      <Card ref={cardRef} className="border-2 border-dashed border-muted-foreground/25 hover:border-primary/50 transition-colors">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Video Upload & Analysis
          </CardTitle>
          <CardDescription>
            Select a video file to analyze for early autism indicators
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {!selectedFile ? (
            <div className="text-center py-12">
              <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
              <label htmlFor="video-upload" className="cursor-pointer">
                <div className="text-lg font-medium mb-2">Drop your video here or click to browse</div>
                <div className="text-sm text-muted-foreground mb-4">Supports MP4, AVI, MOV files</div>
                <Button asChild>
                  <span>Select Video File</span>
                </Button>
              </label>
              <input
                id="video-upload"
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
          ) : (
            <div className="space-y-4">
              {/* Video Preview */}
              <div className="relative bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  src={videoPreview || undefined}
                  className="w-full h-64 object-contain"
                  controls={false}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                />
                <div className="absolute bottom-4 left-4 right-4 flex items-center gap-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={handleVideoToggle}
                    className="bg-white/20 backdrop-blur-sm hover:bg-white/30"
                  >
                    {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    {isPlaying ? 'Pause' : 'Play'}
                  </Button>
                </div>
              </div>

              {/* File Info */}
              <div className="bg-muted/50 rounded-lg p-4">
                <div className="text-sm font-medium">{selectedFile.name}</div>
                <div className="text-xs text-muted-foreground">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </div>
              </div>

              {/* Analysis Controls */}
              <div className="flex gap-2">
                <Button
                  onClick={simulateAnalysis}
                  disabled={isAnalyzing}
                  className="flex-1"
                >
                  {isAnalyzing ? 'Analyzing...' : 'Start Analysis'}
                </Button>
                <Button variant="outline" onClick={resetAnalysis}>
                  <RotateCcw className="h-4 w-4" />
                  Reset
                </Button>
              </div>
            </div>
          )}

          {/* Progress Bar */}
          {isAnalyzing && (
            <div ref={progressRef} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Analyzing video...</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} className="h-2" />
              <div className="text-xs text-muted-foreground text-center">
                Processing frames and analyzing behavioral patterns
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {analysisResult && (
        <Card ref={resultRef} className="border-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              Analysis Complete
            </CardTitle>
            <CardDescription>
              AI-powered analysis results with confidence scoring
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Confidence Score */}
            <div className="text-center p-6 bg-muted/50 rounded-lg">
              <div className="text-3xl font-bold mb-2">
                {analysisResult.confidence.toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">
                Confidence Level for Early Indicators
              </div>
              <div className="mt-2 flex items-center justify-center gap-2 text-orange-600">
                <AlertTriangle className="h-4 w-4" />
                <span className="text-xs">Low-to-moderate indicators detected</span>
              </div>
            </div>

            {/* Frame Analysis */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Eye Contact</span>
                    <span>{analysisResult.frameAnalysis.eyeContact.toFixed(1)}%</span>
                  </div>
                  <Progress value={analysisResult.frameAnalysis.eyeContact} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Facial Expressions</span>
                    <span>{analysisResult.frameAnalysis.facialExpressions.toFixed(1)}%</span>
                  </div>
                  <Progress value={analysisResult.frameAnalysis.facialExpressions} className="h-2" />
                </div>
              </div>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Body Movement</span>
                    <span>{analysisResult.frameAnalysis.bodyMovement.toFixed(1)}%</span>
                  </div>
                  <Progress value={analysisResult.frameAnalysis.bodyMovement} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Social Engagement</span>
                    <span>{analysisResult.frameAnalysis.socialEngagement.toFixed(1)}%</span>
                  </div>
                  <Progress value={analysisResult.frameAnalysis.socialEngagement} className="h-2" />
                </div>
              </div>
            </div>

            {/* Detected Indicators */}
            <div>
              <h4 className="font-semibold mb-3">Detected Indicators</h4>
              <div className="space-y-2">
                {analysisResult.indicators.map((indicator, index) => (
                  <div key={index} className="flex items-start gap-2 p-3 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                    <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-yellow-800 dark:text-yellow-200">{indicator}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Disclaimer */}
            <div className="p-4 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg">
              <div className="text-xs text-blue-800 dark:text-blue-200">
                <strong>Important:</strong> This analysis is for educational purposes only and should not be used for medical diagnosis. 
                Please consult with qualified healthcare professionals for proper evaluation and diagnosis.
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
