'use client';

import React, { useState, useCallback, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Upload, Play, Pause, RotateCcw, Shield, AlertTriangle, CheckCircle } from 'lucide-react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import axios from 'axios';

interface AnalysisResult {
  confidence: number;
  indicators: string[];
  timestamp: string;
  frameAnalysis: {
    eyeContact: number;
    facialExpressions: number;
  };
  tips?: string[];
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

      const url = URL.createObjectURL(file);
      setVideoPreview(url);

      setAnalysisResult(null);
      setProgress(0);

      if (progressRef.current) {
        gsap.fromTo(progressRef.current,
          { opacity: 0, x: -50 },
          { opacity: 1, x: 0, duration: 0.5, ease: "power2.out" }
        );
      }
    }
  }, []);

  const uploadAndAnalyze = useCallback(async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setProgress(10);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await axios.post("http://13.126.67.35:8000/analyze", formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000,
        withCredentials: false,
      });

      const result = response.data;
      setProgress(100);
      console.log("Analysis result:", result);

      // ----------------------------
      // 🔹 Summarize API response
      // ----------------------------
      let avgConfidence = 0;
      let majorityClass = "Unknown";

      if (result.autism_predictions?.length) {
        avgConfidence =
          result.autism_predictions.reduce(
            (sum: number, p: { confidence: number }) => sum + p.confidence,
            0
          ) / result.autism_predictions.length;

        const autisticCount = result.autism_predictions.filter(
          (p: { class: string }) => p.class === "Autistic"
        ).length;
        majorityClass =
          autisticCount > result.autism_predictions.length / 2
            ? "Autistic"
            : "Non-Autistic";
      }

      // Emotions → facial expressions score
      let facialScore = 0;
      if (result.emotions?.length) {
        const emotionCounts: Record<string, number> = {};
        result.emotions.forEach((e: string) => {
          emotionCounts[e] = (emotionCounts[e] || 0) + 1;
        });
        const variety = Object.keys(emotionCounts).length;
        facialScore = Math.min(100, (variety / result.emotions.length) * 100);
      }

      // Gaze → eye contact score
      let eyeContactScore = 0;
      if (result.gaze?.length) {
        const detected = result.gaze.filter(
          (g: string) => g !== "No face detected"
        ).length;
        eyeContactScore = (detected / result.gaze.length) * 100;
      }

      // ----------------------------
      // 🔹 Handle visibility logic
      // ----------------------------
      const indicators: string[] = [];
      const tips: string[] = [];

      if (eyeContactScore < 50) {
        indicators.push("Face not visible enough for reliable analysis.");
      } else {
        indicators.push(`Classified as: ${majorityClass}`);
        indicators.push(`Eye contact score: ${eyeContactScore.toFixed(1)}%`);
        indicators.push(`Facial expressiveness score: ${facialScore.toFixed(1)}%`);

        if (avgConfidence * 100 > 70 && majorityClass === "Autistic") {
          tips.push("Encourage more consistent eye contact during interactions.");
          tips.push("Engage in activities that promote varied facial expressions (e.g., role-play, games).");
        }
      }

      // ----------------------------
      // 🔹 Map to frontend interface
      // ----------------------------
      const mappedResult: AnalysisResult = {
        confidence: eyeContactScore < 50 ? 0 : avgConfidence * 100,
        indicators,
        tips,
        timestamp: new Date().toISOString(),
        frameAnalysis: {
          eyeContact: eyeContactScore,
          facialExpressions: facialScore,
        }
      };

      setAnalysisResult(mappedResult);
    } catch (error) {
      console.error("Analysis failed:", error);
      if (axios.isAxiosError(error)) {
        if (error.code === 'ERR_NETWORK') {
          alert("Cannot connect to analysis server. Please ensure the server is running");
        } else if (error.response) {
          alert(`Server error: ${error.response.status} - ${error.response.data?.message || 'Unknown error'}`);
        } else {
          alert("Request timeout. Please try with a smaller video file.");
        }
      } else {
        alert("Analysis failed. Please try again.");
      }
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedFile]);

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
    <div className="max-w-4xl mx-auto space-y-6 p-6 pt-30">
      {/* Privacy Notice */}
      <Card className="border-green-200 bg-green-50/50 dark:bg-green-950/20 dark:border-green-800">
        <CardContent className="pt-6">
          <div className="flex items-center gap-3">
            <Shield className="text-green-600 h-6 w-6" />
            <div>
              <h3 className="font-semibold text-green-800 dark:text-green-200">Privacy Protected</h3>
              <p className="text-green-700 dark:text-green-300 text-sm">
                Videos are processed locally and discarded after analysis. No data is stored permanently.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Upload Card */}
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

              {/* Controls */}
              <div className="flex flex-col sm:flex-row gap-2">
                <Button
                  onClick={uploadAndAnalyze}
                  disabled={isAnalyzing}
                  className="flex-1 w-full sm:w-auto text-sm sm:text-base py-2 sm:py-3 px-4 sm:px-6"
                >
                  {isAnalyzing ? 'Analyzing...' : 'Start Analysis'}
                </Button>
                <Button 
                  variant="outline" 
                  onClick={resetAnalysis}
                  className="w-full sm:w-auto text-sm sm:text-base py-2 sm:py-3 px-4 sm:px-6"
                >
                  <RotateCcw className="h-4 w-4 mr-2 sm:mr-0" />
                  <span className="sm:hidden">Reset</span>
                  <span className="hidden sm:inline">Reset</span>
                </Button>
              </div>
            </div>
          )}

          {/* Progress */}
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

      {/* Results */}
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
            {/* Confidence */}
            {analysisResult.confidence > 0 ? (
              <div className="text-center p-6 bg-muted/50 rounded-lg">
                <div className="text-3xl font-bold mb-2">
                  {analysisResult.confidence.toFixed(1)}%
                </div>
                <div className="text-sm text-muted-foreground">
                  Confidence Level for Early Indicators
                </div>
              </div>
            ) : (
              <div className="text-center p-6 bg-muted/50 rounded-lg text-red-600 font-medium">
                Face not visible enough for reliable analysis.
              </div>
            )}

            {/* Frame Analysis */}
            <div className="grid grid-cols-2 gap-4">
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

            {/* Indicators */}
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

            {/* Tips (if any) */}
            {analysisResult.tips && analysisResult.tips.length > 0 && (
              <div>
                <h4 className="font-semibold mb-3">Supportive Tips</h4>
                <ul className="list-disc list-inside space-y-1 text-sm text-green-700 dark:text-green-300">
                  {analysisResult.tips.map((tip, idx) => (
                    <li key={idx}>{tip}</li>
                  ))}
                </ul>
              </div>
            )}

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
