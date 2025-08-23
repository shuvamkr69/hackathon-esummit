"use client";

import React, { useRef } from "react";
import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { HoverBorderGradient } from "./ui/hover-border-gradient";
import { GoogleGeminiEffect } from "./ui/google-gemini-effect";
import { useScroll, useTransform } from "motion/react";
import BlurText from "./ui/BlurText";

export default function MainHero() {
  const heroRef = useRef<HTMLDivElement>(null);
  //const ref = React.useRef(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  });

  const pathLengthFirst = useTransform(scrollYProgress, [0, 0.8], [0.2, 1.2]);
  const pathLengthSecond = useTransform(scrollYProgress, [0, 0.8], [0.15, 1.2]);
  const pathLengthThird = useTransform(scrollYProgress, [0, 0.8], [0.1, 1.2]);
  const pathLengthFourth = useTransform(scrollYProgress, [0, 0.8], [0.05, 1.2]);
  const pathLengthFifth = useTransform(scrollYProgress, [0, 0.8], [0, 1.2]);

  useGSAP(() => {
    const tl = gsap.timeline();

    tl.fromTo(
      ".hero-title",
      { y: 100, opacity: 0 },
      { y: 0, opacity: 1, duration: 1, ease: "power3.out" }
    )
      .fromTo(
        ".hero-subtitle",
        { y: 50, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.8, ease: "power3.out" },
        "-=0.5"
      )
      .fromTo(
        ".hero-buttons",
        { y: 30, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.6, ease: "power3.out" },
        "-=0.3"
      );
  }, []);

  const handleAnimationComplete = () => {
    console.log("Animation completed!");
  };

  return (
    <div ref={heroRef} className="relative overflow-hidden">
      {/* Background Gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-blue-950/20 dark:via-indigo-950/20 dark:to-purple-950/20" />

      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-400/10 rounded-full blur-3xl animate-pulse" />
        <div
          className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-400/10 rounded-full blur-3xl animate-pulse"
          style={{ animationDelay: "1s" }}
        />
      </div>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-32 pb-20">
        {/* Main Hero Content */}
        <div className="text-center space-y-20 ">
          <h1 className="hero-title text-4xl md:text-6xl lg:text-7xl font-normal">
            <BlurText
                text="Early Autism Detection"
                delay={150}
                animateBy="words"
                direction="top"
                onAnimationComplete={handleAnimationComplete}
                className="hero-title text-4xl md:text-6xl lg:text-7xl font-normal justify-center items-center align-middle"
              />
            <span className="pointer-events-none select-none bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
              
              <GoogleGeminiEffect description="Advanced computer vision technology analyzes classroom videos to identify early autism indicators, 
            supporting healthcare professionals in early intervention strategies."
                pathLengths={[
                  pathLengthFirst,
                  pathLengthSecond,
                  pathLengthThird,
                  pathLengthFourth,
                  pathLengthFifth,
                ]}
              />
            </span>
            <br />
            {/* <span className="pointer-events-none select-none text-foreground">with AI Precision</span> */}
          </h1>

          {/* <p className="select-none pointer-events-none hero-subtitle text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Advanced computer vision technology analyzes classroom videos to identify early autism indicators, 
            supporting healthcare professionals in early intervention strategies.
          </p> */}

          <div className="hero-buttons flex flex-col sm:flex-row justify-center items-center">
            {/* <Button size="lg" className="text-lg px-8 py-6" asChild>
              <Link href="/dashboard">
                Start Analysis
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </Button> */}
            <HoverBorderGradient as="button" className="flex text-lg px-8 py-6">
              <Link href="/dashboard">
                Start Analysis
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </HoverBorderGradient>

            {/* <Button variant="outline" size="lg" className="text-lg px-8 py-6" asChild>
              <Link href="#demo">
                <Play className="mr-2 h-5 w-5" />
                Watch Demo
              </Link>
            </Button> */}
          </div>
        </div>
      </div>
    </div>
  );
}
